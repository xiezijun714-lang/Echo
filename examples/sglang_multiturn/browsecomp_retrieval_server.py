"""
BrowseComp-Plus retrieval server — supports BM25 and Qwen3-Embedding dense retrieval.

Usage:
    # BM25 (original, backward-compat)
    python browsecomp_retrieval_server.py --mode bm25 --port 8000

    # Dense retrieval (Qwen3-Embedding)
    python browsecomp_retrieval_server.py --mode dense --model Qwen/Qwen3-Embedding \
        --device cuda:7 --port 8000

Endpoints:
    POST /retrieve  {"queries": [...], "topk": 3}
    POST /embed     {"texts": [...], "instruction": ""}   # for reward semantic judge
    GET  /health
"""

import argparse
import asyncio
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Corpus loading (shared between BM25 and dense)
# ---------------------------------------------------------------------------

def extract_title(text: str, url: str = "") -> str:
    m = re.search(r"^---\s*\ntitle:\s*(.+?)\n", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    if url:
        return url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return "Unknown"


def build_corpus_from_file(corpus_file: str):
    """Load corpus from an official BC-Plus parquet file (HuggingFace format).

    Expected columns: docid (str), text (str), url (str).
    Download from: https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus
    """
    print(f"Loading corpus from file: {corpus_file} ...", flush=True)
    if corpus_file.endswith(".jsonl"):
        import json
        docs = []
        with open(corpus_file, "r") as f:
            for line in f:
                doc = json.loads(line.strip())
                if doc.get("docid") and doc.get("text"):
                    docs.append(doc)
    else:
        df = pd.read_parquet(corpus_file)
        docs = []
        for _, row in df.iterrows():
            docid = str(row.get("docid", ""))
            text = str(row.get("text", ""))
            url = str(row.get("url", ""))
            if docid and text:
                docs.append({"docid": docid, "text": text, "url": url})

    # Deduplicate by docid
    seen = {}
    for doc in docs:
        docid = doc["docid"]
        if docid not in seen:
            seen[docid] = doc
    print(f"  Loaded {len(seen)} unique documents from file.", flush=True)
    return seen


def build_corpus(data_dir: str):
    print("Loading parquet files ...", flush=True)
    dfs = []
    for split in ["train.parquet", "test.parquet"]:
        path = f"{data_dir}/{split}"
        try:
            dfs.append(pd.read_parquet(path))
        except FileNotFoundError:
            print(f"  Warning: {path} not found, skipping.", flush=True)
    if not dfs:
        raise RuntimeError(f"No parquet files found in {data_dir}")
    df = pd.concat(dfs, ignore_index=True)

    seen = {}
    for col in ["gold_docs", "negative_docs", "evidence_docs"]:
        if col not in df.columns:
            continue
        for row_docs in df[col].dropna():
            if row_docs is None:
                continue
            for doc in row_docs:
                if not isinstance(doc, dict):
                    continue
                docid = str(doc.get("docid", ""))
                if docid and docid not in seen:
                    seen[docid] = doc

    print(f"  Loaded {len(seen)} unique documents.", flush=True)
    return seen


# ---------------------------------------------------------------------------
# BM25 retriever (original implementation)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Retriever:
    def __init__(self, data_dir: str, cache_path: str = "/root/paddlejob/workspace/xzj/browsecomp_bm25_cache.pkl",
                 corpus_file: str = None):
        from rank_bm25 import BM25Okapi

        if cache_path and os.path.exists(cache_path):
            print(f"Loading BM25 index from cache: {cache_path} ...", flush=True)
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.docids = cached["docids"]
            self.docs = cached["docs"]
            self.bm25 = cached["bm25"]
            self.contents = cached["contents"]
            print(f"  Loaded {len(self.docids)} documents from cache.", flush=True)
        else:
            doc_map = build_corpus_from_file(corpus_file) if corpus_file else build_corpus(data_dir)
            self.docids = list(doc_map.keys())
            self.docs = [doc_map[d] for d in self.docids]

            print("Building BM25 index ...", flush=True)
            corpus_texts = [d.get("text", "") for d in self.docs]
            tokenized = [_tokenize(t) for t in corpus_texts]
            self.bm25 = BM25Okapi(tokenized)

            self.contents = []
            for doc in self.docs:
                title = extract_title(doc.get("text", ""), doc.get("url", ""))
                body = doc.get("text", "")
                self.contents.append(f"{title}\n{body}")

            if cache_path:
                print(f"Saving BM25 cache to {cache_path} ...", flush=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "docids": self.docids,
                        "docs": self.docs,
                        "bm25": self.bm25,
                        "contents": self.contents,
                    }, f)
        print("BM25 ready.", flush=True)

    def search(self, query: str, topk: int = 3):
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:topk]
        return [
            {"docid": self.docids[i], "document": {"contents": self.contents[i]}, "score": float(scores[i])}
            for i in top_idx
        ]

    def embed(self, texts: List[str], instruction: str = "") -> np.ndarray:
        raise NotImplementedError("BM25 mode does not support /embed endpoint")


# ---------------------------------------------------------------------------
# Dense retriever (Qwen3-Embedding + FAISS)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(
        self,
        data_dir: str,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cuda:7",
        cache_path: str = "/root/paddlejob/workspace/env_run/xzj/browsecomp_dense_cache.pkl",
        batch_size: int = 4,
        max_doc_length: int = 8192,
        corpus_file: str = None,
    ):
        self._device_str = device          # lazy: don't create torch.device yet
        self._model_name = model_name
        self.batch_size = batch_size
        self.max_doc_length = max_doc_length
        self._model_on_gpu = False
        self.device = None                 # set lazily on first GPU use
        self.model = None                  # set lazily or eagerly depending on cache

        has_cache = cache_path and os.path.exists(cache_path)

        if has_cache:
            # Cache exists: load model on CPU only, NO torch CUDA init at all.
            # This avoids creating a CUDA context that would steal ~1GB from
            # the GPU and cause SGLang CUDA graph capture to OOM.
            from transformers import AutoTokenizer, AutoModel
            import torch

            print(f"Loading embedding model {model_name} on cpu (cache hit) ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, dtype=torch.float32, trust_remote_code=True
            ).to("cpu").eval()
            print("  Model loaded on CPU.", flush=True)

            print(f"Loading dense index from cache: {cache_path} ...", flush=True)
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.docids = cached["docids"]
            self.contents = cached["contents"]
            embeddings = cached["embeddings"]  # (N, D) float32
            self._build_faiss(embeddings)
            print(f"  Loaded {len(self.docids)} documents from cache.", flush=True)
        else:
            # No cache: need GPU for document encoding — use ALL GPUs for speed
            import torch
            from transformers import AutoTokenizer, AutoModel

            self.device = torch.device(device)

            print(f"Loading embedding model {model_name} on ALL GPUs for encoding ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, dtype=torch.float16, trust_remote_code=True
            ).eval()

            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                # Wrapper: only return last_hidden_state to avoid gathering KV cache (OOM)
                class _HiddenStateOnly(torch.nn.Module):
                    def __init__(self, base):
                        super().__init__()
                        self.base = base
                    def forward(self, **kwargs):
                        return self.base(**kwargs).last_hidden_state
                self._raw_model = self.model
                self.model = torch.nn.DataParallel(_HiddenStateOnly(self.model)).cuda()
                self._dp_device = torch.device("cuda:0")
                print(f"  Using DataParallel on {n_gpus} GPUs.", flush=True)
            else:
                self.model = self.model.to(self.device)
                self._dp_device = self.device
                print(f"  Single GPU: {device}", flush=True)
            self._model_on_gpu = True
            # Use larger batch size with multi-GPU
            self._init_batch_size = batch_size * n_gpus

            doc_map = build_corpus_from_file(corpus_file) if corpus_file else build_corpus(data_dir)
            self.docids = list(doc_map.keys())
            docs = [doc_map[d] for d in self.docids]

            self.contents = []
            raw_texts = []
            for doc in docs:
                title = extract_title(doc.get("text", ""), doc.get("url", ""))
                body = doc.get("text", "")
                self.contents.append(f"{title}\n{body}")
                raw_texts.append(body)

            print(f"Encoding {len(raw_texts)} documents (batch_size={self._init_batch_size}) ...", flush=True)
            embeddings = self._encode_bulk(raw_texts, max_length=max_doc_length)
            self._build_faiss(embeddings)

            if cache_path:
                print(f"Saving dense cache to {cache_path} ...", flush=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "docids": self.docids,
                        "contents": self.contents,
                        "embeddings": embeddings,
                    }, f)

            # Offload: restore raw model, move to CPU for online queries
            if hasattr(self, '_raw_model'):
                self.model = self._raw_model
                del self._raw_model
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            self._model_on_gpu = False

        print("Dense retriever ready.", flush=True)

    # ------------------------------------------------------------------
    def _offload_model(self):
        """Move model to CPU and free GPU memory."""
        import torch
        if self._model_on_gpu:
            self.model = self.model.float().cpu()
            torch.cuda.empty_cache()
            self._model_on_gpu = False

    def _ensure_model_on_gpu(self):
        """Move model to GPU if not already there. Lazily creates CUDA device on first call."""
        import torch
        if self.device is None:
            self.device = torch.device(self._device_str)
        if not self._model_on_gpu:
            self.model = self.model.half().to(self.device)
            self._model_on_gpu = True

    # ------------------------------------------------------------------
    def _last_token_pool(self, last_hidden_states, attention_mask):
        import torch
        # Qwen3-Embedding uses decoder architecture -> last token pooling
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]

    # ------------------------------------------------------------------
    def _encode_bulk(self, texts: List[str], max_length: int = 8192) -> np.ndarray:
        """Multi-GPU bulk encoding for initial document indexing."""
        import torch
        import torch.nn.functional as F

        device = self._dp_device
        bs = self._init_batch_size
        all_embs = []
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            enc = self.tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                hidden = self.model(**enc)  # wrapper returns last_hidden_state directly
            emb = self._last_token_pool(hidden, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().float().numpy())
            if (i // bs) % 20 == 0:
                print(f"  Encoded {i + len(batch)}/{len(texts)}", flush=True)
        return np.vstack(all_embs)

    def _encode(self, texts: List[str], instruction: Optional[str], max_length: int = 512) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if instruction:
            texts = [f"Instruct: {instruction}\n<|embed|>\n{t}" for t in texts]

        self._ensure_model_on_gpu()
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self.tokenizer(
                batch,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            emb = self._last_token_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().float().numpy())
            if (i // self.batch_size) % 50 == 0:
                print(f"  Encoded {i + len(batch)}/{len(texts)}", flush=True)
        # Keep model on GPU after first query for fast subsequent queries.
        # Model was loaded on CPU at startup to avoid interfering with SGLang CUDA graph capture.
        return np.vstack(all_embs)

    def _build_faiss(self, embeddings: np.ndarray):
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity (vectors are L2-normalized)
        index.add(embeddings)
        self.index = index
        self.dim = dim
        print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}", flush=True)

    # ------------------------------------------------------------------
    def search(self, query: str, topk: int = 3) -> List[dict]:
        query_emb = self._encode(
            [query],
            instruction="Given a question, retrieve relevant passages that help answer the question",
            max_length=512,
        )  # (1, D)
        scores, indices = self.index.search(query_emb, topk)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "docid": self.docids[idx],
                "document": {"contents": self.contents[idx]},
                "score": float(score),
            })
        return results

    def embed(self, texts: List[str], instruction: str = "") -> np.ndarray:
        """Public method used by /embed endpoint (for reward semantic judge)."""
        return self._encode(texts, instruction=instruction or None, max_length=512)


# ---------------------------------------------------------------------------
# BM25 process-pool workers (only used in BM25 mode)
# ---------------------------------------------------------------------------

_worker_bm25_state = None


def _pool_init(cache_path):
    global _worker_bm25_state
    with open(cache_path, "rb") as f:
        _worker_bm25_state = pickle.load(f)
    print(f"  [Worker PID {os.getpid()}] Loaded {len(_worker_bm25_state['docids'])} docs.", flush=True)


def _pool_search(args):
    query, topk = args
    tokens = _tokenize(query)
    scores = _worker_bm25_state["bm25"].get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:topk]
    return [
        {"document": {"contents": _worker_bm25_state["contents"][i]}, "score": float(scores[i])}
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()
retriever = None          # BM25Retriever or DenseRetriever
search_pool = None        # ProcessPoolExecutor (BM25 only)
_bm25_cache_path = None   # for pool workers
_search_lock = asyncio.Lock()  # serialize GPU queries in dense mode


class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: int = 3
    return_scores: bool = True


class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = ""


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    if search_pool is not None:
        # BM25 mode: offload to process pool
        loop = asyncio.get_event_loop()
        args = [(q, req.topk) for q in req.queries]
        futures = [loop.run_in_executor(search_pool, _pool_search, a) for a in args]
        result = await asyncio.gather(*futures)
        return {"result": list(result)}
    else:
        # Dense mode: GPU search (serialize to avoid OOM)
        async with _search_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: [retriever.search(q, req.topk) for q in req.queries],
            )
        return {"result": result}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Return L2-normalized embeddings for reward semantic judgment."""
    if not hasattr(retriever, "embed"):
        return {"error": "embed not supported in BM25 mode", "embeddings": None}
    async with _search_lock:
        loop = asyncio.get_event_loop()
        embs = await loop.run_in_executor(
            None, lambda: retriever.embed(req.texts, req.instruction)
        )
    return {"embeddings": embs.tolist()}


class GetDocRequest(BaseModel):
    docid: str


@app.post("/get_doc")
async def get_doc(req: GetDocRequest):
    """Return full document content by docid."""
    if retriever is None:
        return {"error": "retriever not initialized", "document": None}
    try:
        idx = retriever.docids.index(req.docid)
    except ValueError:
        return {"error": f"docid '{req.docid}' not found", "document": None}
    return {"document": {"contents": retriever.contents[idx]}, "docid": req.docid}


# ---------------------------------------------------------------------------
# Sub-document chunk retrieval (second-stage retrieval for open_page)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 512       # tokens per chunk (rough: 1 token ≈ 4 chars)
_CHUNK_OVERLAP = 64     # overlap tokens between adjacent chunks
_CHARS_PER_TOKEN = 4


def _split_into_chunks(text: str, chunk_size: int = _CHUNK_SIZE,
                       overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks by approximate token count."""
    char_chunk = chunk_size * _CHARS_PER_TOKEN
    char_overlap = overlap * _CHARS_PER_TOKEN
    step = max(char_chunk - char_overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start: start + char_chunk]
        if chunk.strip():
            chunks.append(chunk)
        if start + char_chunk >= len(text):
            break
    return chunks


class GetDocChunksRequest(BaseModel):
    docid: str
    query: str
    topk: int = 3
    chunk_size: int = _CHUNK_SIZE
    chunk_overlap: int = _CHUNK_OVERLAP


def _bm25_rank_chunks(query: str, chunks: List[str], topk: int) -> List[tuple]:
    """Rank *chunks* against *query* using BM25.

    Returns a list of (chunk_index, score, chunk_text) sorted by score desc.
    No GPU, no lock, runs in < 10 ms.
    """
    from math import log

    # Tokenize
    q_tokens = _tokenize(query)
    if not q_tokens:
        # Fallback: return first topk chunks in order
        return [(i, 0.0, chunks[i]) for i in range(min(topk, len(chunks)))]

    chunk_tokens = [_tokenize(c) for c in chunks]
    n = len(chunks)
    avg_dl = sum(len(ct) for ct in chunk_tokens) / max(n, 1)

    # BM25 parameters
    k1 = 1.2
    b = 0.75

    # Document frequency for query terms
    df = {}
    for t in q_tokens:
        df[t] = sum(1 for ct in chunk_tokens if t in set(ct))

    # Score each chunk
    scores = []
    for i, ct in enumerate(chunk_tokens):
        tf_map = {}
        for t in ct:
            tf_map[t] = tf_map.get(t, 0) + 1
        dl = len(ct)
        score = 0.0
        for t in q_tokens:
            if t not in tf_map:
                continue
            tf = tf_map[t]
            idf = log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        scores.append((i, score, chunks[i]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]


@app.post("/get_doc_chunks")
async def get_doc_chunks(req: GetDocChunksRequest):
    """Sub-document retrieval: split a document into chunks and rank them
    against *query* using BM25.  No GPU needed — does not block /retrieve.
    """
    if retriever is None:
        return {"error": "retriever not initialized"}

    # Locate the document
    try:
        idx = retriever.docids.index(req.docid)
    except ValueError:
        return {"error": f"docid '{req.docid}' not found"}

    full_text = retriever.contents[idx]
    title = full_text.split("\n")[0]
    body = "\n".join(full_text.split("\n")[1:])

    # Split into chunks
    chunks = _split_into_chunks(body, req.chunk_size, req.chunk_overlap)
    if not chunks:
        return {"error": "document body is empty", "docid": req.docid}

    # BM25 ranking — pure CPU, no lock, < 10 ms
    ranked = _bm25_rank_chunks(req.query, chunks, min(req.topk, len(chunks)))

    results = []
    for chunk_idx, score, text in ranked:
        results.append({
            "chunk_index": chunk_idx,
            "score": round(score, 4),
            "text": text,
        })

    return {
        "docid": req.docid,
        "title": title,
        "total_chunks": len(chunks),
        "chunks": results,
    }


@app.get("/health")
def health():
    mode = "dense" if search_pool is None else "bm25"
    n = len(retriever.docids) if retriever else 0
    return {"status": "ok", "mode": mode, "num_docs": n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global retriever, search_pool, _bm25_cache_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--data_dir", default="/root/paddlejob/workspace/xzj/dataset/browsecomp-plus-processed")
    parser.add_argument("--corpus_file", default=None,
                        help="Path to official BC-Plus corpus file (parquet/jsonl). "
                             "If set, overrides --data_dir. Download from: "
                             "https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # BM25 options
    parser.add_argument("--pool_workers", type=int, default=8)
    parser.add_argument("--bm25_cache", default="/root/paddlejob/workspace/xzj/browsecomp_bm25_cache.pkl")
    # Dense options
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dense_cache", default="/root/paddlejob/workspace/xzj/browsecomp_dense_cache.pkl")
    args = parser.parse_args()

    if args.mode == "bm25":
        retriever = BM25Retriever(args.data_dir, cache_path=args.bm25_cache, corpus_file=args.corpus_file)
        _bm25_cache_path = args.bm25_cache
        print(f"Starting process pool with {args.pool_workers} workers ...", flush=True)
        search_pool = ProcessPoolExecutor(
            max_workers=args.pool_workers,
            initializer=_pool_init,
            initargs=(args.bm25_cache,),
        )
        list(search_pool.map(_pool_search, [("warmup", 1)] * args.pool_workers))
        print(f"BM25 process pool ready.", flush=True)
    else:
        try:
            import faiss  # noqa: F401
        except ImportError:
            print("ERROR: faiss not installed. Run:", flush=True)
            print("  pip install faiss-gpu   # (GPU, recommended)", flush=True)
            print("  pip install faiss-cpu   # (CPU fallback)", flush=True)
            sys.exit(1)
        retriever = DenseRetriever(
            args.data_dir,
            model_name=args.model,
            device=args.device,
            cache_path=args.dense_cache,
            batch_size=args.batch_size,
            corpus_file=args.corpus_file,
        )
        # No process pool for dense mode

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
