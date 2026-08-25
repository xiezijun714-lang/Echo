#!/usr/bin/env python3
"""Apply Context-Folding's released BrowseComp-Plus split to local full rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


UPSTREAM_REPOSITORY = "https://github.com/MiaoLu3/Context_Folding"
UPSTREAM_COMMIT = "81626e97bb5f4eaa79edca045528829ce8850f5b"
UPSTREAM_BLOBS = {
    "bcp_train.parquet": "d540a683e1afa7082616f4c0d917f52eca0cd37a",
    "bcp_test.parquet": "87a12ae03c4fde71ae17d8b136049d595b15e8f3",
}
EXPECTED_COUNTS = {"train": 680, "test": 150}
EXPECTED_TEST_DIFFICULTY = {"easy": 50, "medium": 50, "hard": 50}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, with_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
    }
    if with_hash:
        record["sha256"] = sha256(path)
    return record


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_difficulty(data_source: str) -> str:
    suffix = data_source.rsplit("_", 1)[-1].lower()
    if suffix == "meduim":
        suffix = "medium"
    if suffix not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unknown Context-Folding difficulty label: {data_source!r}")
    return suffix


def read_author_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["data_source", "extra_info"])
    rows = table.to_pylist()
    for row in rows:
        extra = row.get("extra_info") or {}
        for key in ("query_id", "query", "answer"):
            if not extra.get(key):
                raise ValueError(f"{path} has a row without extra_info.{key}")
    return rows


def validate_upstream_file(path: Path) -> dict[str, Any]:
    expected_blob = UPSTREAM_BLOBS[path.name]
    actual_blob = git_blob_sha1(path)
    if actual_blob != expected_blob:
        raise ValueError(
            f"Git blob mismatch for {path}: expected {expected_blob}, got {actual_blob}"
        )
    record = file_record(path)
    record["git_blob_sha1"] = actual_blob
    return record


def load_local_rows(source_dir: Path) -> tuple[pa.Table, dict[str, dict[str, Any]]]:
    paths = [source_dir / "train.paper.parquet", source_dir / "test.paper.parquet"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    tables = [pq.read_table(path) for path in paths]
    if not tables[0].schema.equals(tables[1].schema, check_metadata=True):
        raise ValueError("Local train/test parquet schemas differ")

    combined = pa.concat_tables(tables)
    required = {"query_id", "query", "answer", "extra_info"}
    missing = required.difference(combined.column_names)
    if missing:
        raise ValueError(f"Local data is missing columns: {sorted(missing)}")

    index: dict[str, dict[str, Any]] = {}
    projected = combined.select(["query_id", "query", "answer", "extra_info"]).to_pylist()
    for position, row in enumerate(projected):
        query_id = row["query_id"]
        if query_id in index:
            raise ValueError(f"Duplicate local query_id: {query_id}")
        extra_query_id = (row.get("extra_info") or {}).get("query_id")
        if extra_query_id != query_id:
            raise ValueError(
                f"Local query_id mismatch: top-level={query_id!r}, extra_info={extra_query_id!r}"
            )
        index[query_id] = {
            "position": position,
            "query": row["query"],
            "answer": row["answer"],
        }
    return combined, index


def match_author_rows(
    rows: list[dict[str, Any]], local_index: dict[str, dict[str, Any]], split: str
) -> tuple[list[str], list[int]]:
    query_ids: list[str] = []
    positions: list[int] = []
    seen: set[str] = set()
    for row in rows:
        extra = row["extra_info"]
        query_id = extra["query_id"]
        if query_id in seen:
            raise ValueError(f"Duplicate {split} query_id in author split: {query_id}")
        seen.add(query_id)
        local = local_index.get(query_id)
        if local is None:
            raise ValueError(f"Author {split} query_id is absent locally: {query_id}")
        if extra["query"] != local["query"]:
            raise ValueError(f"Query text differs for query_id={query_id}")
        if extra["answer"] != local["answer"]:
            raise ValueError(f"Answer differs for query_id={query_id}")
        query_ids.append(query_id)
        positions.append(local["position"])
    return query_ids, positions


def write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(
        table,
        path,
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )


def verify_output(path: Path, expected_ids: list[str], schema: pa.Schema) -> None:
    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows != len(expected_ids):
        raise ValueError(
            f"Wrong row count in {path}: {parquet_file.metadata.num_rows} != {len(expected_ids)}"
        )
    if not parquet_file.schema_arrow.equals(schema, check_metadata=True):
        raise ValueError(f"Schema changed while writing {path}")
    actual_ids = pq.read_table(path, columns=["query_id"])["query_id"].to_pylist()
    if actual_ids != expected_ids:
        raise ValueError(f"query_id order/content mismatch in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--author-split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    author_dir = args.author_split_dir.resolve()
    output_dir = args.output_dir.resolve()
    dense_cache = args.dense_cache.resolve() if args.dense_cache else None

    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output directory: {output_dir}")
    if dense_cache is not None and not dense_cache.is_file():
        raise FileNotFoundError(dense_cache)

    author_paths = {
        "train": author_dir / "bcp_train.parquet",
        "test": author_dir / "bcp_test.parquet",
    }
    upstream_records = {
        split: validate_upstream_file(path) for split, path in author_paths.items()
    }
    author_rows = {split: read_author_rows(path) for split, path in author_paths.items()}
    for split, expected in EXPECTED_COUNTS.items():
        actual = len(author_rows[split])
        if actual != expected:
            raise ValueError(f"Wrong author {split} count: {actual} != {expected}")

    source_paths = [source_dir / "train.paper.parquet", source_dir / "test.paper.parquet"]
    source_stats_before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths
    }
    cache_stats_before = (
        (dense_cache.stat().st_size, dense_cache.stat().st_mtime_ns) if dense_cache else None
    )

    combined, local_index = load_local_rows(source_dir)
    split_ids: dict[str, list[str]] = {}
    split_positions: dict[str, list[int]] = {}
    for split in ("train", "test"):
        split_ids[split], split_positions[split] = match_author_rows(
            author_rows[split], local_index, split
        )

    train_set = set(split_ids["train"])
    test_set = set(split_ids["test"])
    if train_set & test_set:
        raise ValueError("Author train/test query_id sets overlap")
    if train_set | test_set != set(local_index):
        missing = set(local_index).difference(train_set | test_set)
        extra = (train_set | test_set).difference(local_index)
        raise ValueError(f"Author/local union differs: missing={missing}, extra={extra}")

    difficulty_by_id = {
        row["extra_info"]["query_id"]: normalize_difficulty(row["data_source"])
        for row in author_rows["test"]
    }
    difficulty_ids = {
        difficulty: [
            query_id
            for query_id in split_ids["test"]
            if difficulty_by_id[query_id] == difficulty
        ]
        for difficulty in ("easy", "medium", "hard")
    }
    difficulty_counts = {key: len(value) for key, value in difficulty_ids.items()}
    if difficulty_counts != EXPECTED_TEST_DIFFICULTY:
        raise ValueError(
            f"Wrong test difficulty distribution: {difficulty_counts} != {EXPECTED_TEST_DIFFICULTY}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        train_table = combined.take(pa.array(split_positions["train"], type=pa.int64()))
        test_table = combined.take(pa.array(split_positions["test"], type=pa.int64()))
        outputs: dict[str, tuple[pa.Table, list[str]]] = {
            "train.paper.parquet": (train_table, split_ids["train"]),
            "test.paper.parquet": (test_table, split_ids["test"]),
        }
        test_position_by_id = {
            query_id: position for position, query_id in enumerate(split_ids["test"])
        }
        for difficulty, query_ids in difficulty_ids.items():
            positions = [test_position_by_id[query_id] for query_id in query_ids]
            outputs[f"test.{difficulty}.paper.parquet"] = (
                test_table.take(pa.array(positions, type=pa.int64())),
                query_ids,
            )

        for name, (table, _) in outputs.items():
            write_parquet(table, temp_dir / name)
        for name, (_, expected_ids) in outputs.items():
            verify_output(temp_dir / name, expected_ids, combined.schema)

        source_records = {path.name: file_record(path) for path in source_paths}
        source_stats_after = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths
        }
        if source_stats_after != source_stats_before:
            raise RuntimeError("A source parquet changed while the split was being generated")

        cache_record = None
        if dense_cache is not None:
            cache_record = file_record(dense_cache)
            cache_stats_after = (dense_cache.stat().st_size, dense_cache.stat().st_mtime_ns)
            if cache_stats_after != cache_stats_before:
                raise RuntimeError("The dense cache changed while the split was being generated")
            cache_record["split_required"] = False
            cache_record["reason"] = (
                "The cache indexes the shared document corpus; the 680/150 split applies to queries."
            )

        output_records = {
            name: {
                **file_record(temp_dir / name),
                "path": name,
                "rows": len(expected_ids),
            }
            for name, (_, expected_ids) in outputs.items()
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Context-Folding author-released BrowseComp-Plus split",
            "matching_key": "query_id",
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "commit": UPSTREAM_COMMIT,
                "files": upstream_records,
            },
            "source_files": source_records,
            "dense_cache": cache_record,
            "counts": {
                "source_total": len(local_index),
                "train": len(split_ids["train"]),
                "test": len(split_ids["test"]),
                "test_by_difficulty": difficulty_counts,
            },
            "checks": {
                "query_id_union_equals_source": True,
                "train_test_disjoint": True,
                "query_text_exact_match": True,
                "answer_exact_match": True,
                "output_schema_equals_source": True,
                "source_files_unchanged": True,
                "dense_cache_unchanged": dense_cache is not None,
            },
            "split_query_ids": {
                "train": split_ids["train"],
                "test": split_ids["test"],
                "test_by_difficulty": difficulty_ids,
            },
            "test_difficulty_by_query_id": difficulty_by_id,
            "outputs": output_records,
        }
        with (temp_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        temp_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"Wrote Context-Folding split to {output_dir}")
    print("train=680 test=150 test_difficulty=easy:50,medium:50,hard:50")
    if dense_cache is not None:
        print(f"Dense cache left unchanged: {dense_cache}")


if __name__ == "__main__":
    main()
