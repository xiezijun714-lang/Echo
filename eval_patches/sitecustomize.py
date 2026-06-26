"""Runtime patch for BrowseComp-Plus official vLLM evaluation.

This keeps the official evaluation script unchanged while allowing this shared
GPU machine to run the judge model with a smaller KV-cache reservation.
"""

import os

import vllm


_OriginalLLM = vllm.LLM


def LLM(*args, **kwargs):
    kwargs.setdefault(
        "gpu_memory_utilization",
        float(os.environ.get("BCP_EVAL_GPU_MEMORY_UTILIZATION", "0.45")),
    )
    kwargs.setdefault("max_model_len", int(os.environ.get("BCP_EVAL_MAX_MODEL_LEN", "8192")))
    return _OriginalLLM(*args, **kwargs)


vllm.LLM = LLM
