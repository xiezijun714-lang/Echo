# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Generator, Optional

import ray
import sglang.srt.entrypoints.engine
import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
try:
    from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
    _has_weight_sync = True
except ImportError:
    _has_weight_sync = False
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_server_runtime_option(config: RolloutConfig, attr: str, env_name: str, default):
    value = os.getenv(env_name)
    if value is None:
        value = default
    return type(default)(value)


async def _sgl_update_weights_no_flush(
    engine: AsyncHttpServerAdapter,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
):
    import torch.distributed as dist
    from sglang.srt.managers.tokenizer_manager import UpdateWeightsFromTensorReqInput
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
    from sglang.srt.patch_torch import monkey_patch_torch_reductions
    from sglang.srt.utils import MultiprocessingSerializer
    from torch.distributed.tensor import DTensor

    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
    monkey_patch_torch_reductions()

    named_tensors_batch = [
        (
            name,
            MultiprocessingSerializer.serialize(tensor.full_tensor() if isinstance(tensor, DTensor) else tensor),
        )
        for name, tensor in params_batch
    ]

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        logical_tensors = zip(*gathered_serialized_batches, strict=True)
        named_tensors = [
            (
                tensor_group[0][0],
                LocalSerializedTensor(values=[rank_part[1] for rank_part in tensor_group]),
            )
            for tensor_group in logical_tensors
        ]
        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors) for _ in range(infer_tp_size)
            ],
            load_format=load_format,
            flush_cache=False,
            abort_all_requests=True,
        )
        return await engine.update_weights_from_tensor(update_weights_request)


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    # Enable faulthandler in subprocesses
    os.environ["PYTHONFAULTHANDLER"] = "1"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        if config.get("quantization", None) == "fp8":
            import sglang
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        super().__init__(config, model_config, device_mesh)
        self._engine: AsyncHttpServerAdapter = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size

    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        # device_mesh is needed to gather cuda ipc handle to update weights
        if self.device_mesh is None:
            assert torch.distributed.is_initialized(), "torch distributed must be initialized"
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            infer_pp = self.config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = torch.distributed.get_world_size() // infer_world_size
            self.device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

        # Only init http server adapter in tp rank 0
        if self.device_mesh["infer_tp"].get_local_rank() != 0:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path,
            host=host,
            port=server_port,
            launch_server=False,
            timeout=_get_server_runtime_option(self.config, "timeout", "SGLANG_SERVER_TIMEOUT", 300.0),
            max_attempts=_get_server_runtime_option(self.config, "max_attempts", "SGLANG_SERVER_MAX_ATTEMPTS", 5),
            retry_delay=_get_server_runtime_option(self.config, "retry_delay", "SGLANG_SERVER_RETRY_DELAY", 5.0),
            trust_remote_code=self.model_config.trust_remote_code,
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])

    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        await self._init_server_adapter()

        update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
        if self.config.get("quantization", None) == "fp8":
            from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper

            logger.info("Convert bf16 weights to fp8 format before loading")
            fp8_quantizer_helper = SGLangFP8QuantizerHelper(self.model_config.hf_config.quantization_config)
            weights = fp8_quantizer_helper.quant_weights_by_name(
                weights,
                dtype=self.model_config.hf_config.dtype,
            )
        else:
            weights = weights

        is_tp_rank_zero = self.device_mesh["infer_tp"].get_local_rank() == 0
        if is_tp_rank_zero:
            await self.server_actor.abort_all_requests.remote()

        try:
            if _has_weight_sync:
                async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
                    await _sgl_update_weights_no_flush(
                        engine=self._engine,
                        params_batch=params_batch,
                        device_mesh_key="infer_tp",
                        device_mesh=self.device_mesh,
                    )
            else:
                # Fallback for sglang < 0.5.0 (no weight_sync module)
                # Use Ray remote call to bypass HTTP overhead (base64 + HTTP POST)
                from sglang.srt.utils import MultiprocessingSerializer

                if is_tp_rank_zero:
                    async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
                        batch_list = list(params_batch)
                        serialized = [
                            MultiprocessingSerializer.serialize(batch_list)
                            for _ in range(self.device_mesh["infer_tp"].size())
                        ]
                        await self.server_actor.update_weights_direct.remote(serialized)

            if is_tp_rank_zero:
                await self.server_actor.clear_kv_cache.remote()
                if global_steps is not None:
                    await self.server_actor.set_global_steps.remote(global_steps)
        finally:
            if is_tp_rank_zero:
                await self.server_actor.resume_generation.remote()
