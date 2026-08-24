#!/usr/bin/env python3
"""Small NCCL all-reduce check for the packaged Echo environment."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    value = torch.tensor([dist.get_rank() + 1.0], device="cuda")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all-reduce mismatch: got {value.item()}, expected {expected}")

    if dist.get_rank() == 0:
        props = torch.cuda.get_device_properties(local_rank)
        print(
            f"NCCL OK: world_size={dist.get_world_size()}, result={value.item():.0f}, "
            f"gpu={props.name}, capability={props.major}.{props.minor}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
