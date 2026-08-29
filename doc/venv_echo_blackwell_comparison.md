# `venv_echo_blackwell` 环境对比

## 对比对象

本文记录以下两个虚拟环境在 2026-08-29 的只读对比结果：

- `/root/paddlejob/gpfsspace/xiezijun/venv_echo_blackwell`
- `/root/paddlejob/gpfsspace/qiruyi/venv_echo_blackwell`

对比使用各自的 `bin/python`、`pip list --format=freeze`、文件清单、符号链接和导入冒烟测试；没有修改两个虚拟环境。

## 总结

两个环境的 Python 基础配置相同，但依赖集合和 CUDA 扩展并不相同。`xiezijun` 环境包含较新的 CUDA 13/Blackwell 相关组件；`qiruyi` 环境保留了 `flash-attn` 和 CUDA 12 NCCL 组件。使用当前文件直接运行时，`qiruyi` 的 PyTorch CUDA 动态库存在符号不匹配，不能正常导入 `torch`。

## 共同基础

两边的 `pyvenv.cfg` 完全一致：

```text
home = /bin
include-system-site-packages = false
version = 3.10.16
```

规范化包名后，结果如下：

| 项目 | xiezijun | qiruyi |
| --- | ---: | ---: |
| 已安装发行包 | 252 | 230 |
| 名称和版本都相同的包 | 228 | 228 |
| 总文件数 | 102,079 | 87,285 |
| 磁盘占用 | 约 13 GB | 约 14 GB |

两边的 `bin/python3` 都指向共享的外部解释器：

```text
/root/paddlejob/gpfsspace/xiezijun/.uv-python31016/cpython-3.10.16-linux-x86_64-gnu/bin/python3.10
```

## 包差异

### xiezijun 独有

通用/模型相关包：

```text
easydict==1.13
gguf==0.19.0
llvmlite==0.47.0
ml_dtypes==0.6.0
numba==0.65.1
onnx==1.22.0
onnx-ir==1.0.0
onnxscript==0.7.1
rank-bm25==0.2.2
sgl-deep-gemm==0.1.5.post3
timm==1.0.28
tomli==2.4.1
watchfiles==1.2.0
zstandard==0.25.0
```

CUDA/编译相关包：

```text
nvidia-cuda-nvdisasm==13.3.73
nvidia-cutlass-dsl==4.6.2
nvidia-cutlass-dsl-libs-base==4.6.2
nvidia-cutlass-dsl-libs-core==4.6.2
nvidia-cutlass-dsl-libs-cu12==4.6.2
nvidia-cutlass-dsl-libs-cu13==4.6.2
nvidia-mathdx==25.6.0
nvidia-nccl-cu13==2.29.7
transformer_engine_cu13==2.18.0
transformer_engine_torch==2.18.0
```

### qiruyi 独有

```text
flash_attn==2.7.4.post1
nvidia-nccl-cu12==2.27.5
```

### 同名包但版本不同

```text
transformer_engine
  xiezijun: 2.18.0
  qiruyi:   2.5.0+f05f12c
```

## 文件和实现差异

- `xiezijun` 有 `site-packages/sitecustomize.py`，包含 Qwen3/Transformers 兼容补丁，以及 CPU-only Ray/Triton 的处理；`qiruyi` 没有该文件。
- `xiezijun` 有 `nvidia_cutlass_dsl_packages.pth`，用于把 CUTLASS DSL 包加入导入路径；`qiruyi` 没有。
- `qiruyi` 有约 1.43 GB 的 `flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`，以及 `flash_attn/`、`hopper/` 目录；`xiezijun` 没有 `flash_attn`。
- 两边的 `deep_gemm` 实现不同：`xiezijun` 是 `sgl-deep-gemm` 新实现，包含 `deep_gemm/_C.so` 和新版 CUTLASS 代码；`qiruyi` 是旧的 JIT 实现，没有 `_C.so`。
- cuDNN 的实际加载方式不同：`xiezijun` 的 `nvidia/cudnn/lib` 中的库链接到系统 `/usr/lib/x86_64-linux-gnu/libcudnn_*.so.9.20.0`，并保留了 `lib-cu12-backup`；`qiruyi` 直接使用环境内的 cuDNN 库文件。
- `qiruyi/bin/flashinfer`、`hf`、`sglang`、`torchrun`、`transformers`、`tvm-ffi-*` 等多个入口脚本的 shebang 都写死为 `/root/paddlejob/gpfsspace/xiezijun/venv_echo_blackwell/bin/python`，调用这些命令时并不会使用 qiruyi 的 Python。

## 激活脚本问题

两个环境的激活脚本内容完全相同，但都写死指向旧的 `venv_echo_megatron`：

```bash
VIRTUAL_ENV="/root/paddlejob/gpfsspace/xiezijun/venv_echo_megatron"
```

因此执行：

```bash
source /root/paddlejob/gpfsspace/qiruyi/venv_echo_blackwell/bin/activate
```

得到的 `VIRTUAL_ENV` 和 `python` 实际都来自 `xiezijun/venv_echo_megatron`，而不是 qiruyi 的 Blackwell 环境。直接调用各自的 `bin/python` 可以避免这个问题。

## 运行时验证

在相同 shell 中直接使用各自的 `bin/python`：

| 导入项 | xiezijun | qiruyi |
| --- | --- | --- |
| `torch` | 正常，`2.13.0+cu130` | 失败：`undefined symbol: ncclCommResume` |
| `transformer_engine` | 正常，`2.18.0` | 失败：cuDNN 符号不匹配 |
| `deep_gemm` | 正常，`0.1.5.post3` | 因 PyTorch CUDA 加载失败 |
| `sglang` | 正常导入 | 因 PyTorch CUDA 加载失败 |
| `flash_attn` | 模块不存在 | 因 PyTorch CUDA 加载失败 |

符号检查也印证了 NCCL 差异：`xiezijun` 的 `libnccl.so.2` 导出 `ncclCommResume`，`qiruyi` 的对应库没有该符号。两边的 `pip check` 都不是完全干净，分别报告 29 条和 38 条依赖问题；其中包括 CUDA 组件、SGLang 可选依赖和部分版本约束不一致。

## 结论和建议

如果目标是 Blackwell/CUDA 13，当前 `xiezijun` 环境更接近目标配置，并且 PyTorch、Transformer Engine、SGLang 的基础导入冒烟测试可以通过。`qiruyi` 应重新对齐到同一套 CUDA 13 的 NCCL/cuDNN/Transformer Engine 依赖，或者重新创建虚拟环境；仅复制少量包或脚本不足以修复动态库 ABI 不匹配。

在修复激活脚本和入口脚本之前，建议使用完整路径直接调用：

```bash
/root/paddlejob/gpfsspace/xiezijun/venv_echo_blackwell/bin/python ...
/root/paddlejob/gpfsspace/qiruyi/venv_echo_blackwell/bin/python ...
```
