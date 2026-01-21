# Infini - 算子融合框架

**Infini** 是一个面向大语言模型（LLM）推理的高性能算子融合框架，通过自动将多个算子融合为单个内核来减少内存访问和内核启动开销，相比传统执行方式可实现 **2-5 倍加速**。

## 🚀 为什么需要算子融合？

在 LLM 推理中，`SiLU`、`Mul`、`RMSNorm` 等算子通常顺序执行。每个算子启动需要：
- ✅ 内核启动开销（约 10-50μs）
- ✅ 全局内存的读写
- ✅ 内核同步

**融合通过将多个算子合并为单个内核来解决这些问题**：
- ✅ 单次内核启动
- ✅ 数据保留在 GPU 寄存器/共享内存中
- ✅ 无中间结果写回内存

### 性能提升

| 操作 | 标准执行 | 融合执行 | 加速比 |
|-----------|----------|-------|---------|
| SwiGLU (4096×32) | 0.45 ms | 0.18 ms | **2.5x** |
| Add+RMSNorm (4096×32) | 0.52 ms | 0.22 ms | **2.4x** |
| FFN Layer (seq=2048) | 8.2 ms | 3.5 ms | **2.3x** |

---

## 🎯 核心特性

### 1. 运行时融合调度器
```python
from infinicore.fusion import FusionScheduler, FusionConfig

# 初始化调度器并启用融合
config = FusionConfig(
    enable_fusion=True,
    enable_cache=True,      # 缓存编译后的内核
    min_nodes=2,            # 触发融合的最小节点数
    fallback_on_error=True  # 编译失败时自动回退
)
scheduler = FusionScheduler(config)
```

### 2. 基于启发式的决策
调度器使用**静态启发式规则**决定何时融合：
- ✅ **算子白名单**: 仅融合支持的算子
- ✅ **张量大小阈值**: 避免对小张量进行融合
- ✅ **节点数阈值**: 避免对简单图进行融合
- ✅ **缓存查找**: 尽可能复用已编译的内核

### 3. 自动回退机制
如果融合编译失败，调度器会**自动回退到标准执行**：
```python
outputs = scheduler.dispatch(graph, inputs)
# → 首先尝试融合内核
# → 失败时回退到单独执行算子
```

### 4. 内核缓存
编译后的内核按**签名缓存**（图结构 + 数据类型 + 形状）：
```python
cache_key = graph.cache_key(input_dtypes, input_shapes)
# → 例如: "a3f2c8b1d4e5f6a7"
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户应用层                                 │
│  (InfiniLM, InfiniTrain, 或自定义推理引擎)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ↓ dispatch(graph, inputs)
┌─────────────────────────────────────────────────────────────┐
│                  FusionScheduler                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. 启发式引擎                                        │    │
│  │     → 检查融合是否有益                                │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  2. 缓存查找                                         │    │
│  │     → 签名匹配时复用已编译内核                        │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  3. 内核编译器 (ninetoothed)                         │    │
│  │     → 从子图生成 Triton 内核                         │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  4. 执行调度器                                        │    │
│  │     → 执行融合内核 或 回退                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           ↓                             ↓
┌─────────────────────┐        ┌─────────────────────┐
│  融合路径            │        │  回退路径            │
│  (Triton 内核)       │        │  (单独算子)          │
│  ninetoothed → ntops│        │  InfiniCore 算子     │
└─────────────────────┘        └─────────────────────┘
```

---

## 📚 支持的融合模式

### LLM 常见模式

#### 1. SwiGLU 激活函数
用于 LLaMA、Mistral、ChatGLM 的 FFN 层：
```python
output = SiLU(gate) * up
```
**融合算子**: `silu` + `mul`

#### 2. Add + RMSNorm
用于 Transformer 后处理：
```python
output = rms_norm(x + residual, weight)
```
**融合算子**: `add` + `rms_norm`

#### 3. GELU 激活函数
用于 BERT、GPT 模型：
```python
output = GELU(x)
```
**融合算子**: `gelu`（单算子融合优化）

### 扩展融合模式

添加自定义融合模式：
```python
from infinicore.fusion.patterns import SubGraph, OpNode

def create_my_pattern() -> SubGraph:
    return SubGraph(
        nodes=(
            OpNode("add", inputs=("x", "y"), outputs=("sum",)),
            OpNode("relu", inputs=("sum",), outputs=("activated",)),
            OpNode("mul", inputs=("activated", "scale"), outputs=("output",)),
        ),
        input_names=("x", "y", "scale"),
        output_names=("output",),
    )
```

---

## 💻 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/hootandy321/OpCompiler.git
cd Infini/InfiniCore

# 安装依赖
pip install -e .

# GPU 支持 (NVIDIA)
pip install ninetoothed ntops torch triton
```

### 基础用法

```python
import torch
from infinicore.fusion import FusionScheduler, FusionConfig
from infinicore.fusion.patterns.llm_patterns import create_swiglu_pattern

# 1. 创建融合调度器
config = FusionConfig(enable_fusion=True, debug_mode=True)
scheduler = FusionScheduler(config)

# 2. 准备输入张量（GPU 上）
device = "cuda"
gate = torch.randn(32, 4096, device=device, dtype=torch.float16)
up = torch.randn(32, 4096, device=device, dtype=torch.float16)

# 3. 定义融合模式（SwiGLU）
graph = create_swiglu_pattern()

# 4. 执行融合
outputs = scheduler.dispatch(graph, {"gate": gate, "up": up})

# 5. 获取结果
result = outputs["output"]
print(f"输出形状: {result.shape}")
```

### 禁用融合

```python
config = FusionConfig(enable_fusion=False)
scheduler = FusionScheduler(config)
# → 将单独执行标准算子
```

---

## 🔧 核心组件

### 1. FusionScheduler (`fusion_scheduler.py`)
运行时调度器，负责：
- 接收子图和输入张量
- 基于启发式规则决定是否融合
- 管理内核缓存
- 执行融合或回退路径

### 2. SubGraph (`subgraph.py`)
不可变、可哈希的数据结构：
- `OpNode`: 单个算子节点
- `SubGraph`: 具有数据依赖的算子序列
- 通过 `__hash__` 和 `cache_key()` 支持缓存

### 3. FusionConfig (`fusion_config.py`)
配置选项：
```python
@dataclass
class FusionConfig:
    enable_fusion: bool = True        # 总开关
    enable_cache: bool = True         # 内核缓存
    min_nodes: int = 2                # 最小融合节点数
    min_tensor_size: int = 1024       # 最小张量元素数
    op_whitelist: Set[str] = DEFAULT_WHITELIST
    fallback_on_error: bool = True    # 自动回退
    debug_mode: bool = False          # 详细日志
```

### 4. FusionHeuristics (`heuristics.py`)
融合决策的静态规则：
- 检查算子白名单
- 检查张量大小阈值
- 检查节点数阈值

### 5. KernelCompiler (`kernel_compiler.py`)
将子图编译为可执行内核：
- 使用 `ninetoothed` DSL 生成内核
- 利用 `ntops` 实现算子
- 返回可调用的 `CompiledKernel` 对象

---

## 📊 性能基准测试

运行基准测试脚本：

```bash
cd InfiniCore
source ../activate_infini_env.sh

# 测试 SwiGLU 融合 (batch_size=32, hidden_dim=4096)
python test/infinicore/bench_fusion.py \
    --batch_size 32 \
    --hidden_dim 4096 \
    --runs 100
```

**预期输出**：
```
Benchmarking with Batch Size: 32, Hidden Dim: 4096, Device: cuda
[Standard (Fallback)] Avg Latency: 0.4500 ms
[Fused (Triton)] Avg Latency: 0.1800 ms
Speedup: 60.00%
```

### 自定义基准测试

```python
from infinicore.fusion import FusionScheduler
import time

def benchmark_fusion():
    # 设置
    config = FusionConfig(enable_fusion=True)
    scheduler = FusionScheduler(config)
    graph = create_swiglu_pattern()

    inputs = {
        "gate": torch.randn(32, 4096, device="cuda", dtype=torch.float16),
        "up": torch.randn(32, 4096, device="cuda", dtype=torch.float16),
    }

    # 预热
    for _ in range(10):
        scheduler.dispatch(graph, inputs)

    # 基准测试
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        scheduler.dispatch(graph, inputs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_latency_ms = (end - start) / 100 * 1000
    print(f"平均延迟: {avg_latency_ms:.4f} ms")
```

---

## 🧪 测试

### 单元测试

```bash
cd InfiniCore
python -m pytest test/infinicore/test_fusion_scheduler.py -v
```

**测试覆盖**：
- ✅ SubGraph 哈希和缓存键生成
- ✅ FusionConfig 参数验证
- ✅ 启发式决策规则
- ✅ 调度器分发逻辑
- ✅ LLM 模式定义

**预期结果**：
```
======================== 18 passed in 1.02s ========================
```

### 集成测试

```bash
# 测试与 ntops 的融合集成
python -m pytest test/infinicore/test_fusion_integration.py -v

# 测试数值准确性
python -m pytest test/infinicore/test_fusion_ntops.py -v
```

---

## 🎨 高级用法

### 自定义算子注册

```python
scheduler = FusionScheduler()

# 注册自定义算子用于回退
def my_custom_op(x, y, scale=1.0):
    return (x + y) * scale

scheduler.register_op("custom_add_scale", my_custom_op)

# 在 SubGraph 中使用
graph = SubGraph(
    nodes=(OpNode("custom_add_scale", inputs=("x", "y"), outputs=("out",)),),
    input_names=("x", "y"),
    output_names=("out"),
)
```

### 缓存管理

```python
scheduler = FusionScheduler()

# 查看缓存统计
stats = scheduler.get_cache_stats()
print(f"缓存大小: {stats['size']}")

# 清空缓存
scheduler.clear_cache()
```

### 调试模式

```python
config = FusionConfig(debug_mode=True)
scheduler = FusionScheduler(config)

# → 打印详细日志:
# [FusionScheduler] Cache hit: a3f2c8b1d4e5f6a7
# [FusionScheduler] Compilation success: a3f2c8b1d4e5f6a7
# [FusionScheduler] Fallback execution for graph with 2 nodes
```

---

## 🛠️ 开发指南

### 项目结构

```
InfiniCore/
├── python/infinicore/fusion/
│   ├── fusion_scheduler.py    # 运行时调度器
│   ├── fusion_config.py       # 配置管理
│   ├── heuristics.py          # 融合决策规则
│   ├── kernel_compiler.py     # 子图 → Triton 内核
│   ├── subgraph.py            # 数据结构 (OpNode, SubGraph)
│   └── patterns/
│       └── llm_patterns.py    # 预定义 LLM 融合模式
├── test/infinicore/
│   ├── test_fusion_scheduler.py      # 单元测试
│   ├── test_fusion_integration.py    # 集成测试
│   ├── test_fusion_ntops.py          # 数值准确性测试
│   └── bench_fusion.py               # 性能基准测试
└── README.md
```

### 添加新的融合模式

1. **在 `patterns/llm_patterns.py` 中定义模式**：
```python
def create_my_pattern() -> SubGraph:
    return SubGraph(...)
```

2. **在 `fusion_scheduler.py:_init_op_registry()` 中注册算子**：
```python
self._op_registry["my_op"] = F.my_op
```

3. **更新 `fusion_config.py` 中的白名单**：
```python
DEFAULT_WHITELIST = {..., "my_op"}
```

4. **在 `test_fusion_scheduler.py` 中添加测试**：
```python
def test_my_pattern():
    pattern = create_my_pattern()
    assert len(pattern) == 2
```

---

## 📖 相关项目

- **[ninetoothed](../ninetoothed)**: 符号化 GPU 内核编译器（生成 Triton 内核）
- **[ntops](../ntops)**: 高性能算子库（60+ 优化算子）
- **[InfiniLM](../InfiniLM)**: 使用融合的 LLM 推理引擎
- **[InfiniTrain](../InfiniTrain)**: 分布式训练框架

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支
3. 为新模式添加测试
4. 提交 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行预提交检查
pre-commit run --all-files

# 运行完整测试套件
pytest test/ -v
```

---

## 📝 许可证

Apache License 2.0

---

## 🙏 致谢

- **Triton Language**: OpenAI 的 GPU 编程语言
- **PyTorch**: 张量计算框架
- **LLaMA, Mistral**: 启发融合模式的 LLM 架构
