# Infini - 算子融合开发仓库

> **⚠️ 内部开发仓库** - 功能正在迭代中，API 可能变更

## 📍 项目状态总览

| 模块 | 状态 | 说明 |
|------|------|------|
| `FusionScheduler` | ✅ 核心完成 | 调度、缓存、回退逻辑可用 |
| `SubGraph`/`OpNode` | ✅ 完成 | 数据结构可工作 |
| `FusionHeuristics` | ✅ 完成 | 静态启发式规则 |
| `KernelCompiler` | ⚠️ 部分完成 | 编译链路存在但端到端融合未验证 |
| `ninetoothed` 交互 | ⚠️ 需要验证 | Node 构建逻辑可能有问题 |
| InfiniLM 集成 | ❌ 未开始 | 推理引擎尚未接入融合调度器 |
| 性能基准 | ❌ 未验证 | README 中的性能数据是预估值 |

---

## 🚧 已知问题 & TODO

### 高优先级

1. **`KernelCompiler._build_fusion_nodes` 可能有问题**
   - 当前传入空 `args=()` 给 `Node`，不确定是否能正确建立数据依赖
   - 文件: `InfiniCore/python/infinicore/fusion/kernel_compiler.py:297-298`
   - 需有 GPU 环境实际测试

2. **端到端融合路径未实测**
   - `test_fusion_ntops.py` 和 `test_fusion_integration.py` 需要 CUDA + ntops + ninetoothed 环境
   - 回退路径 (`enable_fusion=False`) 已验证可用

3. **`rms_norm` 算子签名**  
   - 回退注册表只有 `rms_norm` 不确定 attrs 格式是否对
   - 文件: `fusion_scheduler.py:64-66`

### 中优先级

1. **缺少更多 LLM 融合模式**
   - 目前只有 `SwiGLU` 和 `Add+RMSNorm`
   - 可扩展: `GEGLU`, `LayerNorm+FFN`, `Attention` 内融合

2. **InfiniLM 接入点尚未确定**
   - 需要决定在 model forward 的哪个层级插入调度器

3. **算子注册表同步**
   - `heuristics.py` 和 `kernel_compiler.py` 各维护一份白名单，容易不同步

---

## 🏗️ 代码结构

```
InfiniCore/python/infinicore/fusion/
├── __init__.py              # 导出: FusionScheduler, FusionConfig, SubGraph, OpNode
├── fusion_scheduler.py      # ⭐ 核心调度器 (225 行)
├── fusion_config.py         # 配置 dataclass
├── heuristics.py            # 静态启发式规则
├── subgraph.py              # OpNode, SubGraph 数据结构
├── kernel_compiler.py       # ninetoothed 编译封装 (有风险)
└── patterns/
    ├── __init__.py
    └── llm_patterns.py      # SwiGLU, Add+RMSNorm 模式定义

InfiniCore/test/infinicore/
├── test_fusion_scheduler.py    # ✅ 18 个单元测试
├── test_fusion_integration.py  # ⚠️ 需 CUDA
├── test_fusion_ntops.py        # ⚠️ 需 CUDA + ntops + ninetoothed
└── bench_fusion.py             # ⚠️ 需 CUDA
```

---

## 🚀 快速开始

### 环境准备

```bash
cd /path/to/Infini/InfiniCore

# 基础安装
pip install -e .

# GPU 融合支持 (可选)
pip install ninetoothed ntops torch triton
```

### 运行单元测试 (无 GPU)

```bash
cd InfiniCore
python -m pytest test/infinicore/test_fusion_scheduler.py -v
# 预期: 18 passed
```

### GPU 测试 (需要 CUDA)

```bash
source ../activate_infini_env.sh  # 如有环境脚本

# 集成测试
python -m pytest test/infinicore/test_fusion_integration.py -v

# ntops 对接测试
python -m pytest test/infinicore/test_fusion_ntops.py -v

# 性能基准
python test/infinicore/bench_fusion.py --batch_size 32 --hidden_dim 4096
```

---

## 💻 基本用法

### 回退模式 (fusion 关闭，稳定可用)

```python
from infinicore.fusion import FusionScheduler, FusionConfig, SubGraph, OpNode

config = FusionConfig(enable_fusion=False)  # 禁用融合
scheduler = FusionScheduler(config)

graph = SubGraph(
    nodes=(
        OpNode("silu", ("x",), ("y1",)),
        OpNode("mul", ("y1", "x"), ("y2",)),
    ),
    input_names=("x",),
    output_names=("y2",),
)

# 这会走 infinicore.nn.functional 的标准算子
outputs = scheduler.dispatch(graph, {"x": tensor_x})
```

### 融合模式 (实验性)

```python
from infinicore.fusion import FusionScheduler, FusionConfig
from infinicore.fusion.patterns.llm_patterns import create_swiglu_pattern

config = FusionConfig(
    enable_fusion=True,
    enable_cache=True,
    debug_mode=True,       # 打印调试信息
    fallback_on_error=True # 编译失败自动回退
)
scheduler = FusionScheduler(config)

graph = create_swiglu_pattern()
outputs = scheduler.dispatch(graph, {"gate": gate_tensor, "up": up_tensor})
```

---

## 🔧 配置选项

```python
@dataclass
class FusionConfig:
    enable_fusion: bool = True        # 总开关
    enable_cache: bool = True         # 缓存编译后的内核
    min_tensor_elements: int = 1024   # 最小张量大小才融合
    min_nodes_for_fusion: int = 2     # 最少节点数
    fallback_on_error: bool = True    # 编译失败自动回退
    debug_mode: bool = False          # 详细日志
```

---

## 📦 依赖项目

| 项目 | 路径 | 说明 |
|------|------|------|
| ninetoothed | `../ninetoothed` | 符号化内核编译器，生成 Triton |
| ntops | `../ntops` | 算子库，提供 `premake` 函数 |
| InfiniLM | `../InfiniLM` | 推理引擎 (待接入) |
| InfiniTrain | `../InfiniTrain` | 训练框架 |

---

## 🧪 开发任务

### 接下来要做

- [ ] 在 GPU 环境验证 `KernelCompiler.compile` 端到端
- [ ] 修复 `_build_fusion_nodes` 的 args 传递问题
- [ ] 在 InfiniLM 中选择接入点
- [ ] 添加更多融合模式 (GEGLU 等)

### 如何添加新融合模式

1. 在 `patterns/llm_patterns.py` 添加函数:
```python
def create_my_pattern() -> SubGraph:
    return SubGraph(nodes=(...), ...)
```

2. 确保算子在白名单中:
   - `heuristics.py`: `_DEFAULT_OP_WHITELIST`
   - `kernel_compiler.py`: `_OP_REGISTRY`

3. 添加测试到 `test_fusion_scheduler.py`

### 如何调试

```python
config = FusionConfig(debug_mode=True, enable_fusion=True)
scheduler = FusionScheduler(config)

# 会打印:
# [FusionScheduler] Cache hit: xxx 或 Cache miss
# [KernelCompiler] Compiling graph: ...
# [FusionScheduler] Fallback execution for graph with N nodes
```

---

## 📝 相关文档

- `InfiniCore/test/infinicore/FusionScheduler 单元测试操作说明.md`
- `InfiniCore/test/infinicore/FusionScheduler_测试报告.md`
- `CLAUDE.md` - AI 助手指引

---

## ⚡ 性能预期 (待验证)

以下数据为设计目标，**尚未实测验证**:

| 操作 | 标准执行 | 融合执行 | 预期加速 |
|------|---------|---------|---------|
| SwiGLU (4096×32) | ~0.45 ms | ~0.18 ms | ~2.5x |
| Add+RMSNorm (4096×32) | ~0.52 ms | ~0.22 ms | ~2.4x |

---

*最后更新: 2026-01-21*
