# InfiniLM 融合优化使用教程

本教程介绍如何使用 `FusedInferEngine` 进行推理优化。

---

## 目录

1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [Python API](#python-api)
4. [性能监控](#性能监控)
5. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
# 安装 InfiniCore
cd /path/to/Infini/InfiniCore
pip install -e .

# 安装 InfiniLM
cd /path/to/Infini/InfiniLM
pip install -e .
```

### 2. 验证安装

```python
# 检查 FusedInferEngine 是否可用
from infinilm import FusedInferEngine
print("✅ FusedInferEngine 可用")

# 检查 FusionScheduler 是否可用
from infinilm.fused_infer_engine import FUSION_AVAILABLE
print(f"FusionScheduler: {'✅ 可用' if FUSION_AVAILABLE else '❌ 不可用'}")
```

---

## 快速开始

### 命令行方式

```bash
# 启用融合优化
python examples/jiuge.py --nvidia --model_path=~/models/TinyLlama --enable-fusion

# 禁用融合（对比测试）
python examples/jiuge.py --nvidia --model_path=~/models/TinyLlama
```

### 支持的硬件

| 参数 | 硬件 |
|------|------|
| `--nvidia` | NVIDIA GPU (CUDA) |
| `--metax` | MetaX GPU |
| `--moore` | 摩尔线程 GPU |
| `--iluvatar` | 天数智芯 GPU |
| `--cambricon` | 寒武纪 MLU |
| `--cpu` | CPU |

---

## Python API

### 基本用法

```python
import infinicore
from infinilm import FusedInferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.distributed import DistConfig

# 1. 创建引擎
engine = FusedInferEngine(
    model_path="~/models/TinyLlama",
    enable_fusion=True,       # 启用融合
    warmup_iterations=1,      # 预热次数
    device=infinicore.device("cuda", 0),
    distributed_config=DistConfig(1),
)

# 2. 加载权重
model_path = "~/models/TinyLlama"
load_model_state_dict_by_file(engine, model_path, dtype=engine.config.dtype)

# 3. 推理
import torch
input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")
pos = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device="cuda")

# 第一次调用（录制 + 优化）
output = engine.forward(input_ids=input_ids, pos=pos)

# 后续调用（使用缓存）
output = engine.forward(input_ids=input_ids, pos=pos)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_fusion` | bool | True | 是否启用融合优化 |
| `warmup_iterations` | int | 1 | 预热迭代次数 |
| `fusion_config` | FusionConfig | None | 融合配置（可选） |

### 运行时控制

```python
# 禁用融合
engine.set_fusion_enabled(False)

# 启用融合
engine.set_fusion_enabled(True)

# 清空缓存
engine.clear_cache()

# 检查状态
print(engine.fusion_enabled)            # True/False
print(engine.fusion_scheduler_available) # True/False
```

---

## 性能监控

### 查看统计

```python
stats = engine.get_stats()
print(stats)
```

输出示例：
```python
{
    "enabled": True,
    "fusion_scheduler_available": True,
    "cache_size": 2,
    "cache_hits": 98,
    "cache_misses": 2,
    "recordings": 2,
    "fusion_attempts": 2,
    "fusion_successes": 0,
    "fusion_fallbacks": 2,
    "cached_shapes": ["abc123", "def456"],
    "fusion_modes": {"abc123": "graph_replay", "def456": "graph_replay"}
}
```

### 指标解读

| 指标 | 含义 |
|------|------|
| `cache_hits` | 缓存命中次数（使用已优化路径） |
| `cache_misses` | 缓存未命中次数 |
| `recordings` | Graph 录制次数 |
| `fusion_attempts` | 尝试融合次数 |
| `fusion_successes` | 融合成功次数（SubGraph 转换成功） |
| `fusion_fallbacks` | 回退到 Graph 重放次数 |
| `fusion_modes` | 每个缓存项的执行模式 |

### 命中率计算

```python
stats = engine.get_stats()
total = stats["cache_hits"] + stats["cache_misses"]
hit_rate = stats["cache_hits"] / total if total > 0 else 0
print(f"缓存命中率: {hit_rate:.1%}")
```

---

## 常见问题

### Q1: fusion_fallbacks 很高，融合没有生效？

**原因**：当前 C++ Graph 没有暴露节点信息，SubGraph 转换会失败。

**解决**：这是预期行为。即使融合未生效，Graph 缓存重放仍能提供加速（跳过算子调度开销）。

### Q2: 每次推理都 cache_miss？

**原因**：输入 shape 每次都不同。Graph 缓存基于 shape 区分。

**解决**：对于动态 shape 场景，考虑禁用融合：
```python
engine.set_fusion_enabled(False)
```

### Q3: 内存占用增加？

**原因**：Graph 缓存会保持张量引用。

**解决**：定期清理缓存：
```python
engine.clear_cache()
```

### Q4: 如何对比融合前后性能？

```python
import time

# 不使用融合
engine.set_fusion_enabled(False)
engine.clear_cache()
start = time.time()
for _ in range(100):
    engine.forward(input_ids=input_ids, pos=pos)
print(f"无融合: {time.time() - start:.3f}s")

# 使用融合
engine.set_fusion_enabled(True)
engine.clear_cache()
start = time.time()
for _ in range(100):
    engine.forward(input_ids=input_ids, pos=pos)
print(f"有融合: {time.time() - start:.3f}s")
```

---

## 执行模式说明

| 模式 | 说明 | 性能 |
|------|------|------|
| `fusion` | FusionScheduler 执行融合内核 | 最优（需要完整支持） |
| `graph_replay` | Graph.run() 重放 | 良好（跳过调度开销） |
| 原生 | 直接 forward() | 基准 |

当前实现中，`graph_replay` 是主要执行模式，因为 C++ Graph 尚未暴露节点信息导致 SubGraph 转换失败。
