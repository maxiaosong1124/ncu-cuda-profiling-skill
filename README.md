# NCU CUDA Optimizer v2 (Enhanced)

基于完整 NCU 指标的智能 CUDA 性能分析和自动优化工具。

## 功能特性

- **完整指标解析**: 支持 Speed Of Light、Compute/Memory Workload、Occupancy、Scheduler Stats、Warp Stall Reasons 等全部 NCU 指标
- **智能瓶颈诊断**: 基于决策树自动识别 10+ 种瓶颈类型
- **针对性优化策略**: 10 种内置优化策略，根据瓶颈类型自动匹配
- **双模式支持**: 交互式模式和全自动模式
- **自动回滚**: 性能下降自动回滚到上一版本
- **收敛检测**: 提升 < 3% 自动停止

## 安装

```bash
# 克隆仓库
git clone https://github.com/maxiaosong/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill

# 使用 v2 分支
git checkout v2
```

## 快速开始

### 1. 分析已有 NCU 报告

```bash
python optimizer.py --import-report <report.ncu-rep>
```

输出示例：
```
============================================================
NCU Metrics Summary
============================================================

📊 Speed Of Light Throughput:
  dram_throughput: 8.83%
  memory_throughput: 45.20%
  sm_throughput: 78.50%

🔢 Compute Workload:
  sm_busy: 82.30%
  issue_slots_busy: 81.48%
  executed_ipc_active: 1.25

💾 Memory Workload:
  l1_hit_rate: 65.40%
  l2_hit_rate: 95.41%

📈 Occupancy:
  occupancy: 75.00%
  theoretical_occupancy: 87.50%

⚡ Scheduler Stats:
  active_warps: 12.00
  eligible_warps: 1.49
  no_eligible: 15.30%

⏸️  Warp Stall Reasons:
  stall_barrier: 5.20%
  stall_memory_dependency: 35.40%
  stall_execution_dependency: 12.10%

============================================================
NCU Analysis Report (Imported)
============================================================

📊 Bottleneck: dram_memory_bound
⏱️  GPU Time: 0.633888 μs

💡 Recommendations:
  1. Block Tiling (3.0x)
  2. Vectorized Load (1.3x)
```

### 2. 分析 CUDA 源码

```bash
# 仅分析不优化
python optimizer.py matmul.cu --mode=analyze --build "nvcc -O3 -arch=sm_89 {source} -o {output}"
```

### 3. 全自动优化

```bash
# 全自动优化模式
python optimizer.py matmul.cu --mode=auto --build "nvcc -O3 {source} -o {output}"

# 交互式优化模式
python optimizer.py matmul.cu --mode=interactive
```

## 支持的 NCU 指标 (v2 新增)

### Speed Of Light Throughput
- `memory_throughput` - 整体内存吞吐量
- `dram_throughput` - 显存吞吐量
- `sm_throughput` - SM 计算吞吐量
- `l1tex_throughput` - L1/TEX 缓存吞吐量
- `l2_throughput` - L2 缓存吞吐量

### Compute Workload Analysis
- `sm_busy` - SM 忙碌程度
- `issue_slots_busy` - 发射槽忙碌率
- `executed_ipc_active` - 每周期执行指令数

### Memory Workload Analysis
- `l1_hit_rate` - L1 命中率
- `l2_hit_rate` - L2 命中率
- `mem_busy` - 内存单元忙碌度

### Occupancy
- `occupancy` - 实际占用率
- `theoretical_occupancy` - 理论占用率

### Scheduler Statistics
- `active_warps` - 活跃 warp 数
- `eligible_warps` - 准备好发射的 warp
- `no_eligible` - 无 eligible warp 的周期比例

### Warp State Statistics (Stall Reasons)
- `stall_wait` - 等待指令获取
- `stall_barrier` - 等待同步屏障
- `stall_memory_dependency` - 等待内存操作
- `stall_execution_dependency` - 等待指令结果
- `stall_memory_throttle` - 内存压力过大
- `stall_instruction_fetch` - 指令获取延迟
- `stall_texture` - 等待纹理操作
- `stall_constant` - 等待常量缓存
- `stall_not_selected` - 未被选中 (正常)

## 瓶颈类型与优化策略

| 瓶颈类型 | 判断依据 | 推荐策略 | 预期收益 |
|---------|---------|---------|---------|
| **DRAM_MEMORY_BOUND** | DRAM > 80%, SM < 50% | Block Tiling, Vectorized Load | 3-5x |
| **L1_PRESSURE_BOUND** | L1/TEX > 80%, DRAM < 30% | Shared Memory Padding | 1.2-2x |
| **L2_PRESSURE_BOUND** | L2 > 80%, L2 Hit < 50% | 调整 Tile 大小 | 1.2-1.5x |
| **COMPUTE_BOUND** | SM Throughput > 80% | Tensor Core, FMA | 1.1-3x |
| **LATENCY_BOUND** | SM Busy < 50%, Occupancy > 50% | Double Buffering | 1.2-1.5x |
| **OCCUPANCY_BOUND** | Occupancy < 30% | Launch Bounds | 1.2-2x |
| **BARRIER_STALL_BOUND** | Stall Barrier > 20% | Reduce Barrier, Warp Primitives | 1.3-1.5x |
| **MEMORY_DEPENDENCY_STALL** | Stall Memory Dep > 30% | Double Buffering, ILP | 1.2-1.5x |
| **EXECUTION_DEPENDENCY_STALL** | Stall Exec Dep > 30% | Loop Unrolling, ILP | 1.1-1.3x |
| **MIXED_BOUND** | Memory & Compute 都高 | 分阶段优化 | - |

## 优化策略列表

1. **Block Tiling** - 共享内存块级缓存
2. **Shared Memory Padding** - 避免 bank conflict
3. **Vectorized Load** - float4 向量化加载
4. **Double Buffering** - 双缓冲隐藏延迟
5. **Loop Unrolling** - 循环展开
6. **Register Optimization** - 寄存器优化提升 occupancy
7. **Warp-level Primitives** - warp shuffle 优化规约
8. **Grid-Stride Loops** - grid-stride 处理大数据
9. **Reduce Barrier** - 减少同步屏障
10. **Increase ILP** - 增加指令级并行

## 诊断决策树

```
开始分析
    │
    ▼
┌─────────────────────────────────┐
│ Speed Of Light Throughput       │
│ - DRAM Throughput > 80% ?       │
│ - SM Throughput > 80% ?         │
└─────────────────────────────────┘
    │
    ├─► DRAM > 80%, SM < 50% ───► DRAM_MEMORY_BOUND
    │   ├─ L1 Hit < 30% ───────► L1_PRESSURE_BOUND
    │   └─ L2 Hit < 50% ───────► L2_PRESSURE_BOUND
    │
    ├─► SM > 80%, DRAM < 50% ───► COMPUTE_BOUND
    │
    └─► 两者都低 ───► 检查 Warp Stall / Occupancy
            │
            ├─► Stall Barrier > 20% ───► BARRIER_STALL_BOUND
            ├─► Stall Memory Dep > 30% ──► MEMORY_DEPENDENCY_STALL
            ├─► Stall Exec Dep > 30% ───► EXECUTION_DEPENDENCY_STALL
            ├─► Occupancy < 30% ───────► OCCUPANCY_BOUND
            └─► SM Busy < 50% ─────────► LATENCY_BOUND
```

## 输出报告

分析完成后生成 Markdown 格式报告：

```markdown
# NCU 性能分析报告 (v2)

## Speed Of Light Throughput
| 指标 | 数值 | 状态 |
|------|------|------|
| dram_throughput | 85.30% | ⚠️ |
| sm_throughput | 35.20% | ✅ |

## Warp Stall Reasons
| 指标 | 数值 |
|------|------|
| stall_memory_dependency | 35.40% |
| stall_execution_dependency | 12.10% |

## 诊断详情
**瓶颈类型**: dram_memory_bound

**优化建议**:
1. **Block Tiling** (预期 3.0x 提升)
2. **Vectorized Load** (预期 1.3x 提升)
```

## 与 v1 版本对比

| 特性 | v1 | v2 |
|------|----|----|
| 指标数量 | 6 个基础指标 | 30+ 完整指标 |
| 瓶颈类型 | 6 种 | 10+ 种 |
| Stall Reasons | ❌ | ✅ |
| 优化策略 | 6 种 | 10 种 |
| 诊断精度 | 基础 | 精细 |

## 完整命令参考

```bash
# 导入已有报告分析
python optimizer.py --import-report report.ncu-rep

# 分析 CUDA 源码
python optimizer.py kernel.cu --mode=analyze --build "nvcc -O3 {source} -o {output}"

# 全自动优化
python optimizer.py kernel.cu --mode=auto --build "nvcc -O3 {source} -o {output}"

# 交互式优化
python optimizer.py kernel.cu --mode=interactive

# 指定 NCU 路径
python optimizer.py kernel.cu --ncu-path /usr/local/cuda/bin/ncu

# 调整收敛阈值
python optimizer.py kernel.cu --threshold 0.05 --max-iter 10
```

## 依赖

- Python 3.7+
- NVIDIA Nsight Compute (ncu)
- CUDA Toolkit (nvcc)

## 注意事项

1. **NCU 采集时间 ≠ 真实时间**
   - ncu 会多次重放 kernel 来收集指标
   - 采集时间会大幅膨胀，这是正常现象

2. **Stall Reasons 解读**
   - 只有当调度器无法每周期发射时才需要关注
   - Issue Slot 利用率已高时，stall 可能是正常调度

3. **优化验证**
   - 自动优化后务必验证结果正确性
   - 使用 `--mode=analyze` 对比优化前后指标

## 相关资源

- [NCU Profiling Template](../CUDA_PROFILING_TEMPLATE.md) - 完整性能分析模板
- [NVIDIA Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/)

---

*NCU CUDA Optimizer v2 - Enhanced with comprehensive metrics and intelligent diagnosis*
