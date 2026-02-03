#!/usr/bin/env python3
"""
CUDA Optimization Strategy Library
内置常用 CUDA 优化策略，支持瓶颈类型匹配和代码生成
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import re


class BottleneckType(Enum):
    """瓶颈类型定义"""
    DRAM_MEMORY_BOUND = "dram_memory_bound"
    L1_PRESSURE_BOUND = "l1_pressure_bound"
    LATENCY_BOUND = "latency_bound"
    COMPUTE_BOUND = "compute_bound"
    OCCUPANCY_BOUND = "occupancy_bound"
    MIXED_BOUND = "mixed_bound"
    UNKNOWN = "unknown"


@dataclass
class OptimizationStrategy:
    """优化策略定义"""
    name: str
    description: str
    target_bottlenecks: List[BottleneckType]
    applicable_metrics: Dict[str, Callable[[float], bool]]  # 指标条件函数
    code_template: str  # 代码模板
    insertion_pattern: str  # 插入位置匹配模式
    expected_speedup: float  # 预期加速比
    complexity: str  # 复杂度: low/medium/high
    prerequisites: List[str]  # 前置条件


class CUDAStrategyLibrary:
    """CUDA 优化策略库"""

    def __init__(self):
        self.strategies: Dict[str, OptimizationStrategy] = {}
        self._init_strategies()

    def _init_strategies(self):
        """初始化所有优化策略"""

        # 1. Block Tiling - 解决 DRAM Memory Bound
        self.strategies["block_tiling"] = OptimizationStrategy(
            name="Block Tiling",
            description="使用共享内存进行块级数据缓存，减少全局内存访问",
            target_bottlenecks=[BottleneckType.DRAM_MEMORY_BOUND],
            applicable_metrics={
                "dram_throughput": lambda x: x > 70,
                "l1_hit_rate": lambda x: x < 20,
                "roofline_ratio": lambda x: x < 40
            },
            code_template="""
// Block Tiling 优化
// 配置参数
const int BM = {bm};  // 每个 block 处理的 M 维度
const int BN = {bn};  // 每个 block 处理的 N 维度
const int BK = {bk};  // 每个 block 累加的 K 维度

__shared__ float As[BM][BK];
__shared__ float Bs[BK][BN];

// 加载 A 到共享内存
for (int k = 0; k < K; k += BK) {{
    // 协作加载 A 块
    for (int i = threadIdx.y; i < BM; i += blockDim.y) {{
        for (int j = threadIdx.x; j < BK; j += blockDim.x) {{
            As[i][j] = A[(blockIdx.y * BM + i) * K + k + j];
        }}
    }}
    // 协作加载 B 块
    for (int i = threadIdx.y; i < BK; i += blockDim.y) {{
        for (int j = threadIdx.x; j < BN; j += blockDim.x) {{
            Bs[i][j] = B[(k + i) * N + blockIdx.x * BN + j];
        }}
    }}
    __syncthreads();

    // 计算
    for (int kk = 0; kk < BK; ++kk) {{
        // 使用共享内存数据计算
    }}
    __syncthreads();
}}
""",
            insertion_pattern=r"__global__\s+void\s+(\w+).*?\{",
            expected_speedup=3.0,
            complexity="medium",
            prerequisites=["可分解的循环结构", "数据复用机会"]
        )

        # 2. Shared Memory Padding - 解决 Bank Conflict
        self.strategies["smem_padding"] = OptimizationStrategy(
            name="Shared Memory Padding",
            description="通过填充避免共享内存 bank conflict",
            target_bottlenecks=[BottleneckType.L1_PRESSURE_BOUND, BottleneckType.DRAM_MEMORY_BOUND],
            applicable_metrics={
                "l1tex_throughput": lambda x: x > 80,
                "shared_memory_bank_conflict": lambda x: x > 0,
                "l1_hit_rate": lambda x: x < 50
            },
            code_template="""
// Shared Memory Padding 优化
// 修改前: __shared__ float shared[{dim1}][{dim2}];
// 修改后: 添加 padding 避免 bank conflict
__shared__ float shared[{dim1}][{dim2} + {pad}];  // +{pad} padding

// 访问方式保持不变，但性能提升
// shared[row][col] 不再产生 bank conflict
""",
            insertion_pattern=r"__shared__\s+(\w+)\s+(\w+)\[(\d+)\]\[(\d+)\]",
            expected_speedup=1.5,
            complexity="low",
            prerequisites=["使用共享内存", "存在 bank conflict"]
        )

        # 3. Vectorized Load (float4) - 提升内存带宽
        self.strategies["vectorized_load"] = OptimizationStrategy(
            name="Vectorized Load",
            description="使用 float4 向量化加载提升内存带宽利用率",
            target_bottlenecks=[BottleneckType.DRAM_MEMORY_BOUND],
            applicable_metrics={
                "dram_throughput": lambda x: x > 60,
                "memory_bandwidth_utilization": lambda x: x < 80
            },
            code_template="""
// Vectorized Load 优化 - 使用 float4
// 修改前:
// float a = A[idx];
// float b = A[idx + 1];
// float c = A[idx + 2];
// float d = A[idx + 3];

// 修改后:
float4 vec = reinterpret_cast<const float4*>(A)[idx / 4];
float a = vec.x;
float b = vec.y;
float c = vec.z;
float d = vec.w;

// 或者使用结构体
struct alignas(16) float4_array {{
    float4 data[{size} / 4];
}};
""",
            insertion_pattern=r"(\w+)\s*=\s*(\w+)\[(\w+)\]",
            expected_speedup=1.3,
            complexity="medium",
            prerequisites=["连续内存访问模式", "对齐的内存地址"]
        )

        # 4. Double Buffering - 隐藏延迟
        self.strategies["double_buffering"] = OptimizationStrategy(
            name="Double Buffering",
            description="使用双缓冲技术隐藏内存加载延迟",
            target_bottlenecks=[BottleneckType.LATENCY_BOUND],
            applicable_metrics={
                "sm_busy": lambda x: x < 50,
                "occupancy": lambda x: x > 60,
                "issue_slot_utilization": lambda x: x < 70
            },
            code_template="""
// Double Buffering 优化
__shared__ float As[2][BM][BK];
__shared__ float Bs[2][BK][BN];

int load_stage = 0;
int compute_stage = 0;

// 预加载第一块
// ... 加载代码到 As[0], Bs[0] ...
__syncthreads();

for (int k = BK; k < K; k += BK) {{
    load_stage = 1 - compute_stage;

    // 在计算当前块的同时，加载下一块
    if (threadIdx.x == 0 && threadIdx.y == 0) {{
        // 异步加载到 As[load_stage], Bs[load_stage]
    }}

    // 计算当前块 (从 compute_stage 读取)
    for (int kk = 0; kk < BK; ++kk) {{
        // 使用 As[compute_stage], Bs[compute_stage]
    }}

    __syncthreads();
    compute_stage = load_stage;
}}
""",
            insertion_pattern=r"for\s*\(\s*int\s+\w+\s*=\s*0",
            expected_speedup=1.3,
            complexity="high",
            prerequisites=["循环迭代间无数据依赖", "足够的共享内存"]
        )

        # 5. Loop Unrolling - 减少循环开销
        self.strategies["loop_unrolling"] = OptimizationStrategy(
            name="Loop Unrolling",
            description="手动或编译器指令展开循环减少开销",
            target_bottlenecks=[BottleneckType.LATENCY_BOUND, BottleneckType.COMPUTE_BOUND],
            applicable_metrics={
                "instruction_overhead": lambda x: x > 20,
                "loop_iterations": lambda x: x > 8
            },
            code_template="""
// Loop Unrolling 优化
// 方法1: 编译器指令
#pragma unroll {unroll_factor}
for (int i = 0; i < {loop_bound}; ++i) {{
    // 循环体
}}

// 方法2: 手动展开 (更精细控制)
// 原循环: for (int i = 0; i < 8; ++i) sum += data[i];
// 展开后:
sum += data[0];
sum += data[1];
sum += data[2];
sum += data[3];
sum += data[4];
sum += data[5];
sum += data[6];
sum += data[7];
""",
            insertion_pattern=r"for\s*\(\s*int\s+(\w+)\s*=\s*0",
            expected_speedup=1.2,
            complexity="low",
            prerequisites=["循环边界固定或已知", "循环体简单"]
        )

        # 6. Register Optimization - 提升 Occupancy
        self.strategies["register_opt"] = OptimizationStrategy(
            name="Register Optimization",
            description="减少寄存器使用以提升 occupancy",
            target_bottlenecks=[BottleneckType.OCCUPANCY_BOUND],
            applicable_metrics={
                "occupancy": lambda x: x < 40,
                "registers_per_thread": lambda x: x > 64,
                "theoretical_occupancy": lambda x: x > 70
            },
            code_template="""
// Register Optimization 优化
// 方法1: 使用 launch_bounds 限制寄存器
__launch_bounds__({max_threads}, {min_blocks})
__global__ void kernel(...) {{
    // 方法2: 减少局部变量
    // 合并相关计算，避免中间变量

    // 方法3: 使用更小的数据类型
    // float -> half (如果精度允许)

    // 方法4: 重用变量
    float temp = ...;  // 计算1
    // ... 使用 temp ...
    temp = ...;        // 复用计算2
}}
""",
            insertion_pattern=r"__global__\s+void",
            expected_speedup=1.5,
            complexity="medium",
            prerequisites=["寄存器使用率高", "occupancy 受限"]
        )

        # 7. Warp-level Primitives - 优化规约
        self.strategies["warp_primitives"] = OptimizationStrategy(
            name="Warp-level Primitives",
            description="使用 warp shuffle 指令优化规约操作",
            target_bottlenecks=[BottleneckType.COMPUTE_BOUND, BottleneckType.LATENCY_BOUND],
            applicable_metrics={
                "sm_busy": lambda x: x > 80,
                "synchronization_overhead": lambda x: x > 10
            },
            code_template="""
// Warp-level Primitives 优化
// 修改前: 使用共享内存规约
__shared__ float shared[32];
shared[threadIdx.x] = value;
__syncthreads();
// ... 规约循环 ...

// 修改后: 使用 warp shuffle
float val = value;
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {{
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}}

// 如果需要跨 warp
__shared__ float warpSums[32];
if (lane_id == 0) warpSums[warp_id] = val;
__syncthreads();
// 再对 warpSums 进行规约
""",
            insertion_pattern=r"__syncthreads\(\).*?for.*reduction|sum",
            expected_speedup=1.4,
            complexity="medium",
            prerequisites=["规约/求和操作", "SM 7.0+ (shuffle指令)"]
        )

        # 8. Grid-Stride Loops - 处理大数据集
        self.strategies["grid_stride"] = OptimizationStrategy(
            name="Grid-Stride Loops",
            description="使用 grid-stride 循环处理任意大小数据",
            target_bottlenecks=[BottleneckType.MIXED_BOUND],
            applicable_metrics={
                "grid_efficiency": lambda x: x < 90,
                "tail_effects": lambda x: x > 5
            },
            code_template="""
// Grid-Stride Loops 优化
// 修改前:
// int idx = blockIdx.x * blockDim.x + threadIdx.x;
// if (idx < n) {{ ... }}

// 修改后:
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = idx; i < n; i += stride) {{
    // 处理元素 i
    // 每个线程处理多个元素，提高占用率
}}
""",
            insertion_pattern=r"int\s+\w+\s*=\s*blockIdx\.\w+\s*\*\s*blockDim\.\w+\s*\+\s*threadIdx\.\w+",
            expected_speedup=1.2,
            complexity="low",
            prerequisites=["数据并行任务", "动态数据大小"]
        )

    def diagnose_bottleneck(self, metrics: Dict[str, float]) -> BottleneckType:
        """
        根据指标诊断瓶颈类型

        Args:
            metrics: 包含各项指标的字典
                - dram_throughput: DRAM 吞吐量 (%)
                - l1tex_throughput: L1/TEX 吞吐量 (%)
                - sm_busy: SM 忙碌程度 (%)
                - occupancy: 实际占用率 (%)
                - roofline_ratio: Roofline 性能比 (%)
                - l1_hit_rate: L1 命中率 (%)

        Returns:
            瓶颈类型
        """
        roofline = metrics.get('roofline_ratio', 0)
        dram = metrics.get('dram_throughput', 0)
        l1tex = metrics.get('l1tex_throughput', 0)
        sm_busy = metrics.get('sm_busy', 0)
        occupancy = metrics.get('occupancy', 0)
        l1_hit = metrics.get('l1_hit_rate', 100)

        # 决策树
        if roofline < 30:
            if dram > 70:
                return BottleneckType.DRAM_MEMORY_BOUND
            elif l1tex > 80 and dram < 30:
                return BottleneckType.L1_PRESSURE_BOUND
            elif l1_hit < 20:
                return BottleneckType.L1_PRESSURE_BOUND
            else:
                return BottleneckType.LATENCY_BOUND
        elif roofline > 60:
            if sm_busy > 80:
                return BottleneckType.COMPUTE_BOUND
            elif occupancy < 40:
                return BottleneckType.OCCUPANCY_BOUND
            else:
                return BottleneckType.COMPUTE_BOUND
        else:
            if occupancy < 30 and sm_busy > 70:
                return BottleneckType.OCCUPANCY_BOUND
            else:
                return BottleneckType.MIXED_BOUND

    def get_strategies_for_bottleneck(
        self,
        bottleneck: BottleneckType,
        metrics: Dict[str, float]
    ) -> List[OptimizationStrategy]:
        """
        获取适用于指定瓶颈的策略列表

        Args:
            bottleneck: 瓶颈类型
            metrics: 性能指标

        Returns:
            按优先级排序的策略列表
        """
        matched = []

        for name, strategy in self.strategies.items():
            # 检查瓶颈类型匹配
            if bottleneck not in strategy.target_bottlenecks:
                continue

            # 检查指标条件
            all_match = True
            for metric_name, condition in strategy.applicable_metrics.items():
                if metric_name in metrics:
                    if not condition(metrics[metric_name]):
                        all_match = False
                        break

            if all_match:
                matched.append(strategy)

        # 按预期收益排序
        matched.sort(key=lambda s: s.expected_speedup, reverse=True)
        return matched

    def get_strategy(self, name: str) -> Optional[OptimizationStrategy]:
        """获取指定名称的策略"""
        return self.strategies.get(name)

    def list_all_strategies(self) -> List[str]:
        """列出所有策略名称"""
        return list(self.strategies.keys())

    def generate_optimization_code(
        self,
        strategy_name: str,
        params: Dict[str, any]
    ) -> str:
        """
        生成优化代码

        Args:
            strategy_name: 策略名称
            params: 模板参数

        Returns:
            生成的代码
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return ""

        return strategy.code_template.format(**params)


# 便捷函数
def create_strategy_library() -> CUDAStrategyLibrary:
    """创建策略库实例"""
    return CUDAStrategyLibrary()


def diagnose_and_recommend(metrics: Dict[str, float]) -> Dict:
    """
    诊断瓶颈并推荐策略

    Args:
        metrics: NCU 采集的性能指标

    Returns:
        包含诊断结果和建议的字典
    """
    library = create_strategy_library()
    bottleneck = library.diagnose_bottleneck(metrics)
    strategies = library.get_strategies_for_bottleneck(bottleneck, metrics)

    return {
        "bottleneck": bottleneck.value,
        "bottleneck_description": _get_bottleneck_description(bottleneck),
        "strategies": [
            {
                "name": s.name,
                "description": s.description,
                "expected_speedup": s.expected_speedup,
                "complexity": s.complexity,
                "prerequisites": s.prerequisites
            }
            for s in strategies[:3]  # 返回前3个策略
        ]
    }


def _get_bottleneck_description(bottleneck: BottleneckType) -> str:
    """获取瓶颈类型描述"""
    descriptions = {
        BottleneckType.DRAM_MEMORY_BOUND: "DRAM 内存带宽受限，需要减少全局内存访问",
        BottleneckType.L1_PRESSURE_BOUND: "L1/TEX 缓存压力高，可能存在 bank conflict",
        BottleneckType.LATENCY_BOUND: "延迟受限，需要更多指令级并行",
        BottleneckType.COMPUTE_BOUND: "计算受限，已达到计算峰值",
        BottleneckType.OCCUPANCY_BOUND: "占用率受限，寄存器使用过多或 block 配置不当",
        BottleneckType.MIXED_BOUND: "混合瓶颈，需要综合优化",
        BottleneckType.UNKNOWN: "未知瓶颈类型"
    }
    return descriptions.get(bottleneck, "未知")


if __name__ == "__main__":
    # 测试策略库
    library = create_strategy_library()

    print("=== CUDA Optimization Strategy Library ===\n")
    print(f"可用策略: {library.list_all_strategies()}\n")

    # 测试诊断
    test_metrics = {
        "dram_throughput": 85,
        "l1tex_throughput": 40,
        "sm_busy": 45,
        "occupancy": 65,
        "roofline_ratio": 25,
        "l1_hit_rate": 5
    }

    result = diagnose_and_recommend(test_metrics)
    print(f"诊断结果: {result['bottleneck']}")
    print(f"描述: {result['bottleneck_description']}\n")
    print("推荐策略:")
    for i, s in enumerate(result['strategies'], 1):
        print(f"  {i}. {s['name']} (预期 {s['expected_speedup']}x 提升)")
        print(f"     复杂度: {s['complexity']}")
