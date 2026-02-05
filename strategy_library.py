#!/usr/bin/env python3
"""
CUDA Optimization Strategy Library v2
基于完整 NCU 指标的智能瓶颈诊断和优化策略推荐
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import re


class BottleneckType(Enum):
    """瓶颈类型定义 (v2 扩展)"""
    # Memory 相关
    DRAM_MEMORY_BOUND = "dram_memory_bound"
    L1_PRESSURE_BOUND = "l1_pressure_bound"
    L2_PRESSURE_BOUND = "l2_pressure_bound"
    
    # Compute 相关
    COMPUTE_BOUND = "compute_bound"
    COMPUTE_INTEGER_BOUND = "compute_integer_bound"
    COMPUTE_FP_BOUND = "compute_fp_bound"
    
    # Latency/Occupancy 相关
    LATENCY_BOUND = "latency_bound"
    OCCUPANCY_BOUND = "occupancy_bound"
    
    # Warp 停滞相关
    BARRIER_STALL_BOUND = "barrier_stall_bound"
    MEMORY_DEPENDENCY_STALL = "memory_dependency_stall"
    EXECUTION_DEPENDENCY_STALL = "execution_dependency_stall"
    
    # 混合
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
    """CUDA 优化策略库 v2 - 支持完整 NCU 指标诊断"""

    def __init__(self):
        self.strategies: Dict[str, OptimizationStrategy] = {}
        self._init_strategies()

    def _init_strategies(self):
        """初始化所有优化策略 (v2 扩展)"""

        # 1. Block Tiling - 解决 DRAM Memory Bound
        self.strategies["block_tiling"] = OptimizationStrategy(
            name="Block Tiling",
            description="使用共享内存进行块级数据缓存，减少全局内存访问",
            target_bottlenecks=[
                BottleneckType.DRAM_MEMORY_BOUND,
                BottleneckType.L2_PRESSURE_BOUND
            ],
            applicable_metrics={
                "dram_throughput": lambda x: x > 70,
                "l2_hit_rate": lambda x: x < 50,
                "sm_busy": lambda x: x < 50
            },
            code_template="""
// Block Tiling 优化
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
            target_bottlenecks=[
                BottleneckType.L1_PRESSURE_BOUND,
                BottleneckType.DRAM_MEMORY_BOUND
            ],
            applicable_metrics={
                "l1tex_throughput": lambda x: x > 80,
                "l1_hit_rate": lambda x: x < 50,
                "dram_throughput": lambda x: x < 40
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
                "sm_busy": lambda x: x < 60
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
            target_bottlenecks=[
                BottleneckType.LATENCY_BOUND,
                BottleneckType.MEMORY_DEPENDENCY_STALL
            ],
            applicable_metrics={
                "sm_busy": lambda x: x < 50,
                "occupancy": lambda x: x > 60,
                "stall_memory_dependency": lambda x: x > 30
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
            target_bottlenecks=[
                BottleneckType.LATENCY_BOUND,
                BottleneckType.COMPUTE_BOUND,
                BottleneckType.EXECUTION_DEPENDENCY_STALL
            ],
            applicable_metrics={
                "executed_ipc_active": lambda x: x < 0.5,
                "stall_execution_dependency": lambda x: x > 20
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
            target_bottlenecks=[
                BottleneckType.COMPUTE_BOUND,
                BottleneckType.BARRIER_STALL_BOUND
            ],
            applicable_metrics={
                "sm_busy": lambda x: x > 80,
                "stall_barrier": lambda x: x > 10
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
                "occupancy": lambda x: x < 50,
                "grid_size": lambda x: x < 100  # 假设 grid size 小
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

        # 9. Reduce Barrier - 减少同步
        self.strategies["reduce_barrier"] = OptimizationStrategy(
            name="Reduce Barrier Synchronization",
            description="减少 __syncthreads() 调用或使用 warp-level 原语替代",
            target_bottlenecks=[BottleneckType.BARRIER_STALL_BOUND],
            applicable_metrics={
                "stall_barrier": lambda x: x > 20,
                "sm_busy": lambda x: x < 60
            },
            code_template="""
// 减少 Barrier 优化
// 策略1: 合并同步点
// 修改前:
// __syncthreads();
// ... 一些计算 ...
// __syncthreads();

// 修改后:
// ... 合并计算 ...
// __syncthreads();

// 策略2: 使用 warp shuffle 替代共享内存规约
// 减少 __syncthreads 调用次数
""",
            insertion_pattern=r"__syncthreads\(\)",
            expected_speedup=1.3,
            complexity="medium",
            prerequisites=["频繁的 barrier 同步", "可合并的同步点"]
        )

        # 10. Increase ILP - 增加指令级并行
        self.strategies["increase_ilp"] = OptimizationStrategy(
            name="Increase ILP",
            description="增加指令级并行度，隐藏指令延迟",
            target_bottlenecks=[
                BottleneckType.EXECUTION_DEPENDENCY_STALL,
                BottleneckType.LATENCY_BOUND
            ],
            applicable_metrics={
                "stall_execution_dependency": lambda x: x > 30,
                "executed_ipc_active": lambda x: x < 1.0
            },
            code_template="""
// Increase ILP 优化
// 策略: 独立计算指令交错

// 修改前 (依赖链长):
// float a = load(i);
// float b = compute(a);
// float c = compute2(b);

// 修改后 (交错独立计算):
float a0 = load(i);
float a1 = load(i+1);
float b0 = compute(a0);
float b1 = compute(a1);
float c0 = compute2(b0);
float c1 = compute2(b1);
// 更多独立指令...
""",
            insertion_pattern=r"for\s*\(\s*int\s+\w+\s*=\s*0",
            expected_speedup=1.2,
            complexity="medium",
            prerequisites=["存在独立计算", "长依赖链"]
        )

    def diagnose_bottleneck(self, metrics: Dict[str, float]) -> BottleneckType:
        """
        根据完整 NCU 指标智能诊断瓶颈类型 (v2 增强版)
        
        决策树 (按优先级):
        1. Speed Of Light Throughput - 判断 Memory vs Compute
        2. Warp Stall Reasons - 细粒度停滞分析
        3. Occupancy & Scheduler - 调度效率
        
        Args:
            metrics: 包含完整 NCU 指标的字典
                - Speed Of Light: memory_throughput, dram_throughput, sm_throughput
                - Compute: sm_busy, issue_slots_busy, executed_ipc_active
                - Memory: l1_hit_rate, l2_hit_rate
                - Occupancy: occupancy, theoretical_occupancy
                - Scheduler: active_warps, eligible_warps, no_eligible
                - Warp Stall: stall_barrier, stall_memory_dependency, stall_execution_dependency
        
        Returns:
            瓶颈类型
        """
        # === Level 1: Speed Of Light Throughput Analysis ===
        memory_throughput = metrics.get('memory_throughput', 0)
        dram_throughput = metrics.get('dram_throughput', 0)
        sm_throughput = metrics.get('sm_throughput', 0)
        l1tex_throughput = metrics.get('l1tex_throughput', 0)
        l2_throughput = metrics.get('l2_throughput', 0)
        
        # === Level 2: Compute & Memory Details ===
        sm_busy = metrics.get('sm_busy', 0)
        issue_slots_busy = metrics.get('issue_slots_busy', 0)
        executed_ipc = metrics.get('executed_ipc_active', 0)
        
        l1_hit_rate = metrics.get('l1_hit_rate', 100)
        l2_hit_rate = metrics.get('l2_hit_rate', 100)
        
        # === Level 3: Occupancy & Scheduler ===
        occupancy = metrics.get('occupancy', 0)
        theoretical_occupancy = metrics.get('theoretical_occupancy', 0)
        no_eligible = metrics.get('no_eligible', 0)
        
        # === Level 4: Warp Stall Reasons ===
        stall_barrier = metrics.get('stall_barrier', 0)
        stall_memory_dep = metrics.get('stall_memory_dependency', 0)
        stall_execution_dep = metrics.get('stall_execution_dependency', 0)
        stall_memory_throttle = metrics.get('stall_memory_throttle', 0)
        
        # === Decision Tree ===
        
        # 1. Memory Bound Check
        if dram_throughput > 80 and sm_throughput < 50:
            # DRAM 压力大，进一步细分
            if l1_hit_rate < 30 and l1tex_throughput > 80:
                return BottleneckType.L1_PRESSURE_BOUND
            elif l2_hit_rate < 50 and l2_throughput > 80:
                return BottleneckType.L2_PRESSURE_BOUND
            else:
                return BottleneckType.DRAM_MEMORY_BOUND
        
        # 2. L1 Pressure Check (高 L1 吞吐量但低 DRAM)
        if l1tex_throughput > 80 and dram_throughput < 30 and l1_hit_rate < 50:
            return BottleneckType.L1_PRESSURE_BOUND
        
        # 3. Compute Bound Check
        if sm_throughput > 80 and dram_throughput < 50:
            if sm_busy > 80:
                return BottleneckType.COMPUTE_BOUND
            else:
                # Throughput 高但 SM Busy 低可能是测量误差
                pass
        
        # 4. Warp Stall Analysis (当 SM Busy 低时)
        if sm_busy < 50:
            # 分析 stall reasons
            if stall_barrier > 20:
                return BottleneckType.BARRIER_STALL_BOUND
            elif stall_memory_dep > 30:
                return BottleneckType.MEMORY_DEPENDENCY_STALL
            elif stall_execution_dep > 30:
                return BottleneckType.EXECUTION_DEPENDENCY_STALL
            else:
                # 没有特定 stall，可能是一般延迟
                return BottleneckType.LATENCY_BOUND
        
        # 5. Occupancy Check
        if occupancy < 30:
            if theoretical_occupancy > 50:
                # 理论高但实际低 → 可能是 workload imbalance
                return BottleneckType.OCCUPANCY_BOUND
        
        # 6. Issue Slot Utilization
        if issue_slots_busy < 40 and no_eligible > 30:
            # Issue slot 空闲多，warp 停滞
            if stall_memory_dep > stall_execution_dep:
                return BottleneckType.MEMORY_DEPENDENCY_STALL
            else:
                return BottleneckType.EXECUTION_DEPENDENCY_STALL
        
        # 7. Mixed Check
        if sm_throughput > 60 and dram_throughput > 60:
            return BottleneckType.MIXED_BOUND
        
        # Default: Unknown
        return BottleneckType.UNKNOWN

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

    def get_bottleneck_description(self, bottleneck: BottleneckType) -> str:
        """获取瓶颈类型详细描述"""
        descriptions = {
            BottleneckType.DRAM_MEMORY_BOUND: 
                "DRAM 内存带宽受限，需要减少全局内存访问。建议: Block Tiling, Vectorized Load",
            BottleneckType.L1_PRESSURE_BOUND: 
                "L1/TEX 缓存压力高，可能存在 bank conflict。建议: Shared Memory Padding",
            BottleneckType.L2_PRESSURE_BOUND: 
                "L2 缓存压力高，数据复用不足。建议: 调整 tile 大小",
            BottleneckType.COMPUTE_BOUND: 
                "计算受限，已达到计算峰值。建议: 使用 Tensor Core, FMA",
            BottleneckType.COMPUTE_INTEGER_BOUND: 
                "整数运算瓶颈。建议: 使用浮点运算替代",
            BottleneckType.COMPUTE_FP_BOUND: 
                "浮点运算瓶颈。建议: 使用 Tensor Core",
            BottleneckType.LATENCY_BOUND: 
                "延迟受限，需要更多指令级并行。建议: Double Buffering, Loop Unrolling",
            BottleneckType.OCCUPANCY_BOUND: 
                "占用率受限，寄存器使用过多或 block 配置不当。建议: Launch Bounds",
            BottleneckType.BARRIER_STALL_BOUND: 
                "同步屏障等待时间长。建议: 减少同步点, Warp Primitives",
            BottleneckType.MEMORY_DEPENDENCY_STALL: 
                "内存依赖等待。建议: Double Buffering, ILP Increase",
            BottleneckType.EXECUTION_DEPENDENCY_STALL: 
                "执行依赖等待。建议: Loop Unrolling, ILP Increase",
            BottleneckType.MIXED_BOUND: 
                "混合瓶颈，需要综合优化。建议: 分阶段优化",
            BottleneckType.UNKNOWN: 
                "未知瓶颈类型，需要进一步分析"
        }
        return descriptions.get(bottleneck, "未知")


# 便捷函数
def create_strategy_library() -> CUDAStrategyLibrary:
    """创建策略库实例"""
    return CUDAStrategyLibrary()


def diagnose_and_recommend(metrics: Dict[str, float]) -> Dict:
    """
    诊断瓶颈并推荐策略 (v2 增强版)

    Args:
        metrics: NCU 采集的完整性能指标

    Returns:
        包含诊断结果和建议的字典
    """
    library = create_strategy_library()
    bottleneck = library.diagnose_bottleneck(metrics)
    strategies = library.get_strategies_for_bottleneck(bottleneck, metrics)

    # 获取主要 stall reasons
    stall_reasons = {
        k: v for k, v in metrics.items() 
        if k.startswith('stall_') and v > 5  # 只显示 >5% 的 stall
    }

    return {
        "bottleneck": bottleneck.value,
        "bottleneck_description": library.get_bottleneck_description(bottleneck),
        "stall_reasons": stall_reasons,
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


if __name__ == "__main__":
    # 测试策略库
    library = create_strategy_library()

    print("=== CUDA Optimization Strategy Library v2 ===\n")
    print(f"可用策略: {library.list_all_strategies()}\n")

    # 测试诊断 - DRAM Memory Bound
    print("--- Test Case 1: DRAM Memory Bound ---")
    test_metrics = {
        "dram_throughput": 85,
        "memory_throughput": 90,
        "sm_throughput": 35,
        "l1tex_throughput": 40,
        "sm_busy": 45,
        "occupancy": 65,
        "l1_hit_rate": 60,
        "stall_memory_dependency": 40
    }

    result = diagnose_and_recommend(test_metrics)
    print(f"诊断结果: {result['bottleneck']}")
    print(f"描述: {result['bottleneck_description']}")
    print(f"主要 Stall: {result['stall_reasons']}")
    print("推荐策略:")
    for i, s in enumerate(result['strategies'], 1):
        print(f"  {i}. {s['name']} (预期 {s['expected_speedup']}x 提升)")
        print(f"     复杂度: {s['complexity']}")
    print()

    # 测试诊断 - Barrier Stall
    print("--- Test Case 2: Barrier Stall ---")
    test_metrics2 = {
        "dram_throughput": 30,
        "sm_throughput": 50,
        "sm_busy": 40,
        "occupancy": 70,
        "stall_barrier": 35,
        "stall_memory_dependency": 10
    }

    result2 = diagnose_and_recommend(test_metrics2)
    print(f"诊断结果: {result2['bottleneck']}")
    print(f"描述: {result2['bottleneck_description']}")
    print(f"主要 Stall: {result2['stall_reasons']}")
    print("推荐策略:")
    for i, s in enumerate(result2['strategies'], 1):
        print(f"  {i}. {s['name']} (预期 {s['expected_speedup']}x 提升)")
