#!/usr/bin/env python3
"""
NCU CUDA Optimizer v2 - Enhanced with Comprehensive Metrics
支持交互式和全自动两种优化模式，包含完整的 NCU 指标解析
"""

import os
import re
import sys
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import argparse

from strategy_library import (
    CUDAStrategyLibrary,
    BottleneckType,
    diagnose_and_recommend
)


@dataclass
class OptimizationVersion:
    """优化版本记录"""
    version_id: str
    iteration: int
    code_path: str
    strategy_name: str
    strategy_description: str
    metrics: Dict[str, float]
    speedup_vs_baseline: float
    speedup_vs_previous: float
    build_success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OptimizationState:
    """优化状态"""
    project_dir: str
    source_file: str
    baseline_code: str
    baseline_metrics: Dict[str, float]
    versions: List[OptimizationVersion] = field(default_factory=list)
    current_iteration: int = 0
    best_version_id: str = "baseline"
    converged: bool = False
    convergence_reason: str = ""


class NCUProfiler:
    """NCU 性能分析器 - 支持全量指标采集和解析"""

    def __init__(self, ncu_path: str = "ncu"):
        self.ncu_path = ncu_path
        # 扩展的指标名称映射 (使用子字符串匹配)
        # 按照分析优先级排序
        self.metrics_map = {
            # === Speed Of Light Throughput (首要) ===
            "gpu_time": "gpu__time_duration.avg",
            "memory_throughput": "gpu__memory_throughput.avg.pct",
            "dram_throughput": "gpu__dram_throughput.avg.pct",
            "sm_throughput": "sm__throughput.avg.pct",
            "l1tex_throughput": "l1tex__throughput.avg.pct",
            "l2_throughput": "lts__throughput.avg.pct",
            
            # === Compute Workload Analysis ===
            "sm_busy": "sm__cycles_active.avg.pct",
            "issue_slots_busy": "smsp__issue_active.avg.pct",
            "executed_ipc_active": "smsp__ipc.avg",
            
            # === Memory Workload Analysis ===
            "l1_hit_rate": "l1tex__t_sector_hit_rate.pct",
            "l2_hit_rate": "lts__t_sector_hit_rate.pct",
            "mem_busy": "gpu__mem_busy.avg.pct",
            
            # === Occupancy ===
            "occupancy": "sm__occupancy.avg.pct",
            "theoretical_occupancy": "sm__theoretical_occupancy.avg.pct",
            
            # === Scheduler Statistics ===
            "active_warps": "smsp__warps_active.avg",
            "eligible_warps": "smsp__warps_eligible.avg",
            "issued_warps": "smsp__issue_warps.avg",
            "no_eligible": "smsp__warps_no_eligible.avg.pct",
            
            # === Warp State Statistics (Stall Reasons) ===
            "stall_wait": "smsp__warp_issue_stalled_wait.avg.pct",
            "stall_barrier": "smsp__warp_issue_stalled_barrier.avg.pct",
            "stall_memory_dependency": "smsp__warp_issue_stalled_memory_dependency.avg.pct",
            "stall_execution_dependency": "smsp__warp_issue_stalled_execution_dependency.avg.pct",
            "stall_memory_throttle": "smsp__warp_issue_stalled_memory_throttle.avg.pct",
            "stall_instruction_fetch": "smsp__warp_issue_stalled_inst_fetch.avg.pct",
            "stall_texture": "smsp__warp_issue_stalled_texture.avg.pct",
            "stall_constant": "smsp__warp_issue_stalled_constant_memory_dependency.avg.pct",
            "stall_not_selected": "smsp__warp_issue_stalled_not_selected.avg.pct",
            
            # === Launch Statistics ===
            "registers_per_thread": "launch__registers_per_thread",
            "shared_memory_per_block": "launch__shared_mem_configured_size",
            "block_size": "launch__block_size",
            "grid_size": "launch__grid_size",
        }

    def profile_from_report(self, report_path: str) -> Tuple[bool, Dict[str, float]]:
        """
        从已有的 NCU 报告文件导入分析

        Args:
            report_path: .ncu-rep 文件路径

        Returns:
            (success, metrics)
        """
        if not os.path.exists(report_path):
            print(f"报告文件不存在: {report_path}")
            return False, {}

        # 创建临时 CSV
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "imported_metrics.csv")

        try:
            # 导出 CSV
            cmd = [
                self.ncu_path,
                "--import", report_path,
                "--page", "raw",
                "--csv"
            ]

            with open(csv_path, 'w') as f:
                result = subprocess.run(cmd, stdout=f, timeout=60)
                if result.returncode != 0:
                    return False, {}

            # 解析指标
            metrics = self._parse_metrics(csv_path)
            return True, metrics

        except Exception as e:
            print(f"导入报告失败: {e}")
            return False, {}
        finally:
            # 清理临时文件
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def profile(self, executable: str, report_name: str) -> Tuple[bool, Dict[str, float]]:
        """
        运行 NCU 性能分析

        Returns:
            (success, metrics)
        """
        report_path = f"{report_name}.ncu-rep"
        csv_path = f"{report_name}.csv"

        # 运行 NCU 采集
        cmd = [
            self.ncu_path,
            "--set", "full",
            "-o", report_name,
            "--target-processes", "all",
            "--force-overwrite",
            executable
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0 and not os.path.exists(report_path):
                print(f"NCU 运行失败: {result.stderr}")
                return False, {}

            # 导出 CSV
            self._export_csv(report_path, csv_path)

            # 解析指标
            metrics = self._parse_metrics(csv_path)
            return True, metrics

        except subprocess.TimeoutExpired:
            print("NCU 分析超时")
            return False, {}
        except Exception as e:
            print(f"NCU 分析异常: {e}")
            return False, {}

    def _export_csv(self, report_path: str, csv_path: str):
        """导出 CSV 格式报告"""
        cmd = [
            self.ncu_path,
            "--import", report_path,
            "--page", "raw",
            "--csv"
        ]

        try:
            with open(csv_path, 'w') as f:
                subprocess.run(cmd, stdout=f, timeout=60)
        except Exception as e:
            print(f"CSV 导出失败: {e}")

    def _parse_metrics(self, csv_path: str) -> Dict[str, float]:
        """
        解析 NCU CSV 报告提取关键指标
        
        支持全量指标解析，包括：
        - Speed Of Light Throughput
        - Compute Workload Analysis
        - Memory Workload Analysis
        - Occupancy
        - Scheduler Statistics
        - Warp State Statistics
        """
        metrics = {}

        if not os.path.exists(csv_path):
            return metrics

        try:
            import csv
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) < 3:
                return metrics

            # 找到 kernel 执行行
            header = rows[0]

            # NCU CSV 格式：第0行是表头，第1行是空行/单位行，第2行开始是数据
            # 从第3行开始读取实际数据（索引2）
            for row_data in rows[2:]:
                if len(row_data) != len(header):
                    continue

                row = dict(zip(header, row_data))

                # 提取关键指标 (使用子字符串匹配)
                for key, metric_pattern in self.metrics_map.items():
                    # 查找匹配的列
                    for col_name, col_value in row.items():
                        if metric_pattern in col_name:
                            try:
                                # 处理带单位的值 (如 "17.66 us", "8.34 %")
                                clean_value = col_value.replace('"', '').strip()
                                # 分割数值和单位
                                parts = clean_value.split()
                                if parts:
                                    # 确保第一个部分是数字
                                    value = float(parts[0])
                                    metrics[key] = value
                                    # 记录时间单位用于后续转换
                                    if 'time' in key and len(parts) > 1:
                                        metrics[key + '_unit'] = parts[1]
                            except (ValueError, TypeError, IndexError):
                                pass
                            break  # 找到第一个匹配就停止

                # 如果有 kernel 名称，记录下来
                if 'Kernel Name' in row:
                    metrics['kernel_name'] = row['Kernel Name']

                # 只处理第一个有效的 kernel 行
                if metrics.get('gpu_time'):
                    break

        except Exception as e:
            print(f"解析指标失败: {e}")

        return metrics

    def print_metrics_summary(self, metrics: Dict[str, float]):
        """打印指标摘要（用于调试）"""
        print("\n" + "="*60)
        print("NCU Metrics Summary")
        print("="*60)
        
        # Speed Of Light
        print("\n📊 Speed Of Light Throughput:")
        for key in ['memory_throughput', 'dram_throughput', 'sm_throughput', 
                    'l1tex_throughput', 'l2_throughput']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.2f}%")
        
        # Compute
        print("\n🔢 Compute Workload:")
        for key in ['sm_busy', 'issue_slots_busy', 'executed_ipc_active']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.2f}")
        
        # Memory
        print("\n💾 Memory Workload:")
        for key in ['l1_hit_rate', 'l2_hit_rate', 'mem_busy']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.2f}%")
        
        # Occupancy
        print("\n📈 Occupancy:")
        for key in ['occupancy', 'theoretical_occupancy']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.2f}%")
        
        # Scheduler
        print("\n⚡ Scheduler Stats:")
        for key in ['active_warps', 'eligible_warps', 'no_eligible']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.2f}")
        
        # Stall Reasons
        print("\n⏸️  Warp Stall Reasons:")
        stall_keys = [k for k in metrics.keys() if k.startswith('stall_')]
        for key in sorted(stall_keys):
            print(f"  {key}: {metrics[key]:.2f}%")
        
        print("="*60 + "\n")


class CodeModifier:
    """CUDA 代码修改器"""

    def __init__(self, strategy_library: CUDAStrategyLibrary):
        self.library = strategy_library

    def apply_strategy(
        self,
        code: str,
        strategy_name: str,
        params: Dict[str, any]
    ) -> Tuple[bool, str]:
        """
        应用优化策略到代码

        Returns:
            (success, modified_code)
        """
        strategy = self.library.get_strategy(strategy_name)
        if not strategy:
            return False, code

        # 生成优化代码片段
        optimization_code = self.library.generate_optimization_code(
            strategy_name, params
        )

        # 根据策略类型应用修改
        if strategy_name == "block_tiling":
            return self._apply_block_tiling(code, optimization_code, params)
        elif strategy_name == "smem_padding":
            return self._apply_smem_padding(code, params)
        elif strategy_name == "vectorized_load":
            return self._apply_vectorized_load(code, params)
        elif strategy_name == "loop_unrolling":
            return self._apply_loop_unrolling(code, params)
        elif strategy_name == "register_opt":
            return self._apply_register_opt(code, params)
        elif strategy_name == "warp_primitives":
            return self._apply_warp_primitives(code, params)
        elif strategy_name == "double_buffering":
            return self._apply_double_buffering(code, params)
        else:
            # 通用策略：在 kernel 开头插入优化代码
            return self._insert_at_kernel_start(code, optimization_code)

    def _apply_block_tiling(
        self,
        code: str,
        optimization_code: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """应用 Block Tiling 优化"""
        lines = code.split('\n')
        modified = []
        inserted = False

        for i, line in enumerate(lines):
            # 在 kernel 函数开始后插入共享内存声明
            if '__global__' in line and 'void' in line and not inserted:
                modified.append(line)
                # 找到函数体的开始
                j = i + 1
                while j < len(lines) and '{' not in lines[j]:
                    modified.append(lines[j])
                    j += 1
                if j < len(lines):
                    modified.append(lines[j])  # {
                    # 插入共享内存声明
                    bm = params.get('bm', 32)
                    bn = params.get('bn', 32)
                    bk = params.get('bk', 8)
                    modified.append(f'    // Block Tiling Optimization')
                    modified.append(f'    const int BM = {bm};')
                    modified.append(f'    const int BN = {bn};')
                    modified.append(f'    const int BK = {bk};')
                    modified.append(f'    __shared__ float As[BM][BK];')
                    modified.append(f'    __shared__ float Bs[BK][BN];')
                    inserted = True
                    i = j
            else:
                modified.append(line)

        return inserted, '\n'.join(modified)

    def _apply_smem_padding(
        self,
        code: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """应用 Shared Memory Padding"""
        pad = params.get('pad', 1)

        # 查找共享内存声明并添加 padding
        pattern = r'(__shared__\s+\w+\s+\w+)\[(\d+)\]\[(\d+)\]'

        def replace_with_padding(match):
            base = match.group(1)
            dim1 = match.group(2)
            dim2 = match.group(3)
            return f'{base}[{dim1}][{dim2} + {pad}]  // Padding to avoid bank conflict'

        modified_code = re.sub(pattern, replace_with_padding, code)
        success = modified_code != code

        return success, modified_code

    def _apply_vectorized_load(
        self,
        code: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """应用 Vectorized Load 优化"""
        modified = code

        # 添加 float4 类型定义（如果不存在）
        if 'float4' not in code:
            modified = 'struct alignas(16) float4 { float x, y, z, w; };\n' + modified

        return True, modified

    def _apply_loop_unrolling(
        self,
        code: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """应用 Loop Unrolling 优化"""
        unroll_factor = params.get('unroll_factor', 4)

        # 查找 for 循环并添加 #pragma unroll
        pattern = r'(\n\s*)(for\s*\(\s*int\s+(\w+)\s*=\s*0)'

        def add_pragma(match):
            indent = match.group(1)
            loop_start = match.group(2)
            return f'{indent}#pragma unroll {unroll_factor}{indent}{loop_start}'

        modified_code = re.sub(pattern, add_pragma, code)
        success = modified_code != code

        return success, modified_code

    def _apply_register_opt(
        self,
        code: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """应用 Register 优化"""
        max_threads = params.get('max_threads', 256)
        min_blocks = params.get('min_blocks', 2)

        # 在 __global__ 前添加 __launch_bounds__
        pattern = r'(__global__\s+void)'
        replacement = f'__launch_bounds__({max_threads}, {min_blocks})\n__global__ void'

        modified_code = re.sub(pattern, replacement, code)
        success = modified_code != code

        return success, modified_code

    def _apply_warp_primitives(
        self,
        code: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """应用 Warp-level Primitives 优化"""
        # 这是一个复杂的转换，简化版本只添加注释提示
        marker = '// WARP_PRIMITIVE_OPTIMIZATION: Consider using __shfl_down_sync for reduction'
        if marker in code:
            return False, code

        # 在包含 __syncthreads 的规约操作附近添加提示
        lines = code.split('\n')
        modified = []

        for line in lines:
            modified.append(line)
            if '__syncthreads()' in line:
                modified.append(marker)

        return True, '\n'.join(modified)

    def _apply_double_buffering(
        self,
        code: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """应用 Double Buffering 优化"""
        # 简化实现：在 kernel 开头添加双缓冲声明
        bm = params.get('bm', 32)
        bn = params.get('bn', 32)
        bk = params.get('bk', 8)

        lines = code.split('\n')
        modified = []
        inserted = False

        for i, line in enumerate(lines):
            if '__global__' in line and 'void' in line and not inserted:
                modified.append(line)
                j = i + 1
                while j < len(lines) and '{' not in lines[j]:
                    modified.append(lines[j])
                    j += 1
                if j < len(lines):
                    modified.append(lines[j])
                    modified.append(f'    // Double Buffering Optimization')
                    modified.append(f'    __shared__ float As[2][{bm}][{bk}];')
                    modified.append(f'    __shared__ float Bs[2][{bk}][{bn}];')
                    modified.append(f'    int compute_stage = 0, load_stage = 0;')
                    inserted = True
                    i = j
            else:
                modified.append(line)

        return inserted, '\n'.join(modified)

    def _insert_at_kernel_start(
        self,
        code: str,
        optimization_code: str
    ) -> Tuple[bool, str]:
        """在 kernel 函数开头插入代码"""
        lines = code.split('\n')
        modified = []
        inserted = False

        for i, line in enumerate(lines):
            if '__global__' in line and not inserted:
                modified.append(line)
                # 找到 {
                j = i + 1
                while j < len(lines) and '{' not in lines[j]:
                    modified.append(lines[j])
                    j += 1
                if j < len(lines):
                    modified.append(lines[j])
                    modified.append(optimization_code)
                    inserted = True
                    i = j
            else:
                modified.append(line)

        return inserted, '\n'.join(modified)


class CUDAOptimizer:
    """CUDA 优化器主类"""

    CONVERGENCE_THRESHOLD = 0.03  # 3% 提升阈值
    MAX_ITERATIONS = 5

    def __init__(
        self,
        source_file: str,
        build_command: str,
        mode: str = "auto",
        ncu_path: str = "ncu"
    ):
        self.source_file = Path(source_file)
        self.build_command = build_command
        self.mode = mode
        self.ncu_profiler = NCUProfiler(ncu_path)
        self.strategy_library = CUDAStrategyLibrary()
        self.code_modifier = CodeModifier(self.strategy_library)

        # 创建工作目录
        self.work_dir = Path(tempfile.mkdtemp(prefix="ncu_opt_"))
        self.state = None

    def analyze_only(self, save_to_project: bool = True) -> Dict:
        """
        仅分析性能，不执行优化 (对应 v1 功能)

        Args:
            save_to_project: 是否将报告保存到项目目录

        Returns:
            分析结果字典
        """
        print(f"{'='*60}")
        print("NCU CUDA Profiler - Analysis Mode (v2 Enhanced)")
        print(f"{'='*60}")
        print(f"Source: {self.source_file}")
        print()

        # 编译
        executable = self.work_dir / "analyze_target"
        if not self._build(str(self.source_file), str(executable)):
            return {"success": False, "error": "Failed to build target"}

        # 运行 NCU
        success, metrics = self.ncu_profiler.profile(
            str(executable),
            str(self.work_dir / "analysis_report")
        )

        if not success:
            return {"success": False, "error": "Failed to profile"}

        # 打印完整指标摘要
        self.ncu_profiler.print_metrics_summary(metrics)

        # 诊断瓶颈
        bottleneck = self.strategy_library.diagnose_bottleneck(metrics)
        recommendations = self.strategy_library.get_strategies_for_bottleneck(
            bottleneck, metrics
        )

        # 生成分析报告
        report = self._generate_analysis_report(metrics, bottleneck, recommendations)

        # 保存到项目目录 (如果需要)
        if save_to_project:
            self._save_to_project_dir(report, metrics)

        return {
            "success": True,
            "metrics": metrics,
            "bottleneck": bottleneck.value,
            "recommendations": [r.name for r in recommendations[:3]],
            "report": report
        }

    def analyze_from_report(self, report_path: str) -> Dict:
        """
        从已有的 NCU 报告分析 (对应 v1 功能)

        Args:
            report_path: .ncu-rep 文件路径

        Returns:
            分析结果字典
        """
        print(f"{'='*60}")
        print("NCU CUDA Profiler - Import Mode (v2 Enhanced)")
        print(f"{'='*60}")
        print(f"Report: {report_path}")
        print()

        # 从报告导入
        success, metrics = self.ncu_profiler.profile_from_report(report_path)

        if not success:
            return {"success": False, "error": "Failed to import report"}

        # 打印完整指标摘要
        self.ncu_profiler.print_metrics_summary(metrics)

        # 诊断瓶颈
        bottleneck = self.strategy_library.diagnose_bottleneck(metrics)
        recommendations = self.strategy_library.get_strategies_for_bottleneck(
            bottleneck, metrics
        )

        # 生成分析报告
        report = self._generate_analysis_report(metrics, bottleneck, recommendations)

        return {
            "success": True,
            "metrics": metrics,
            "bottleneck": bottleneck.value,
            "recommendations": [r.name for r in recommendations[:3]],
            "report": report
        }

    def _save_to_project_dir(self, report: str, metrics: Dict):
        """保存分析结果到项目目录 (v1 功能)"""
        # 创建 ncu_reports 目录
        reports_dir = Path("ncu_reports")
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = Path(self.source_file).stem

        # 保存报告
        report_path = reports_dir / f"{source_name}_{timestamp}_analysis.md"
        with open(report_path, 'w') as f:
            f.write(report)

        # 保存指标 JSON
        metrics_path = reports_dir / f"{source_name}_{timestamp}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n📁 Report saved to: {report_path}")
        print(f"📊 Metrics saved to: {metrics_path}")

    def _generate_analysis_report(
        self,
        metrics: Dict[str, float],
        bottleneck: 'BottleneckType',
        recommendations: List
    ) -> str:
        """生成分析报告 (v2 增强版)"""
        report_lines = [
            "# NCU 性能分析报告 (v2)",
            "",
            f"**分析时间**: {datetime.now().isoformat()}",
            f"**源文件**: {self.source_file}",
            "",
            "## 执行摘要",
            "",
            f"| 项目 | 数值 |",
            f"|------|------|",
            f"| **主要瓶颈** | {bottleneck.value} |",
            f"| **GPU 执行时间** | {metrics.get('gpu_time', 'N/A')} μs |",
        ]
        
        # Speed Of Light Throughput
        report_lines.extend([
            "",
            "## Speed Of Light Throughput",
            "",
            "| 指标 | 数值 | 状态 |",
            "|------|------|------|",
        ])
        for key in ['memory_throughput', 'dram_throughput', 'sm_throughput', 'l1tex_throughput']:
            if key in metrics:
                val = metrics[key]
                status = "⚠️" if val > 80 else "✅"
                report_lines.append(f"| {key} | {val:.2f}% | {status} |")
        
        # Compute Workload
        report_lines.extend([
            "",
            "## Compute Workload",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
        ])
        for key in ['sm_busy', 'issue_slots_busy', 'executed_ipc_active']:
            if key in metrics:
                report_lines.append(f"| {key} | {metrics[key]:.2f} |")
        
        # Memory Workload
        report_lines.extend([
            "",
            "## Memory Workload",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
        ])
        for key in ['l1_hit_rate', 'l2_hit_rate']:
            if key in metrics:
                report_lines.append(f"| {key} | {metrics[key]:.2f}% |")
        
        # Occupancy
        report_lines.extend([
            "",
            "## Occupancy",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
        ])
        for key in ['occupancy', 'theoretical_occupancy']:
            if key in metrics:
                report_lines.append(f"| {key} | {metrics[key]:.2f}% |")
        
        # Warp Stall Reasons
        stall_keys = [k for k in metrics.keys() if k.startswith('stall_')]
        if stall_keys:
            report_lines.extend([
                "",
                "## Warp Stall Reasons",
                "",
                "| 指标 | 数值 |",
                "|------|------|",
            ])
            for key in sorted(stall_keys):
                report_lines.append(f"| {key} | {metrics[key]:.2f}% |")
        
        # 关键指标汇总
        report_lines.extend([
            "",
            "## 关键指标汇总",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
        ])

        for key, value in metrics.items():
            if not key.endswith('_unit') and not key.startswith('stall_'):
                if isinstance(value, float):
                    report_lines.append(f"| {key} | {value:.2f} |")
                else:
                    report_lines.append(f"| {key} | {value} |")

        report_lines.extend([
            "",
            "## 诊断详情",
            "",
            f"**瓶颈类型**: {bottleneck.value}",
            "",
            "**优化建议**:",
            ""
        ])

        for i, strategy in enumerate(recommendations[:3], 1):
            report_lines.extend([
                f"{i}. **{strategy.name}** (预期 {strategy.expected_speedup}x 提升)",
                f"   - {strategy.description}",
                f"   - 复杂度: {strategy.complexity}",
                ""
            ])

        return '\n'.join(report_lines)

    def run(self) -> Dict:
        """
        运行优化流程

        Returns:
            优化结果字典
        """
        print(f"{'='*60}")
        print(f"NCU CUDA Optimizer v2 - {self.mode.upper()} Mode")
        print(f"{'='*60}")
        print(f"Source: {self.source_file}")
        print(f"Work Directory: {self.work_dir}")
        print(f"Max Iterations: {self.MAX_ITERATIONS}")
        print(f"Convergence Threshold: {self.CONVERGENCE_THRESHOLD*100}%")
        print()

        # 1. 保存 baseline
        if not self._setup_baseline():
            return {"success": False, "error": "Failed to setup baseline"}

        # 2. 运行优化循环
        while self.state.current_iteration < self.MAX_ITERATIONS:
            if self.state.converged:
                break

            self.state.current_iteration += 1
            print(f"\n{'-'*60}")
            print(f"Iteration {self.state.current_iteration}/{self.MAX_ITERATIONS}")
            print(f"{'-'*60}")

            success = self._run_iteration()
            if not success:
                print(f"Iteration {self.state.current_iteration} failed, stopping...")
                break

        # 3. 生成最终报告
        return self._generate_report()

    def _setup_baseline(self) -> bool:
        """设置 baseline"""
        print("Setting up baseline...")

        # 读取源代码
        try:
            with open(self.source_file, 'r') as f:
                baseline_code = f.read()
        except Exception as e:
            print(f"Failed to read source file: {e}")
            return False

        # 保存 baseline 到工作目录
        baseline_path = self.work_dir / "baseline.cu"
        with open(baseline_path, 'w') as f:
            f.write(baseline_code)

        # 编译并分析 baseline
        executable = self.work_dir / "baseline"
        if not self._build(str(baseline_path), str(executable)):
            print("Failed to build baseline")
            return False

        success, metrics = self.ncu_profiler.profile(
            str(executable),
            str(self.work_dir / "baseline_report")
        )

        if not success:
            print("Failed to profile baseline")
            return False

        # 打印指标摘要
        self.ncu_profiler.print_metrics_summary(metrics)

        self.state = OptimizationState(
            project_dir=str(self.work_dir),
            source_file=str(self.source_file),
            baseline_code=baseline_code,
            baseline_metrics=metrics,
            current_iteration=0,
            versions=[]
        )

        return True

    def _run_iteration(self) -> bool:
        """运行单次优化迭代"""
        iteration = self.state.current_iteration

        # 获取当前最佳版本的代码
        current_code = self._get_best_code()

        # 诊断瓶颈
        current_metrics = self._get_best_metrics()
        bottleneck = self.strategy_library.diagnose_bottleneck(current_metrics)
        print(f"\nDiagnosed bottleneck: {bottleneck.value}")

        # 获取推荐策略
        strategies = self.strategy_library.get_strategies_for_bottleneck(
            bottleneck, current_metrics
        )

        if not strategies:
            print("No applicable strategies found")
            self.state.converged = True
            self.state.convergence_reason = "No applicable strategies"
            return False

        # 选择策略
        selected_strategy = strategies[0]
        print(f"Selected strategy: {selected_strategy.name}")
        print(f"  Description: {selected_strategy.description}")
        print(f"  Expected speedup: {selected_strategy.expected_speedup}x")

        # 交互式模式：询问用户确认
        if self.mode == "interactive":
            if not self._ask_user_confirmation(selected_strategy):
                print("User skipped this strategy")
                return self._try_next_strategy(strategies[1:], current_code)

        # 生成策略参数
        params = self._generate_strategy_params(selected_strategy.name)

        # 应用优化
        success, modified_code = self.code_modifier.apply_strategy(
            current_code,
            selected_strategy.name,
            params
        )

        if not success:
            print(f"Failed to apply strategy: {selected_strategy.name}")
            return self._try_next_strategy(strategies[1:], current_code)

        # 保存新版本
        version_id = f"v{iteration}"
        version_path = self.work_dir / f"{version_id}.cu"
        with open(version_path, 'w') as f:
            f.write(modified_code)

        # 编译新版本
        executable = self.work_dir / version_id
        if not self._build(str(version_path), str(executable)):
            print(f"Build failed for {version_id}")
            return False

        # 性能测试
        success, new_metrics = self.ncu_profiler.profile(
            str(executable),
            str(self.work_dir / f"{version_id}_report")
        )

        if not success:
            print(f"Profiling failed for {version_id}")
            return False

        # 计算加速比
        speedup_vs_baseline = self._calculate_speedup(
            new_metrics, self.state.baseline_metrics
        )
        speedup_vs_previous = self._calculate_speedup(
            new_metrics, current_metrics
        )

        print(f"\nResults:")
        print(f"  Speedup vs baseline: {speedup_vs_baseline:.2f}x")
        print(f"  Speedup vs previous: {speedup_vs_previous:.2f}x")

        # 创建版本记录
        version = OptimizationVersion(
            version_id=version_id,
            iteration=iteration,
            code_path=str(version_path),
            strategy_name=selected_strategy.name,
            strategy_description=selected_strategy.description,
            metrics=new_metrics,
            speedup_vs_baseline=speedup_vs_baseline,
            speedup_vs_previous=speedup_vs_previous,
            build_success=True
        )
        self.state.versions.append(version)

        # 检查是否性能下降，需要回滚
        if speedup_vs_previous < 0.95:  # 下降超过 5%
            print(f"⚠️  Performance regression detected ({speedup_vs_previous:.2f}x)")
            print("Auto-rolling back to previous version...")
            return self._handle_regression(version)

        # 更新最佳版本
        if speedup_vs_baseline > self._get_best_speedup():
            self.state.best_version_id = version_id
            print(f"✅ New best version: {version_id}")

        # 检查收敛
        if speedup_vs_previous < (1 + self.CONVERGENCE_THRESHOLD):
            print(f"📊 Convergence detected (improvement < {self.CONVERGENCE_THRESHOLD*100}%)")
            self.state.converged = True
            self.state.convergence_reason = f"Diminishing returns ({speedup_vs_previous:.3f}x)"

        return True

    def _try_next_strategy(
        self,
        strategies: List,
        code: str
    ) -> bool:
        """尝试下一个策略"""
        if not strategies:
            return False
        # 简化处理：直接返回失败，让主循环处理
        return False

    def _handle_regression(self, version: OptimizationVersion) -> bool:
        """处理性能下降"""
        # 标记为失败版本
        version.build_success = False

        # 在交互模式下询问用户
        if self.mode == "interactive":
            response = input("Continue with next strategy? [y/n]: ")
            if response.lower() != 'y':
                self.state.converged = True
                self.state.convergence_reason = "User stopped after regression"
                return False

        return True

    def _ask_user_confirmation(self, strategy) -> bool:
        """交互模式下询问用户确认"""
        print(f"\n{'='*40}")
        print("Strategy Application Confirmation")
        print(f"{'='*40}")
        print(f"Strategy: {strategy.name}")
        print(f"Description: {strategy.description}")
        print(f"Expected speedup: {strategy.expected_speedup}x")
        print(f"Complexity: {strategy.complexity}")
        print(f"Prerequisites: {', '.join(strategy.prerequisites)}")
        print()

        response = input("Apply this strategy? [y/n/skip]: ").lower()
        return response == 'y'

    def _build(self, source: str, output: str) -> bool:
        """编译 CUDA 代码"""
        # 解析原始构建命令并替换输入输出
        cmd = self.build_command.replace("{source}", source).replace("{output}", output)

        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Build error: {e}")
            return False

    def _get_best_code(self) -> str:
        """获取当前最佳版本的代码"""
        if self.state.best_version_id == "baseline":
            return self.state.baseline_code

        for v in self.state.versions:
            if v.version_id == self.state.best_version_id:
                with open(v.code_path, 'r') as f:
                    return f.read()

        return self.state.baseline_code

    def _get_best_metrics(self) -> Dict[str, float]:
        """获取当前最佳版本的指标"""
        if self.state.best_version_id == "baseline":
            return self.state.baseline_metrics

        for v in self.state.versions:
            if v.version_id == self.state.best_version_id:
                return v.metrics

        return self.state.baseline_metrics

    def _get_best_speedup(self) -> float:
        """获取当前最佳加速比"""
        if not self.state.versions:
            return 1.0

        best = max(v.speedup_vs_baseline for v in self.state.versions)
        return best

    def _calculate_speedup(
        self,
        new_metrics: Dict[str, float],
        old_metrics: Dict[str, float]
    ) -> float:
        """计算加速比 - 以 GPU 执行时间为主要指标"""
        # 优先使用 gpu_time (kernel 执行时间，单位纳秒)
        new_time = new_metrics.get('gpu_time', 0)
        old_time = old_metrics.get('gpu_time', 0)

        if new_time > 0 and old_time > 0:
            # 加速比 = 旧时间 / 新时间 (时间越短越好)
            return old_time / new_time

        # 回退到 sm_busy 作为性能指标
        new_perf = new_metrics.get('sm_busy', 0)
        old_perf = old_metrics.get('sm_busy', 0)

        if old_perf == 0:
            return 1.0

        return new_perf / old_perf

    def _format_time(self, nanoseconds: float) -> str:
        """格式化时间为人类可读格式"""
        if nanoseconds == 0:
            return "N/A"
        if nanoseconds >= 1e9:
            return f"{nanoseconds / 1e9:.2f}s"
        elif nanoseconds >= 1e6:
            return f"{nanoseconds / 1e6:.2f}ms"
        elif nanoseconds >= 1e3:
            return f"{nanoseconds / 1e3:.2f}μs"
        else:
            return f"{nanoseconds:.2f}ns"

    def _generate_strategy_params(self, strategy_name: str) -> Dict:
        """生成策略参数"""
        defaults = {
            "block_tiling": {"bm": 32, "bn": 32, "bk": 8},
            "smem_padding": {"pad": 1},
            "vectorized_load": {},
            "loop_unrolling": {"unroll_factor": 4},
            "register_opt": {"max_threads": 256, "min_blocks": 2},
            "warp_primitives": {},
            "double_buffering": {"bm": 32, "bn": 32, "bk": 8},
            "grid_stride": {}
        }

        return defaults.get(strategy_name, {})

    def _generate_report(self) -> Dict:
        """生成优化报告"""
        report_path = self.work_dir / "optimization_report.md"

        # 生成 Markdown 报告
        report_lines = [
            "# NCU CUDA 自动优化报告 (v2)",
            "",
            f"**优化时间**: {datetime.now().isoformat()}",
            f"**源文件**: {self.source_file}",
            f"**优化模式**: {self.mode}",
            f"**迭代次数**: {self.state.current_iteration}/{self.MAX_ITERATIONS}",
            "",
            "## 优化概览",
            "",
            f"- **初始执行时间**: {self._format_time(self.state.baseline_metrics.get('gpu_time', 0))}",
            f"- **初始 SM Busy**: {self.state.baseline_metrics.get('sm_busy', 0):.1f}%",
            f"- **初始 Memory Throughput**: {self.state.baseline_metrics.get('memory_throughput', 0):.1f}%",
        ]

        if self.state.versions:
            best_metrics = self._get_best_metrics()
            best_speedup = self._get_best_speedup()
            report_lines.extend([
                f"- **最终执行时间**: {self._format_time(best_metrics.get('gpu_time', 0))}",
                f"- **最终 SM Busy**: {best_metrics.get('sm_busy', 0):.1f}%",
                f"- **总加速比**: {best_speedup:.2f}x (以执行时间为准)",
                f"- **最佳版本**: {self.state.best_version_id}",
                f"- **收敛状态**: {'已收敛' if self.state.converged else '未收敛'}",
            ])

            if self.state.convergence_reason:
                report_lines.append(f"- **收敛原因**: {self.state.convergence_reason}")

        report_lines.extend([
            "",
            "## 优化历程",
            "",
            "| 版本 | 策略 | 执行时间 | SM Busy | 相对Baseline | 相对上一轮 | 状态 |",
            "|------|------|----------|---------|--------------|------------|------|",
        ])

        # Baseline
        baseline_sm = self.state.baseline_metrics.get('sm_busy', 0)
        baseline_time = self.state.baseline_metrics.get('gpu_time', 0)
        report_lines.append(
            f"| baseline | - | {self._format_time(baseline_time)} | {baseline_sm:.1f}% | 1.00x | - | ✅ |"
        )

        # 每个版本
        for v in self.state.versions:
            sm = v.metrics.get('sm_busy', 0)
            time = v.metrics.get('gpu_time', 0)
            status = "✅" if v.build_success else "❌"
            report_lines.append(
                f"| {v.version_id} | {v.strategy_name} | {self._format_time(time)} | {sm:.1f}% | "
                f"{v.speedup_vs_baseline:.2f}x | {v.speedup_vs_previous:.2f}x | {status} |"
            )

        report_lines.extend([
            "",
            "## 详细分析",
            "",
        ])

        for v in self.state.versions:
            report_lines.extend([
                f"### {v.version_id}: {v.strategy_name}",
                "",
                f"- **策略描述**: {v.strategy_description}",
                f"- **相对Baseline提升**: {v.speedup_vs_baseline:.2f}x",
                f"- **相对上一轮提升**: {v.speedup_vs_previous:.2f}x",
                "",
                "**关键指标**:",
                "",
            ])

            for key, value in v.metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"- {key}: {value:.2f}")

            report_lines.append("")

        # 保存报告
        report_content = '\n'.join(report_lines)
        with open(report_path, 'w') as f:
            f.write(report_content)

        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print(f"{'='*60}")
        print(f"Report saved to: {report_path}")

        # 保存最终最佳代码
        if self.state.best_version_id != "baseline":
            best_code = self._get_best_code()
            best_code_path = self.work_dir / "best_optimized.cu"
            with open(best_code_path, 'w') as f:
                f.write(best_code)
            print(f"Best code saved to: {best_code_path}")

        # 返回结果字典
        return {
            "success": True,
            "work_dir": str(self.work_dir),
            "report_path": str(report_path),
            "best_version": self.state.best_version_id,
            "best_speedup": self._get_best_speedup(),
            "iterations": self.state.current_iteration,
            "converged": self.state.converged,
            "versions": [asdict(v) for v in self.state.versions]
        }


def main():
    parser = argparse.ArgumentParser(
        description="NCU CUDA Optimizer v2 - Analysis & Optimization"
    )

    # 主要参数
    parser.add_argument(
        "source",
        nargs="?",
        help="CUDA source file to optimize/analyze"
    )

    # 模式选择
    parser.add_argument(
        "--mode",
        choices=["auto", "interactive", "analyze"],
        default="auto",
        help=("Mode: auto=全自动优化, interactive=交互式优化, "
              "analyze=只分析不优化 (default: auto)")
    )

    # 从报告导入 (v1 功能)
    parser.add_argument(
        "--import-report",
        metavar="REPORT",
        help="从已有的 .ncu-rep 文件导入分析 (分析模式)"
    )

    # 编译相关
    parser.add_argument(
        "--build",
        default="nvcc -O3 {source} -o {output}",
        help="Build command template"
    )

    # 保存选项
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存报告到项目目录 (analyze 模式)"
    )

    # 其他参数
    parser.add_argument(
        "--ncu-path",
        default="ncu",
        help="Path to ncu executable"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum optimization iterations"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Convergence threshold"
    )

    args = parser.parse_args()

    # 处理导入报告模式
    if args.import_report:
        # 创建临时 optimizer (不需要 source file)
        profiler = NCUProfiler(args.ncu_path)
        success, metrics = profiler.profile_from_report(args.import_report)

        if success:
            library = CUDAStrategyLibrary()
            bottleneck = library.diagnose_bottleneck(metrics)
            recommendations = library.get_strategies_for_bottleneck(
                bottleneck, metrics
            )

            # 打印指标摘要
            profiler.print_metrics_summary(metrics)

            print(f"\n{'='*60}")
            print("NCU Analysis Report (Imported)")
            print(f"{'='*60}")
            print(f"\n📊 Bottleneck: {bottleneck.value}")
            print(f"⏱️  GPU Time: {metrics.get('gpu_time', 'N/A')} μs")
            print(f"\n💡 Recommendations:")
            for i, r in enumerate(recommendations[:3], 1):
                print(f"  {i}. {r.name} ({r.expected_speedup}x)")

            sys.exit(0)
        else:
            print("❌ Failed to import report")
            sys.exit(1)

    # 检查 source file
    if not args.source:
        parser.error("source file is required (unless using --import-report)")

    # 创建 optimizer
    optimizer = CUDAOptimizer(
        source_file=args.source,
        build_command=args.build,
        mode=args.mode if args.mode in ["auto", "interactive"] else "auto",
        ncu_path=args.ncu_path
    )

    # 覆盖默认参数
    CUDAOptimizer.MAX_ITERATIONS = args.max_iter
    CUDAOptimizer.CONVERGENCE_THRESHOLD = args.threshold

    # 执行对应模式
    if args.mode == "analyze":
        # 分析模式 (v1 功能)
        result = optimizer.analyze_only(save_to_project=not args.no_save)
        if result["success"]:
            print("\n✅ Analysis completed!")
            print(f"📊 Bottleneck: {result['bottleneck']}")
            print(f"💡 Top recommendation: {result['recommendations'][0] if result['recommendations'] else 'N/A'}")
            sys.exit(0)
        else:
            print(f"\n❌ Analysis failed: {result.get('error')}")
            sys.exit(1)
    else:
        # 优化模式 (v2 功能)
        result = optimizer.run()
        if result["success"]:
            print("\n✅ Optimization completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Optimization failed: {result.get('error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
