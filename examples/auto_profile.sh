#!/bin/bash
# NCU 自动化性能分析脚本
# 使用方法: ./auto_profile.sh <kernel_executable> [output_prefix]
# 示例: ./auto_profile.sh ./matmul my_report

set -e

KERNEL=$1
PREFIX=${2:-"ncu_report_$(date +%Y%m%d_%H%M%S)"}
REPORT_DIR="ncu_reports"

if [ -z "$KERNEL" ]; then
    echo "Usage: ./auto_profile.sh <kernel_executable> [output_prefix]"
    echo "Example: ./auto_profile.sh ./matmul my_analysis"
    exit 1
fi

# 检查 ncu 是否可用
if ! command -v ncu &> /dev/null; then
    # 尝试常见 CUDA 路径
    if [ -x "/usr/local/cuda/bin/ncu" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    else
        echo "Error: ncu not found. Please ensure CUDA toolkit is installed."
        exit 1
    fi
fi

# 创建报告目录
mkdir -p "$REPORT_DIR"

echo "🚀 开始 NCU 自动化性能分析..."
echo "================================"
echo "目标: $KERNEL"
echo "报告前缀: $PREFIX"
echo "报告目录: $REPORT_DIR"
echo ""

# Phase 1: 完整采集 (使用 --set full)
echo "📊 Phase 1: 完整指标采集 (--set full)..."
echo "This may take a while..."

ncu --set full \
    -o "${REPORT_DIR}/${PREFIX}" \
    --target-processes all \
    --force-overwrite \
    "$KERNEL" 2>&1 | tee "${REPORT_DIR}/${PREFIX}_ncu_log.txt"

echo "✅ 完整报告已生成: ${REPORT_DIR}/${PREFIX}.ncu-rep"
echo "   日志文件: ${REPORT_DIR}/${PREFIX}_ncu_log.txt"
echo ""

# Phase 2: 提取关键指标到 CSV
echo "📈 Phase 2: 提取关键性能指标..."

ncu --import "${REPORT_DIR}/${PREFIX}.ncu-rep" \
    --page raw \
    --csv \
    > "${REPORT_DIR}/${PREFIX}_raw.csv" 2>/dev/null

echo "✅ 指标已提取: ${REPORT_DIR}/${PREFIX}_raw.csv"
echo ""

# Phase 3: 生成摘要报告
echo "🔍 Phase 3: 生成性能摘要..."

ncu --import "${REPORT_DIR}/${PREFIX}.ncu-rep" \
    --print-summary per-kernel \
    > "${REPORT_DIR}/${PREFIX}_summary.txt" 2>/dev/null

echo "✅ 摘要已生成: ${REPORT_DIR}/${PREFIX}_summary.txt"
echo ""

# Phase 4: Python 深度分析 (如果可用)
echo "🤖 Phase 4: 运行 AI 分析..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${SCRIPT_DIR}/ncu_analyzer.py" ]; then
    python3 "${SCRIPT_DIR}/ncu_analyzer.py" \
        --import "${REPORT_DIR}/${PREFIX}.ncu-rep" \
        -o "${REPORT_DIR}/${PREFIX}_analysis.md" 2>/dev/null || {
        echo "⚠️  Python 分析器运行失败，跳过 AI 分析"
    }
else
    echo "⚠️  Python 分析器未找到，跳过 AI 分析"
fi

echo ""
echo "🎉 分析完成!"
echo "================================"
echo "生成的文件:"
echo "  📄 ${REPORT_DIR}/${PREFIX}.ncu-rep      (完整 NCU 报告)"
echo "  📊 ${REPORT_DIR}/${PREFIX}_raw.csv      (原始指标 CSV)"
echo "  📝 ${REPORT_DIR}/${PREFIX}_summary.txt  (性能摘要)"
echo "  📋 ${REPORT_DIR}/${PREFIX}_ncu_log.txt  (NCU 日志)"
if [ -f "${REPORT_DIR}/${PREFIX}_analysis.md" ]; then
    echo "  🤖 ${REPORT_DIR}/${PREFIX}_analysis.md  (AI 分析报告)"
fi
echo ""
echo "💡 后续操作:"
echo "  查看摘要: ncu --import ${REPORT_DIR}/${PREFIX}.ncu-rep --print-summary per-kernel"
echo "  查看详情: ncu --import ${REPORT_DIR}/${PREFIX}.ncu-rep --page details"
echo "  导出 CSV: ncu --import ${REPORT_DIR}/${PREFIX}.ncu-rep --page raw --csv"
