#!/bin/bash
#
# 环境检查脚本
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        return 1
    fi
}

echo "========================================"
echo "  NCU CUDA Profiling Skill 环境检查"
echo "========================================"
echo ""

echo "📋 必要依赖:"
echo "-------------"

# CUDA
if check_command nvcc "CUDA Toolkit"; then
    nvcc --version | grep "release"
fi

# NCU
if check_command ncu "Nsight Compute (ncu)"; then
    ncu --version | grep "Version"
fi

# nvidia-smi
if check_command nvidia-smi "NVIDIA GPU Driver"; then
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

echo ""
echo "📋 可选依赖:"
echo "-------------"

# Python
if check_command python3 "Python 3"; then
    python3 --version
fi

# pip
if check_command pip3 "pip3"; then
    echo "   pip3 已安装"
fi

echo ""
echo "📋 环境变量:"
echo "-------------"
echo "   CUDA_PATH: ${CUDA_PATH:-"未设置"}"
echo "   PATH 包含 ncu: $(echo $PATH | grep -q ncu && echo "是" || echo "否/不确定")"

echo ""
echo "========================================"

# 总结
if command -v nvcc &> /dev/null && command -v ncu &> /dev/null; then
    echo -e "${GREEN}✅ 环境检查通过！${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  部分依赖缺失${NC}"
    echo ""
    echo "安装指南:"
    echo "  1. CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    echo "  2. Nsight Compute: 随 CUDA Toolkit 安装或单独下载"
    exit 1
fi
