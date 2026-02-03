#!/bin/bash
#
# NCU CUDA Optimizer v2 - 一键安装脚本
# 支持多 AI Agent 安装
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# AI Agent 安装路径
CLAUDE_PATH="$HOME/.claude/skills/ncu-cuda-optimizer-v2"
KIMI_PATH="$HOME/.config/agents/skills/ncu-cuda-optimizer-v2"
CURSOR_PATH="$HOME/.cursor/rules/ncu-cuda-optimizer-v2"
CODEX_PATH="$HOME/.codex/skills/ncu-cuda-optimizer-v2"

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 打印消息
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示帮助
show_help() {
    cat << EOF
NCU CUDA Optimizer v2 安装脚本

用法: $0 [选项]

选项:
    -h, --help              显示帮助信息
    --claude                安装到 Claude Code (默认)
    --kimi                  安装到 Kimi Code CLI
    --cursor                安装到 Cursor
    --codex                 安装到 Codex
    --all-agents            安装到所有支持的 AI Agent
    --uninstall             卸载
    --check                 检查环境依赖

示例:
    $0                      # 默认安装到 Claude Code
    $0 --claude             # 同上
    $0 --all-agents         # 安装到所有 Agent
    $0 --check              # 检查环境
    $0 --uninstall          # 卸载

EOF
}

# 检查环境
check_environment() {
    print_info "检查环境依赖..."
    local all_good=true

    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "✓ CUDA 已安装: $NVCC_VERSION"
    else
        print_error "✗ CUDA 未安装或不在 PATH 中"
        all_good=false
    fi

    if command -v ncu &> /dev/null; then
        NCU_VERSION=$(ncu --version 2>/dev/null | grep -i "version" | head -1 | awk '{print $3}' || echo "unknown")
        print_success "✓ NCU 已安装: $NCU_VERSION"
    else
        print_error "✗ NCU (Nsight Compute) 未安装"
        all_good=false
    fi

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        print_success "✓ $PYTHON_VERSION"
    else
        print_warning "✗ Python3 未安装"
        all_good=false
    fi

    if [ "$all_good" = true ]; then
        print_success "环境检查通过！"
        return 0
    else
        print_error "环境检查未通过"
        return 1
    fi
}

# 安装 skill 文件
install_skill_files() {
    local target_path=$1
    local agent_type=$2

    print_info "安装 NCU CUDA Optimizer v2 到 $target_path..."

    # 创建目录
    mkdir -p "$target_path"

    # 复制核心文件
    cp "$SCRIPT_DIR/SKILL.md" "$target_path/"
    cp "$SCRIPT_DIR/optimizer.py" "$target_path/"
    cp "$SCRIPT_DIR/strategy_library.py" "$target_path/"

    # Cursor 特殊处理
    if [ "$agent_type" = "cursor" ]; then
        mv "$target_path/SKILL.md" "$target_path/ncu-cuda-optimizer-v2.md" 2>/dev/null || true
    fi

    print_success "✓ 已安装到 $target_path"
}

# 创建 CLI 工具
install_cli_tools() {
    local target_path=$1

    mkdir -p "$target_path/bin"

    # ncu-optimize 命令
    cat > "$target_path/bin/ncu-optimize" << EOF
#!/bin/bash
# NCU Optimize v2 快捷命令

SCRIPT_PATH="$target_path/optimizer.py"

if [ \$# -eq 0 ]; then
    echo "用法: ncu-optimize <cuda_file.cu> [options]"
    echo ""
    echo "模式:"
    echo "  --mode=auto          全自动优化 (默认)"
    echo "  --mode=interactive   交互式优化"
    echo "  --mode=analyze       只分析不优化"
    echo ""
    echo "示例:"
    echo "  ncu-optimize matmul.cu"
    echo "  ncu-optimize matmul.cu --mode=analyze"
    exit 1
fi

python3 "\$SCRIPT_PATH" "\$@"
EOF
    chmod +x "$target_path/bin/ncu-optimize"

    print_success "✓ CLI 工具已安装 (ncu-optimize)"
}

# 安装到 Claude Code
install_claude() {
    print_info "安装到 Claude Code..."
    install_skill_files "$CLAUDE_PATH" "claude"
    install_cli_tools "$CLAUDE_PATH"

    echo ""
    print_success "Claude Code 安装完成！"
    echo ""
    echo "使用方式:"
    echo "  1. 直接用自然语言: '帮我全自动优化这个 CUDA 算子'"
    echo "  2. 分析模式: '分析这个 kernel 的性能瓶颈'"
    echo "  3. 交互模式: '一步步优化让我确认每一步'"
    echo ""
    echo "命令行:"
    echo "  ncu-optimize matmul.cu --mode=auto"
    echo "  ncu-optimize matmul.cu --mode=analyze"
}

# 安装到 Kimi
install_kimi() {
    print_info "安装到 Kimi Code CLI..."
    install_skill_files "$KIMI_PATH" "kimi"
    install_cli_tools "$KIMI_PATH"

    echo ""
    print_success "Kimi Code CLI 安装完成！"
}

# 安装到 Cursor
install_cursor() {
    print_info "安装到 Cursor..."
    install_skill_files "$CURSOR_PATH" "cursor"

    echo ""
    print_success "Cursor 安装完成！"
}

# 安装到 Codex
install_codex() {
    print_info "安装到 Codex..."
    install_skill_files "$CODEX_PATH" "codex"
    install_cli_tools "$CODEX_PATH"

    echo ""
    print_success "Codex 安装完成！"
}

# 卸载
uninstall_skill() {
    print_info "卸载 NCU CUDA Optimizer v2..."

    for path in "$CLAUDE_PATH" "$KIMI_PATH" "$CURSOR_PATH" "$CODEX_PATH"; do
        if [ -d "$path" ]; then
            rm -rf "$path"
            print_success "✓ 已删除 $path"
        fi
    done

    print_success "卸载完成！"
}

# 主函数
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        --check)
            check_environment
            exit $?
            ;;
        --uninstall)
            uninstall_skill
            exit 0
            ;;
        --kimi)
            check_environment && install_kimi
            ;;
        --cursor)
            check_environment && install_cursor
            ;;
        --codex)
            check_environment && install_codex
            ;;
        --all-agents)
            check_environment
            install_claude
            install_kimi
            install_cursor
            install_codex
            print_success "所有 Agent 安装完成！"
            ;;
        ""|--claude)
            check_environment && install_claude
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
