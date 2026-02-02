#!/bin/bash
#
# NCU CUDA Profiling Skill - 一键安装脚本
# 支持系统级安装、用户级安装和多 AI Agent 安装
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认安装路径
DEFAULT_SYSTEM_PATH="/opt/ncu-cuda-profiling-skill"
DEFAULT_USER_PATH="$HOME/.config/agents/skills/ncu-cuda-profiling"

# AI Agent 安装路径
KIMI_PATH="$HOME/.config/agents/skills/ncu-cuda-profiling"
CLAUDE_PATH="$HOME/.claude/skills/ncu-cuda-profiling"
CURSOR_PATH="$HOME/.cursor/rules/ncu-cuda-profiling"
CODEX_PATH="$HOME/.codex/skills/ncu-cuda-profiling"

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
NCU CUDA Profiling Skill 安装脚本

用法: $0 [选项]

选项:
    -h, --help              显示帮助信息
    -t, --target PATH       指定安装目录
    -s, --system            系统级安装 (需要 sudo)
    -u, --user              用户级安装 (默认, 安装到 Kimi Code CLI)
    
    # AI Agent 安装选项
    --kimi                  安装到 Kimi Code CLI (默认)
    --claude                安装到 Claude Code
    --cursor                安装到 Cursor (rules)
    --codex                 安装到 Codex
    --all-agents            安装到所有支持的 AI Agent
    
    --uninstall             卸载
    --check                 检查环境依赖

示例:
    $0                      # 默认安装到 Kimi Code CLI
    $0 --kimi               # 同上
    $0 --claude             # 安装到 Claude Code
    $0 --all-agents         # 安装到所有 Agent
    $0 --system             # 系统级安装
    $0 --target ~/.config/agents/skills/  # 安装到指定目录
    $0 --check              # 检查环境
    $0 --uninstall          # 卸载

EOF
}

# 检查环境依赖
check_environment() {
    print_info "检查环境依赖..."
    
    local all_good=true
    
    # 检查 CUDA
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "✓ CUDA 已安装: $NVCC_VERSION"
    else
        print_error "✗ CUDA 未安装或不在 PATH 中"
        all_good=false
    fi
    
    # 检查 NCU
    if command -v ncu &> /dev/null; then
        NCU_VERSION=$(ncu --version | grep "Version" | awk '{print $2}')
        print_success "✓ NCU 已安装: $NCU_VERSION"
    else
        print_error "✗ NCU (Nsight Compute) 未安装或不在 PATH 中"
        print_info "  请从 https://developer.nvidia.com/nsight-compute 下载安装"
        all_good=false
    fi
    
    # 检查 nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_success "✓ GPU: $GPU_INFO"
    else
        print_warning "✗ nvidia-smi 不可用，可能未安装 NVIDIA 驱动"
        all_good=false
    fi
    
    # 检查 Python (可选)
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        print_success "✓ $PYTHON_VERSION"
    else
        print_warning "✗ Python3 未安装 (可选，用于高级分析)"
    fi
    
    if [ "$all_good" = true ]; then
        print_success "环境检查通过！"
        return 0
    else
        print_error "环境检查未通过，请安装缺失的依赖"
        return 1
    fi
}

# 安装 skill 核心文件
install_skill_files() {
    local target_path=$1
    local agent_type=$2
    
    print_info "安装 NCU CUDA Profiling Skill 到 $target_path..."
    
    # 创建目录
    mkdir -p "$target_path"
    
    # 复制核心文件
    cp "$SCRIPT_DIR/SKILL.md" "$target_path/"
    
    # 复制 examples
    if [ -d "$SCRIPT_DIR/examples" ]; then
        cp -r "$SCRIPT_DIR/examples" "$target_path/"
    fi
    
    # 根据 agent 类型调整
    case $agent_type in
        cursor)
            # Cursor 使用 .md 后缀
            mv "$target_path/SKILL.md" "$target_path/ncu-cuda-profiling.md" 2>/dev/null || true
            ;;
        *)
            # 其他 agent 保持默认
            ;;
    esac
    
    print_success "✓ 已安装到 $target_path"
}

# 创建命令行工具
install_cli_tools() {
    local target_path=$1
    
    # 创建 bin 目录
    mkdir -p "$target_path/bin"
    
    # 创建 ncu-profile 快捷命令
    cat > "$target_path/bin/ncu-profile" << 'EOF'
#!/bin/bash
# NCU Profile 快捷命令

if [ $# -eq 0 ]; then
    echo "用法: ncu-profile <kernel_executable> [args...]"
    echo "示例: ncu-profile ./matmul"
    exit 1
fi

KERNEL=$1
shift
REPORT_NAME="ncu_report_$(date +%Y%m%d_%H%M%S)"

echo "🚀 开始 NCU 性能分析..."
echo "   Kernel: $KERNEL"
echo "   报告: $REPORT_NAME.ncu-rep"

ncu --set full \
    -o "$REPORT_NAME" \
    --target-processes all \
    "$KERNEL" "$@"

echo ""
echo "✅ 分析完成！"
echo "   报告文件: $REPORT_NAME.ncu-rep"
echo "   查看结果: ncu --import $REPORT_NAME.ncu-rep --print-summary per-kernel"
EOF
    chmod +x "$target_path/bin/ncu-profile"
    
    # 创建 ncu-analyze 快捷命令
    cat > "$target_path/bin/ncu-analyze" << 'EOF'
#!/bin/bash
# NCU Analyze 快捷命令

if [ $# -eq 0 ]; then
    echo "用法: ncu-analyze <report.ncu-rep>"
    echo "示例: ncu-analyze my_report.ncu-rep"
    exit 1
fi

REPORT=$1

if [ ! -f "$REPORT" ]; then
    echo "错误: 找不到报告文件 $REPORT"
    exit 1
fi

echo "📊 NCU 报告分析"
echo "==============="
ncu --import "$REPORT" --print-summary per-kernel
EOF
    chmod +x "$target_path/bin/ncu-analyze"
    
    print_success "✓ CLI 工具已安装"
}

# 安装到 Kimi Code CLI
install_kimi() {
    print_info "安装到 Kimi Code CLI..."
    install_skill_files "$KIMI_PATH" "kimi"
    install_cli_tools "$KIMI_PATH"
    
    echo ""
    print_success "Kimi Code CLI 安装完成！"
    echo ""
    echo "使用方式:"
    echo "  启动 Kimi Code CLI 后，skill 会自动加载"
    echo "  你可以直接询问: '帮我分析这个 CUDA kernel'"
}

# 安装到 Claude Code
install_claude() {
    print_info "安装到 Claude Code..."
    install_skill_files "$CLAUDE_PATH" "claude"
    
    echo ""
    print_success "Claude Code 安装完成！"
    echo ""
    echo "使用方式:"
    echo "  启动 Claude Code 后，可以使用 /skill ncu-cuda-profiling 加载"
    echo "  或直接询问: '使用 ncu-cuda-profiling skill 分析这个 kernel'"
}

# 安装到 Cursor
install_cursor() {
    print_info "安装到 Cursor..."
    install_skill_files "$CURSOR_PATH" "cursor"
    
    echo ""
    print_success "Cursor 安装完成！"
    echo ""
    echo "使用方式:"
    echo "  Cursor 会自动读取 ~/.cursor/rules/ 下的规则文件"
    echo "  你也可以在项目根目录创建 .cursorrules 文件"
}

# 安装到 Codex
install_codex() {
    print_info "安装到 Codex..."
    install_skill_files "$CODEX_PATH" "codex"
    
    echo ""
    print_success "Codex 安装完成！"
    echo ""
    echo "使用方式:"
    echo "  Codex 会在处理 CUDA 相关问题时自动引用 skill 内容"
}

# 通用安装函数
install_skill() {
    local target_path=$1
    
    print_info "安装 NCU CUDA Profiling Skill..."
    print_info "目标目录: $target_path"
    
    install_skill_files "$target_path" "generic"
    install_cli_tools "$target_path"
    
    # 添加到 PATH 的提示
    print_success "安装完成！"
    echo ""
    echo "⚠️  请添加以下路径到您的 PATH:"
    echo "   export PATH=\"$target_path/bin:\$PATH\""
    echo ""
    echo "你可以将其添加到 ~/.bashrc 或 ~/.zshrc:"
    echo "   echo 'export PATH=\"$target_path/bin:\$PATH\"' >> ~/.bashrc"
    echo ""
    echo "📖 使用说明:"
    echo "   ncu-profile ./your_kernel    # 运行分析"
    echo "   ncu-analyze report.ncu-rep   # 分析已有报告"
}

# 卸载函数
uninstall_skill() {
    local target_path=$1
    
    if [ -d "$target_path" ]; then
        print_warning "将删除目录: $target_path"
        read -p "确认卸载? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$target_path"
            print_success "已卸载"
        else
            print_info "取消卸载"
        fi
    else
        print_warning "未找到安装目录: $target_path"
    fi
}

# 主函数
main() {
    local install_path=""
    local do_check=false
    local do_uninstall=false
    local install_type="user"
    local install_kimi=false
    local install_claude=false
    local install_cursor=false
    local install_codex=false
    local install_all=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--target)
                install_path="$2"
                install_type="custom"
                shift 2
                ;;
            -s|--system)
                install_type="system"
                shift
                ;;
            -u|--user)
                install_type="user"
                install_kimi=true
                shift
                ;;
            --kimi)
                install_kimi=true
                shift
                ;;
            --claude)
                install_claude=true
                shift
                ;;
            --cursor)
                install_cursor=true
                shift
                ;;
            --codex)
                install_codex=true
                shift
                ;;
            --all-agents)
                install_all=true
                shift
                ;;
            --check)
                do_check=true
                shift
                ;;
            --uninstall)
                do_uninstall=true
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查环境
    if [ "$do_check" = true ]; then
        check_environment
        exit $?
    fi
    
    # 如果没有任何 agent 选项，默认安装 Kimi
    if [ "$install_kimi" = false ] && [ "$install_claude" = false ] && \
       [ "$install_cursor" = false ] && [ "$install_codex" = false ] && \
       [ "$install_all" = false ] && [ -z "$install_path" ] && [ "$install_type" != "system" ]; then
        install_kimi=true
    fi
    
    # 处理 --all-agents
    if [ "$install_all" = true ]; then
        install_kimi=true
        install_claude=true
        install_cursor=true
        install_codex=true
    fi
    
    # 显示安装信息
    echo "========================================"
    echo "  NCU CUDA Profiling Skill 安装程序"
    echo "========================================"
    echo ""
    
    # 检查环境
    if ! check_environment; then
        print_warning "环境检查未通过，是否继续安装? [y/N]"
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo ""
    
    # 系统级安装需要 sudo
    if [ "$install_type" = "system" ]; then
        if [ "$EUID" -ne 0 ]; then
            print_error "系统级安装需要 root 权限"
            print_info "请使用: sudo $0 --system"
            exit 1
        fi
        install_skill "$DEFAULT_SYSTEM_PATH"
        exit 0
    fi
    
    # 自定义路径安装
    if [ -n "$install_path" ]; then
        install_skill "$install_path"
        exit 0
    fi
    
    # 安装到各 AI Agent
    if [ "$install_kimi" = true ]; then
        install_kimi
        echo ""
    fi
    
    if [ "$install_claude" = true ]; then
        install_claude
        echo ""
    fi
    
    if [ "$install_cursor" = true ]; then
        install_cursor
        echo ""
    fi
    
    if [ "$install_codex" = true ]; then
        install_codex
        echo ""
    fi
    
    echo "========================================"
    print_success "全部安装完成！"
    echo "========================================"
    echo ""
    echo "📚 详细兼容性说明: AGENTS_COMPATIBILITY.md"
}

# 运行主函数
main "$@"
