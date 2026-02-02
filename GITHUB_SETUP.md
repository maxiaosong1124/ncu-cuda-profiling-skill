# GitHub CLI 安装和配置指南

## 1. 安装 GitHub CLI

### Ubuntu/Debian
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

### 其他系统
访问: https://github.com/cli/cli#installation

## 2. 登录 GitHub

```bash
# 浏览器登录（推荐）
gh auth login

# 按提示选择:
# ? What account do you want to log into? GitHub.com
# ? What is your preferred protocol for Git operations? HTTPS
# ? Authenticate Git with your GitHub credentials? Yes
# ? How would you like to authenticate GitHub CLI? Login with a web browser
```

## 3. 运行发布脚本

```bash
cd /home/maxiaosong/work_space/cuda_learning/cuda_code/ncu-cuda-profiling-skill
./publish.sh hellofss
```

