# 构建新分支
git switch --orphan new_branch # 构建新分支

# 安装 Sphinx 核心和最常用的 Read the Docs 主题
pip install sphinx sphinx-rtd-theme 
# 验证安装
sphinx-build --version

# 创建第一个Sphinx项目
mkdir my_docs && cd my_docs

# 直接 sphinx-quickstart 如果有问题的话
# 代替 sphinx-quickstart
python -m sphinx.cmd.quickstart

# 代替 sphinx-build
python -m sphinx.cmd.build source build/html


# 使用虚拟环境
# 1. 创建项目文件夹
mkdir my_docs && cd my_docs
# 2. 创建虚拟环境
C:\Users\40603\AppData\Local\Python\bin\python.exe -m venv venv

# 如果不生效
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
<!-- RemoteSigned：允许运行本地创建的脚本（包括虚拟环境激活脚本），但从互联网下载的脚本需要数字签名才能运行
-Scope CurrentUser：只对当前用户生效，不需要管理员权限，也不会影响系统其他用户 -->
# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 永远使用python -m pip，可以 100% 确保安装到当前虚拟环境：
python -m pip install --upgrade pip
python -m pip install sphinx sphinx-rtd-theme sphinx-autobuild


# 初始化项目（100%可靠的模块方式）
python -m sphinx.cmd.quickstart


# 实时预览（修改自动刷新浏览器）
python -m sphinx_autobuild source build/html

# 清理构建文件(POWERSHELL 命令 )
Remove-Item build -Recurse -Force 




