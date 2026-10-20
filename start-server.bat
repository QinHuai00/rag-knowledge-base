@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

rem 优先用项目内虚拟环境，其次是 PATH 里的 python（不依赖仓库外的路径）
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到可用的 Python。
    echo.
    echo 解决方法：
    echo   1. 安装 Python 3.10+ 并勾选 "Add python.exe to PATH"；
    echo   2. 或在本项目根目录创建虚拟环境：python -m venv .venv
    echo      然后安装依赖：.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PY%" -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo [错误] 缺少 HTTP 服务依赖（fastapi / uvicorn）。
    echo 解决方法：在项目根目录执行
    echo   "%PY%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo 启动 RAG 知识库 HTTP 服务：http://127.0.0.1:8000/docs
echo 按 Ctrl+C 停止。
"%PY%" -m app.server
pause
