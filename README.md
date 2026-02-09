# kb_chat

一个轻量的「可检索知识库（BM25）+ DeepSeek 对话 UI（Streamlit）」原型。

## 给合作者试用（每个人用自己的 PDF/目录）

如果你希望合作者：
- 自己管理/存放 PDF（用自己的目录，不用你的硬盘路径）
- 你更新代码后他能快速拿到最新版

推荐做法：**让他在自己电脑上跑一份 kb_chat**（本地目录自然就是他自己的），再用 `git pull` 或同步工具更新代码。

### 快速启动（Windows）

在 PowerShell 运行：
- `./run.ps1`

默认只允许本机访问（更安全）：`127.0.0.1:8501`  
如果你想同一局域网其它设备也能访问，把环境变量设成：
- `$env:KB_STREAMLIT_ADDR="0.0.0.0"`

端口也可改：
- `$env:KB_STREAMLIT_PORT="8501"`

### 目录设置（每个人独立）

在页面左侧「设置」里选择：
- `PDF 路径`：你的 PDF 根目录
- `DB 路径 / MD 路径`：你的知识库目录

这些偏好会写在本机的 `user_prefs.json`（已在 `.gitignore` 中忽略，不会影响拉取更新）。

### 如何拿到最新版

最省事的是把这个目录做成 git 仓库（GitHub/Gitee/私有服务器均可）：
1) 你 push 更新
2) 合作者每次运行 `run.ps1` 会自动 `git pull`（如果该目录是 git repo）

如果你不想用 git，也可以用 OneDrive/网盘/共享盘同步整个文件夹，但要注意同步冲突。

## 1) 安装依赖

如果你没有管理员权限，建议用 `--user` 安装到当前用户目录：

```powershell
cd F:\research-papers\2026\Jan\else\kb_chat
python -m pip install --user -r requirements.txt
```

## 2) 配置 DeepSeek

建议只用环境变量，不要把 key 写进代码文件里。

PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="你的key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

CMD：

```bat
set DEEPSEEK_API_KEY=你的key
set DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
set DEEPSEEK_MODEL=deepseek-chat
```

> 可选：如果你把转换脚本（默认会自动找上级目录的 `test2.py`）挪到了别处，可以设置：
>
> ```powershell
> $env:KB_PDF_CONVERTER="D:\\path\\to\\test2.py"
> ```

## 3) 建库（把 Markdown 喂进去）

建议把“最终版 md”喂进去，默认会跳过 `temp/` 这类页级临时文件，避免重复。

```powershell
python ingest.py --src F:\research-papers\2026\Jan\else\tosave_md_strict --db .\db --incremental --prune
```

## 4) 启动 UI

```powershell
python -m streamlit run app.py
```
