# Pi_zaya / kb_chat

这份 README 是我（开发者）写给使用者的。目标是让你在自己的电脑上，稳定地把 PDF 转成 Markdown、更新知识库，并在对话里可追溯引用。

## 你可以用它做什么

- 在「文献管理」页批量上传 PDF，并转换为 Markdown（含图片、公式、参考文献处理）。
- 一键「更新知识库」，把 Markdown 分块索引到本地 DB。
- 在「对话」页基于知识库检索回答，并显示可点击的文内引用 `[n]`。
- 点击引用可弹出文献信息（来源、题录、DOI 链接）。
- 可将引用加入右侧「文献篮」，集中查看并跳转 DOI 页面。
- 参考文献索引支持后台 Crossref 同步，不阻塞页面使用。

---

## 当前默认模型（我现在使用）

本项目当前优先使用 **Qwen**（OpenAI 兼容接口）：

- 默认 `QWEN_BASE_URL`: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 默认 `QWEN_MODEL`: `qwen3-vl-plus`

代码逻辑是：

1. 优先读 `QWEN_API_KEY`
2. 否则回退 `DEEPSEEK_API_KEY`
3. 再回退 `OPENAI_API_KEY`

所以你只要设置 `QWEN_API_KEY`，就会走 Qwen。

---

## 快速启动（Windows）

### 1) 克隆项目

```powershell
git clone https://github.com/LittlePyx/Pi_zaya.git
cd Pi_zaya
```

### 2) 设置 Qwen API Key（当前推荐）

```powershell
$env:QWEN_API_KEY="你的key"
```

可选（一般不用改）：

```powershell
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL="qwen3-vl-plus"
```

### 3) 启动

```powershell
.\run.ps1
```

或双击 `run.cmd`。

脚本会自动：

- 创建 `.venv`（若不存在）
- 安装 `requirements.txt`
- 启动 Streamlit（默认 `http://127.0.0.1:8501`）

---

## 标准使用流程

1. 打开「文献管理」页。
2. 设置：
   - `文献目录（PDF）`
   - `输出目录（Markdown）`
3. 上传 PDF（支持批量）。
4. 选择转换模式：
   - `normal`：质量优先（截图识别 + VL）
   - `ultra_fast`：更快，质量略降
   - `no_llm`：不使用多模态模型（基础提取）
5. 点击「更新知识库」。
6. 回到「对话」页提问。

---

## 引用与文献篮

在回答里看到 `[n]` 后：

- 点击 `[n]` 会弹出引用详情（支持拖动）。
- 可直接打开 DOI。
- 可「加入文献篮」。
- 文献篮会在右侧汇总，支持定位与高亮条目。

说明：不是每条参考文献都一定有 DOI（历史文献、会议条目、源数据缺失时常见）。

---

## 常用环境变量

### 模型相关

- `QWEN_API_KEY`
- `QWEN_BASE_URL`
- `QWEN_MODEL`
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`（回退）
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`（回退）

### 路径与服务

- `KB_PDF_DIR`：默认 PDF 根目录
- `KB_MD_DIR`：默认 Markdown 目录
- `KB_DB_DIR`：知识库索引目录
- `KB_CHAT_DB`：对话数据库路径
- `KB_LIBRARY_DB`：文献库数据库路径
- `KB_STREAMLIT_ADDR`：默认 `127.0.0.1`
- `KB_STREAMLIT_PORT`：默认 `8501`

### 参考文献索引

- `KB_CROSSREF_BUDGET_S`：Crossref 后台同步预算秒数（默认 45）

---

## 常见问题

### 1) 页面一直 Running

通常是前端脚本缓存或后台任务异常。建议：

1. 重启 Streamlit
2. 浏览器 `Ctrl+F5` 强刷
3. 再看「文献管理」页中的后台状态

### 2) 为什么有的引用没有 DOI

可能原因：

- 文献本身未注册 DOI
- 参考文献条目不完整或噪声大
- Crossref 未命中

### 3) 更新知识库后对话没变化

请确认你更新的是当前使用的 `DB` 目录，并且 Markdown 文件确实已生成到对应目录。

---

## 给使用者的说明

- 这个项目目前是我持续迭代中的版本，界面和细节会更新。
- 你遇到“可复现”的问题时，附上：PDF 名称、页面截图、生成的 `.md` 片段，我可以更快定位并修复。
