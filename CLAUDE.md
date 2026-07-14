# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **personal AI/ML learning notes repository**. It is not a software project — it's a knowledge base of markdown notes covering deep learning, LLM/RAG/LangChain, NLP, quantitative finance, and quantitative trading, organized by topic from multiple online courses.

## Git Workflow

- `图灵/` (raw PDFs, extracted .txt subtitles) is **gitignored** — only committed notes live in `图灵学习笔记/`
- Commit convention: `docs(notes): <description>` with footer `Co-Authored-By: Claude <noreply@anthropic.com>`

## Directory Structure

```
notes/
├── 图灵/                          # 🔒 Raw course materials (gitignored)
│   ├── RAG应用开发/                # 12 PDFs — RAG → LangChain → RAG进阶 → 项目实战 (柏汌)
│   ├── Agent应用开发/              # 3 PDFs — Agent智能体 1/2/3 (初见)
│   ├── 大模型面试/                 # 23 PDFs — Transformer/PEFT/RLHF/量化 (面试向)
│   └── 大模型私有化微调/            # 9 PDFs — 微调101 → LoRA → 量化 → 多模态 → 最佳实践 → 项目 (陈钢)
│
├── 图灵学习笔记/                   # 🎯 Organized notes from all 图灵 courses (tracked in git)
│   ├── 大模型RAG应用开发（第一期）/  # 10 chapters — RAG基础 → LangChain → Advanced RAG → 项目
│   ├── Agent应用开发/              # 3 notes — Agent 1/2/3 (智能体开发)
│   ├── 大模型面试/                 # 23 notes — Transformer各组件 + PEFT/RLHF/量化
│   └── 大模型私有化微调/            # 6 notes (so far) — 微调101 → PEFT → 量化+多模态
│
├── NLP学习笔记/                   # NLP fundamentals — RNN, Attention, 中英翻译案例 (尚硅谷)
├── PyTorch学习笔记/               # PyTorch from scratch — 张量 → Autograd → 损失函数 → 优化器
│
├── 21位游资悟道心法/              # Stock market / trading psychology (21 traders)
│
├── 深度学习笔记_PyTorch完整版.md    # Standalone deep learning reference (PyTorch)
├── 深度学习_矩阵求导笔记.md
├── 深度学习笔记_反向传播与计算图.md
└── 期权入门学习笔记（三讲合集）.md
```

## Content Architecture

### 图灵学习笔记 series (primary curriculum)

Three parallel course tracks, all from 图灵教育:

**1. 大模型RAG应用开发（第一期）** (柏汌) — 10 chapters, complete:
- RAG基础 → LangChain框架 (×3) → RAG进阶 → Advanced RAG技术实现/实战 → 上下文压缩 → 评估优化 → 项目实战

**2. Agent应用开发** (初见) — 3 chapters:
- Agent智能体 1/2/3 — Function Calling, Tool Use, Agent 设计模式

**3. 大模型面试** (面试向) — 23 Q&A notes:
- Transformer系列 (整体/词表/掩码/多头注意力/位置编码/Encoder/Decoder/前馈网络)
- PEFT/LoRA/RLHF系列 (Reward Hacking, 缩放定律, 剪枝/蒸馏, 正则化, 稀疏微调, 幻觉, 数据扩增)
- 量化系列 (量化和二值化)

**4. 大模型私有化微调** (陈钢) — 6 organized notes (9 lectures total):
- 微调101 → PEFT/LoRA → 量化 → 多模态(CNN) → 多模态(Transformer)
- Raw subtitles fully cleaned for lectures 1-9, organized notes ready for 1-6

### Note writing style (Feynman technique)
- Explain concepts in plain language first, then formalize
- Python code examples for every practical concept
- Comparison tables for similar concepts (PTQ vs QAT, LoRA vs Adapter, etc.)
- ASCII diagrams for workflows and model architectures
- `费曼说：...` plain-language summaries at section ends
- Every `.md` in 图灵学习笔记/ is human-edited — raw PDF subtitle extracts live in `图灵/` as `.txt`

### Python packages commonly referenced
- `chromadb`, `langchain`, `langchain-openai`, `langchain-community`
- `dashscope` (阿里千问 API), `PyPDF2`, `pymupdf` (fitz)
- `sentence-transformers`, `faiss-cpu`, `unsloth`, `bitsandbytes`
- `numpy`, `pandas`, `torch`, `transformers`, `fvcore`

## Common Workflows

### Extract text from course PDFs
```bash
# Using pymupdf (preferred — handles Chinese text well)
python -c "
import fitz
doc = fitz.open('图灵/<course>/<filename>.pdf')
text = ''.join(doc[i].get_text() + '\n' for i in range(doc.page_count))
doc.close()
with open('图灵/<course>/<filename>.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f'{len(text)} chars')
"

# Alternative: pdftotext (better layout preservation for subtitle PDFs)
pdftotext -layout "图灵/<course>/<filename>.pdf" "图灵/<course>/<filename>.txt"
```

### Clean subtitle text into notes
1. Extract raw text from PDF → `.txt` (in `图灵/`)
2. Remove: song lyrics, speaker labels, timestamps, verbal repetitions, classroom management chatter
3. Restructure into markdown with headings, tables, code blocks
4. Save as `.md` under `图灵学习笔记/<course>/`

## Environment
- OS: Windows 11
- Shell: Git Bash (POSIX syntax, not cmd/PowerShell)
- Python 3.14
- PDF tools: `pymupdf` (fitz), `pdftotext` (poppler-utils)
- Note editor: Obsidian (.obsidian config present)
- IDE: PyCharm (.idea config present)
