<div align="center">
  
  <h1>miniclaw: Research & Study of Personal AI AGENT </h1>
  

📕 **miniclaw** is an **Research & Study** personal AI agent inspired by [nanobot](https://github.com/HKUDS/nanobot).
</div>

## 📢 News

- **2026-04-27** 🎉 实现Skills技能扩展能力：渐进式加载技能/技能缓存/热加载监控/skill并行加载.
- **2026-04-21** 🚀 实现核心能力：ReAct Loop/分层记忆管理（存储、压缩、Dream）/工具集调用（权限管控、文件管理工具、命令行工具）/HTTP请求响应.

</details>

> miniclaw is for educational, research, and technical exchange purposes only.

## Key Features of miniclaw:

🔬 **Implement Core**: Implement the core capabilities of an AI Agent.


## Table of Contents

- [📢 News](#-news)
- [Key Features of miniclaw:](#key-features-of-miniclaw)
- [Table of Contents](#table-of-contents)
- [📦 Install](#-install)
- [🚀 Quick Start](#-quick-start)
- [💬 Http Request Format](#-http-request-format)


## 📦 Install

**Install from source** (latest features, experimental changes may land here first; recommended for development)

```bash
git clone https://github.com/wones/miniclaw.git
cd miniclaw
pip install -e .
```

## 🚀 Quick Start

> [!TIP]
> Set your api key,base url and modelin `./.env`.


**1. Configure** (`./.env`)
>Create the configuration file if it does not exist.

*Set your api key,base url and model* :
```json
PROVIDERS__OPENAI__APIKEY="XXX"
PROVIDERS__OPENAI__BASEURL="XXX"
PROVIDERS__OPENAI__MODEL="XXX"
```

**2. Run**

```bash
python -m miniclaw
```

That's it! You have a working AI agent in 2 minutes.

## 💬 Http Request Format

curl -X POST http://localhost:8765/webhook/msg \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，我的名字是张三", "session_key": "user123", "callback_url": "http://localhost:8765/webhook/test"}'.


