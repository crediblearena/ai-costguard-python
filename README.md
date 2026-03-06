# AI Cost Guard — Python SDK

> Track, analyze, and optimize your AI/LLM API costs with a single line of code. Supports **50+ models** from OpenAI, Anthropic, Google Gemini, Cohere, Mistral, and more.

[![PyPI version](https://img.shields.io/pypi/v/ai-cost-guard-sdk?color=blue)](https://pypi.org/project/ai-cost-guard-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/ai-cost-guard-sdk)](https://pypi.org/project/ai-cost-guard-sdk/)
[![Downloads](https://img.shields.io/pypi/dm/ai-cost-guard-sdk)](https://pypi.org/project/ai-cost-guard-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why AI Cost Guard?

| Problem | Solution |
|---------|----------|
| 💸 Surprise AI bills at month end | Real-time cost tracking with budget alerts |
| 🔍 Can't tell which features cost most | Per-feature & per-user cost breakdowns |
| 📊 No visibility into token usage | Automatic token counting & analytics |
| 🤝 Works only with one provider | Universal support for OpenAI, Anthropic, Gemini, Cohere, Mistral |

## Installation

```bash
pip install ai-cost-guard-sdk
```

## Quick Start

```python
from ai_cost_guard import AICostGuard

guard = AICostGuard(api_key="acg_live_your_key_here")

# Track any AI API call
guard.track(
    provider="openai",
    model="gpt-4",
    input_tokens=500,
    output_tokens=150,
    latency_ms=1200,
    feature="chat",
)

# Provider-specific helpers
import openai
response = openai.chat.completions.create(model="gpt-4", messages=[...])
guard.track_openai(model=response.model, usage=response.usage)

# Don't forget to flush on shutdown
guard.shutdown()
```

## Provider Helpers

### OpenAI
```python
guard.track_openai(model="gpt-4", usage=response.usage, feature="chat")
```

### Anthropic
```python
guard.track_anthropic(model="claude-3-opus", usage=response.usage, feature="analysis")
```

### Google Gemini
```python
guard.track_gemini(model="gemini-pro", usage=response.usage_metadata, feature="search")
```

### Cohere
```python
guard.track_cohere(model="command-r-plus", usage=response.meta.billed_units, feature="rag")
```

## Configuration

```python
guard = AICostGuard(
    api_key="acg_live_...",
    api_url="https://api.aicostguard.com/api/v1",  # Custom API URL
    debug=True,                    # Enable debug logging
    batch_events=True,             # Batch events (default: True)
    batch_interval_ms=5000,        # Batch flush interval (default: 5000ms)
    max_batch_size=50,             # Max events per batch (default: 50)
    max_retries=3,                 # Max retry attempts (default: 3)
    max_events_per_second=100,     # Rate limit (default: 100)
    default_feature="my-app",      # Default feature tag
)
```

## Context Manager Support

```python
from ai_cost_guard import AICostGuard

# Automatically flushes and shuts down on exit
with AICostGuard(api_key="acg_live_...") as guard:
    guard.track(provider="openai", model="gpt-4o", input_tokens=500, output_tokens=100)
```

## Async Support

```python
import asyncio
from ai_cost_guard import AsyncAICostGuard

async def main():
    guard = AsyncAICostGuard(api_key="acg_live_...")

    await guard.track(
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=200,
        feature="async-chatbot",
    )

    await guard.shutdown()

asyncio.run(main())
```

## FastAPI Integration

```python
from fastapi import FastAPI
from ai_cost_guard import AICostGuard

app = FastAPI()
guard = AICostGuard(api_key="acg_live_...")

@app.post("/chat")
async def chat(prompt: str):
    import openai
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    guard.track_openai(model=response.model, usage=response.usage, feature="chat-api")
    return {"reply": response.choices[0].message.content}

@app.on_event("shutdown")
def shutdown():
    guard.shutdown()
```

## Django Integration

```python
# settings.py
from ai_cost_guard import AICostGuard
AI_GUARD = AICostGuard(api_key="acg_live_...")

# views.py
from django.conf import settings

def chat_view(request):
    response = openai.chat.completions.create(model="gpt-4o", messages=[...])
    settings.AI_GUARD.track_openai(model=response.model, usage=response.usage, feature="django-chat")
    return JsonResponse({"reply": response.choices[0].message.content})
```

## Supported Models (50+)

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4o, GPT-4o mini, GPT-4 Turbo, GPT-4, GPT-3.5, o3, o3-mini, o1, o1-mini |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, Claude 3 Sonnet |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro |
| **Cohere** | Command R+, Command R, Command, Embed |
| **Mistral** | Large, Medium, Small, Tiny |

## Ecosystem

- 📦 **[TypeScript SDK](https://www.npmjs.com/package/@crediblearena/ai-costguard)** — `npm install @crediblearena/ai-costguard`
- 🖥️ **[CLI Tool](https://www.npmjs.com/package/ai-cost-cli)** — `npx ai-cost-cli`
- 🧩 **[VS Code Extension](https://marketplace.visualstudio.com/)** — IDE integration
- 🌐 **[Chrome Extension](https://chromewebstore.google.com/)** — OpenAI cost overlay
- 📊 **[Dashboard](https://aicostguard.com)** — Full web analytics

## Links

- 🌐 [Website](https://aicostguard.com)
- 📖 [Documentation](https://aicostguard.com/docs/python-sdk)
- 💰 [Cost Calculator](https://aicostguard.com/ai-cost-calculator)
- 🐛 [Issues](https://github.com/crediblearena/ai-cost-guard/issues)

## License

MIT
