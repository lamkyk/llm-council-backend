Credit to https://x.com/karpathy/status/1992381094667411768

# LLM Council (Secure Multi-Model Deliberation)

LLM Council is a lightweight system for generating higher-quality answers by using multiple LLMs instead of depending on just one. A single prompt is sent to a group of diverse models, each produces its own response, they anonymously review and rank each other, and a designated “chairman” synthesizes everything into one final answer.

The result is simple: **more reliable, more insightful output through structured disagreement and collective reasoning.**

---

## How It Works

### 1. Parallel Opinions
Each model receives the user prompt independently and generates an initial answer.

### 2. Cross-Model Review
Every model is given anonymized responses from the others and asked to rank them based on:
- accuracy  
- clarity  
- depth  
- confidence  

A weighted voting system produces the final model order.

### 3. Final Synthesis
A chairman model uses the ranked responses and reasoning chains to create a single, polished answer.

---

## Features

- **Backend-only API key usage** (no browser exposure)  
- **Multi-provider support**: Groq, OpenRouter, DeepSeek, Gemini, Mistral, and more  
- **Confidence-weighted scoring and ranking**  
- **Collapsible transcripts** for each model’s answer and reasoning  
- **Ranking panel** that reveals how the council voted  
- **Clean, responsive UI** (fully compatible with Google Sites)  
- **Health check endpoint** that verifies model availability  
- **Built with simple HTML/JS + Node/Express** — no frameworks required  

---

## Why This Exists

No single model is best for every question.  
LLM Council acts as an **advisory board of models**, combining different strengths, catching blind spots, and improving consistency—especially on harder reasoning problems.

---

## Deployment

The project consists of:

### Backend
A Node/Express service that:
- distributes the prompt to all models  
- collects and evaluates responses  
- computes rankings and confidence scores  
- generates the final chairman answer  

### Frontend
A static HTML/JS client that:
- works on any host  
- safe to embed (API keys stay on the server)  
- minimal, fast, and easy to customize  

Runs smoothly on Render, Vercel, Netlify, or any Node-compatible environment.
