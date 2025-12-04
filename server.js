// server.js - 8-model council backend (Groq + OpenRouter, ESM)

import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(cors());
app.use(express.json());

// Environment keys (never exposed to frontend)
const GROQ_KEY = process.env.GROQ_KEY;
const OR_KEY   = process.env.OR_KEY;

if (!GROQ_KEY || !OR_KEY) {
  console.warn("WARNING: GROQ_KEY or OR_KEY missing in environment.");
}

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const OR_URL   = "https://openrouter.ai/api/v1/chat/completions";

// 8-model council
const MODELS = [
  // Groq
  { id: "llama-3.3-70b-versatile", provider: "groq", label: "Groq • Llama-3.3 70B" },
  { id: "llama-3.1-8b-instant",    provider: "groq", label: "Groq • Llama-3.1 8B" },
  { id: "gemma2-9b-it",            provider: "groq", label: "Groq • Gemma2 9B" },
  { id: "gemma2-27b-it",           provider: "groq", label: "Groq • Gemma2 27B" },

  // OpenRouter
  { id: "mistralai/mistral-nemo",  provider: "openrouter", label: "OR • Mistral Nemo" },
  { id: "deepseek/deepseek-chat",  provider: "openrouter", label: "OR • DeepSeek Chat" },
  { id: "qwen/qwen-2.5-72b-instruct", provider: "openrouter", label: "OR • Qwen 2.5 72B" },
  { id: "nousresearch/nous-capybara-34b", provider: "openrouter", label: "OR • Nous Capybara 34B" }
];

// Pick a strong model as chairman (used for ranking + final answer)
const CHAIRMAN = MODELS.find(m => m.id === "qwen/qwen-2.5-72b-instruct") || MODELS[0];

async function callModel(model, prompt, maxTokens = 500, temperature = 0.7) {
  try {
    if (model.provider === "groq") {
      const r = await fetch(GROQ_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${GROQ_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: model.id,
          messages: [{ role: "user", content: prompt }],
          max_tokens: maxTokens,
          temperature
        })
      });

      if (!r.ok) {
        console.error("Groq error", model.id, r.status, await r.text().catch(() => ""));
        return null;
      }

      const d = await r.json();
      return d.choices?.[0]?.message?.content || null;
    }

    if (model.provider === "openrouter") {
      const r = await fetch(OR_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${OR_KEY}`,
          "Content-Type": "application/json",
          "HTTP-Referer": "https://lamkyk.github.io/llm-council-frontend/",
          "X-Title": "LLM Council"
        },
        body: JSON.stringify({
          model: model.id,
          messages: [{ role: "user", content: prompt }],
          max_tokens: maxTokens,
          temperature
        })
      });

      if (!r.ok) {
        console.error("OpenRouter error", model.id, r.status, await r.text().catch(() => ""));
        return null;
      }

      const d = await r.json();
      return d.choices?.[0]?.message?.content || null;
    }

    return null;
  } catch (err) {
    console.error("callModel error", model.id, err.message || err);
    return null;
  }
}

// health check root
app.get("/", (req, res) => {
  res.send("LLM Council backend running");
});

// main council endpoint
app.post("/council", async (req, res) => {
  const prompt = req.body?.prompt;
  if (!prompt) {
    return res.status(400).json({ error: "Missing prompt in body" });
  }

  try {
    const answers = [];

    // Step 1: collect individual answers from all models (sequential for now)
    for (const model of MODELS) {
      const answer = await callModel(
        model,
        `User question:\n${prompt}\n\nAnswer as clearly and helpfully as you can.`
      );

      if (answer) {
        answers.push({
          modelId: model.id,
          label: model.label,
          text: answer
        });
      }
    }

    if (!answers.length) {
      return res.status(500).json({ error: "No models returned an answer." });
    }

    // Step 2: ask chairman to rank answers
    const n = answers.length;
    const rankingPrompt =
      `You are judging answers from ${n} different models to the same user question.\n` +
      `Question:\n"${prompt}"\n\n` +
      `Here are the answers:\n\n` +
      answers
        .map((a, i) => `${i + 1}. (${a.label})\n${a.text}`)
        .join("\n\n") +
      `\n\n` +
      `Reply EXACTLY in this format:\n` +
      `RANKING: [a permutation of 1..${n}] | BEST REASON: [one short sentence explaining why the #1 answer is best]`;

    const rankingRaw = await callModel(CHAIRMAN, rankingPrompt, 300, 0.0);

    let ranking = [];
    let rankingReason = "No reason provided.";
    if (rankingRaw) {
      const rankMatch = rankingRaw.match(/RANKING[:\s]*\[(.*?)\]/i);
      const reasonMatch = rankingRaw.match(/BEST REASON[:\s]*(.+)/i);
      if (rankMatch) {
        ranking = rankMatch[1]
          .split(/[\s,]+/)
          .map(v => parseInt(v, 10))
          .filter(nv => !Number.isNaN(nv) && nv >= 1 && nv <= n);
      }
      if (reasonMatch) {
        rankingReason = reasonMatch[1].trim();
      }
    }

    if (!ranking.length) {
      ranking = answers.map((_, i) => i + 1); // fallback: 1,2,3,...
    }

    // Step 3: chairman synthesizes final answer
    const rankingSummary = ranking
      .map((pos, idx) => `#${idx + 1}: Answer ${pos} (${answers[pos - 1].label})`)
      .join(" | ");

    const finalPrompt =
      `You are a meta-model combining multiple model answers into one best response.\n\n` +
      `User question:\n${prompt}\n\n` +
      `Answers:\n${answers
        .map((a, i) => `${i + 1}. (${a.label})\n${a.text}`)
        .join("\n\n")}\n\n` +
      `Ranking of answers (best to worst):\n${rankingSummary}\n` +
      `Reason the top-ranked answer is best: ${rankingReason}\n\n` +
      `Now write a single final answer that is as strong as possible.\n` +
      `Do NOT mention models, ranking, or that you are combining answers. Just answer the user.`;

    const finalAnswer = await callModel(CHAIRMAN, finalPrompt, 700, 0.6);

    res.json({
      chairman: CHAIRMAN.label,
      modelsUsed: answers.map(a => a.label),
      answers,
      ranking,          // array like [3,1,2,...]
      rankingReason,
      final: finalAnswer || "No final answer generated."
    });
  } catch (err) {
    console.error("council error", err);
    res.status(500).json({ error: "Council error: " + (err.message || String(err)) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("LLM Council backend running on port", PORT));
