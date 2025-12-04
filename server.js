// server.js - 12-model free-tier LLM Council (Groq + OpenRouter)
// ESM, for "type": "module" in package.json

import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(cors());
app.use(express.json());

// Environment variables from Render
const GROQ_KEY = process.env.GROQ_KEY;
const OR_KEY   = process.env.OR_KEY;

if (!GROQ_KEY) {
  console.warn("WARNING: GROQ_KEY not set in environment.");
}
if (!OR_KEY) {
  console.warn("WARNING: OR_KEY not set in environment.");
}

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const OR_URL   = "https://openrouter.ai/api/v1/chat/completions";

// 12 model council: 3 from Groq, 9 from OpenRouter (free tier)
const MODELS = [
  // Groq
  { id: "llama-3.3-70b-versatile", provider: "groq", label: "Groq • Llama-3.3 70B" },
  { id: "llama-3.1-8b-instant",    provider: "groq", label: "Groq • Llama-3.1 8B" },
  { id: "llama-3.2-1b-preview",    provider: "groq", label: "Groq • Llama-3.2 1B" },

  // OpenRouter (free models)
  { id: "mistralai/mistral-nemo",                provider: "openrouter", label: "OR • Mistral Nemo" },
  { id: "google/gemma-2-9b-it",                  provider: "openrouter", label: "OR • Gemma 2 9B" },
  { id: "google/gemma-2-2b-it",                  provider: "openrouter", label: "OR • Gemma 2 2B" },
  { id: "qwen/qwen-2-7b-instruct",               provider: "openrouter", label: "OR • Qwen 2 7B Instruct" },
  { id: "nousresearch/nous-hermes-2-mistral-7b", provider: "openrouter", label: "OR • Hermes 2 Mistral 7B" },
  { id: "openchat/openchat-7b",                  provider: "openrouter", label: "OR • OpenChat 7B" },
  { id: "undi95/toppy-m-7b",                     provider: "openrouter", label: "OR • Toppy M 7B" },
  { id: "neversleep/noromaid-7b",                provider: "openrouter", label: "OR • Noromaid 7B" },
  { id: "sophosympatheia/muenn-7b",              provider: "openrouter", label: "OR • Muenn 7B" }
];

// Chairman model used for ranking and final synthesis
const CHAIRMAN = MODELS[0]; // Groq 70B

// Per model base delay for backoff (milliseconds)
const BASE_DELAY = {
  groq: 900,
  openrouter: 2000
};

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Call one model with retry and backoff
async function callModelWithRetry(model, prompt, {
  temperature = 0.7,
  maxTokens = 400,
  maxAttempts = 3
} = {}) {
  let attempt = 1;
  let delay = BASE_DELAY[model.provider] || 1500;

  while (attempt <= maxAttempts) {
    try {
      let url;
      let headers;
      let body;

      if (model.provider === "groq") {
        url = GROQ_URL;
        headers = {
          Authorization: `Bearer ${GROQ_KEY}`,
          "Content-Type": "application/json"
        };
        body = JSON.stringify({
          model: model.id,
          messages: [{ role: "user", content: prompt }],
          temperature,
          max_tokens: maxTokens
        });
      } else if (model.provider === "openrouter") {
        url = OR_URL;
        headers = {
          Authorization: `Bearer ${OR_KEY}`,
          "Content-Type": "application/json"
        };
        body = JSON.stringify({
          model: model.id,
          messages: [{ role: "user", content: prompt }],
          temperature,
          max_tokens: maxTokens
        });
      } else {
        return null;
      }

      const resp = await fetch(url, { method: "POST", headers, body });

      if (!resp.ok) {
        const status = resp.status;
        const text = await resp.text().catch(() => "");
        console.error(`Model ${model.label} attempt ${attempt} failed: ${status} ${text}`);

        // Rate limit or transient errors: retry with backoff
        if (status === 429 || status === 500 || status === 502 || status === 503 || status === 504) {
          if (attempt < maxAttempts) {
            await sleep(delay);
            delay = Math.min(delay * 2, 20000);
            attempt += 1;
            continue;
          }
        }
        // Non retryable or exhausted
        return null;
      }

      const data = await resp.json();
      const content = data.choices?.[0]?.message?.content || null;
      return content;
    } catch (err) {
      console.error(`Model ${model.label} network error on attempt ${attempt}:`, err?.message || err);
      if (attempt >= maxAttempts) return null;
      await sleep(delay);
      delay = Math.min(delay * 2, 20000);
      attempt += 1;
    }
  }

  return null;
}

// Root health check
app.get("/", (_req, res) => {
  res.send("LLM Council backend running");
});

// Main council endpoint
app.post("/council", async (req, res) => {
  const prompt = req.body?.prompt;
  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ error: "Missing prompt in body." });
  }

  try {
    const answers = [];
    const attemptedModels = MODELS.map(m => m.label);

    // 1) Collect answers sequentially to ease rate limits
    for (const m of MODELS) {
      const answer = await callModelWithRetry(
        m,
        `User question:\n${prompt}\n\nRespond clearly and helpfully in markdown.`,
        { temperature: 0.7, maxTokens: 450 }
      );
      if (answer) {
        answers.push({
          label: m.label,
          provider: m.provider,
          text: answer
        });
      }
    }

    if (!answers.length) {
      return res.status(500).json({ error: "No models returned an answer." });
    }

    const n = answers.length;

    // 2) Ask chairman to rank available answers
    const rankingPrompt =
      `You are judging answers from ${n} models to the same user question.\n` +
      `User question:\n"${prompt}"\n\n` +
      `Here are the answers:\n\n` +
      answers
        .map((a, i) => `${i + 1}. (${a.label})\n${a.text}`)
        .join("\n\n") +
      `\n\nReply EXACTLY in this format:\n` +
      `RANKING: [a permutation of 1..${n}] | BEST REASON: [one short sentence]`;

    const rankingRaw = await callModelWithRetry(
      CHAIRMAN,
      rankingPrompt,
      { temperature: 0.0, maxTokens: 220 }
    );

    let ranking = [];
    let rankingReason = "No reason provided.";

    if (rankingRaw) {
      const rankMatch = rankingRaw.match(/RANKING[:\s]*\[(.*?)\]/i);
      const reasonMatch = rankingRaw.match(/BEST REASON[:\s]*(.+)/i);

      if (rankMatch) {
        ranking = rankMatch[1]
          .split(/[\s,]+/)
          .map(v => parseInt(v, 10))
          .filter(v => !Number.isNaN(v) && v >= 1 && v <= n);
      }
      if (reasonMatch) {
        rankingReason = reasonMatch[1].trim();
      }
    }

    if (!ranking.length) {
      ranking = answers.map((_, i) => i + 1);
    }

    // 3) Chairman synthesizes final answer
    const rankingSummary = ranking
      .map((pos, idx) => {
        const a = answers[pos - 1];
        return a ? `#${idx + 1}: Answer ${pos} (${a.label})` : null;
      })
      .filter(Boolean)
      .join(" | ");

    const finalPrompt =
      `Combine the following model answers into one best answer.\n\n` +
      `User question:\n${prompt}\n\n` +
      `Answers:\n` +
      answers.map((a, i) => `${i + 1}. (${a.label})\n${a.text}`).join("\n\n") +
      `\n\nRanking from best to worst:\n${rankingSummary || "Not available"}\n` +
      `Reason the top answer is best: ${rankingReason}\n\n` +
      `Now write ONE final answer in markdown for the user. Do not mention models, ranking, or that you are combining anything.`;

    const finalAnswer = await callModelWithRetry(
      CHAIRMAN,
      finalPrompt,
      { temperature: 0.65, maxTokens: 700 }
    );

    return res.json({
      chairman: CHAIRMAN.label,
      modelsAttempted: attemptedModels,
      modelsUsed: answers.map(a => a.label),
      answers,          // array of {label, provider, text}
      ranking,          // array of indices 1..n
      rankingReason,
      final: finalAnswer || "Final answer unavailable."
    });
  } catch (err) {
    console.error("Council error", err);
    return res.status(500).json({
      error: "Council error: " + (err?.message || String(err))
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("LLM Council backend listening on port", PORT);
});
