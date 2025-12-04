// server.js - Groq + OpenRouter + Gemini + DeepSeek council with confidence-weighted voting
// Requires: "type": "module" in package.json

import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(cors());
app.use(express.json());

// Environment variables
const GROQ_KEY   = process.env.GROQ_KEY;
const OR_KEY     = process.env.OR_KEY;
const GEMINI_KEY = process.env.GEMINI_KEY;
const DEEP_KEY   = process.env.DEEP_KEY;

if (!GROQ_KEY)   console.warn("WARNING: GROQ_KEY not set.");
if (!OR_KEY)     console.warn("WARNING: OR_KEY not set.");
if (!GEMINI_KEY) console.warn("WARNING: GEMINI_KEY not set.");
if (!DEEP_KEY)   console.warn("WARNING: DEEP_KEY not set.");

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const OR_URL   = "https://openrouter.ai/api/v1/chat/completions";
const DEEP_URL = "https://api.deepseek.com/chat/completions";

// Model list (as requested)
const MODELS = [
  // Groq
  { id: "llama-3.1-8b-instant",            provider: "groq",     label: "Groq • Llama-3.1 8B" },
  { id: "llama-3.3-70b-versatile",         provider: "groq",     label: "Groq • Llama-3.3 70B" }, // chairman

  // OpenRouter
  { id: "openai/gpt-oss-20b",              provider: "openrouter", label: "OR • GPT-OSS-20B" },
  { id: "meta-llama/llama-3.1-8b-instruct", provider: "openrouter", label: "OR • LLaMA-3.1 8B Instruct" },
  { id: "moonshotai/kimi-k2-instruct-0905", provider: "openrouter", label: "OR • Kimi-K2 Instruct 0905" },
  { id: "deepseek/deepseek-v3",            provider: "openrouter", label: "OR • DeepSeek V3" },

  // Gemini
  { id: "gemini-2.0-flash-exp",            provider: "gemini",  label: "Gemini • Flash 2.0 Exp" },

  // DeepSeek direct
  { id: "deepseek-chat",                   provider: "deepseek", label: "DeepSeek • Chat" },
  { id: "deepseek-reasoner",               provider: "deepseek", label: "DeepSeek • Reasoner" }
];

// Chairman is Groq 70B
const CHAIRMAN = MODELS[1];

// Per-provider base delays (ms)
const BASE_DELAY = {
  groq: 1000,
  openrouter: 2500,
  gemini: 2000,
  deepseek: 2500
};

function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

// Generic call with retries and per-provider handling
async function callModelWithRetry(model, prompt, {
  temperature = 0.7,
  maxTokens = 400,
  maxAttempts = 3
} = {}) {
  let attempt = 1;
  let delay = BASE_DELAY[model.provider] || 2000;

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
      } else if (model.provider === "gemini") {
        url = `https://generativelanguage.googleapis.com/v1beta/models/${model.id}:generateContent?key=${GEMINI_KEY}`;
        headers = { "Content-Type": "application/json" };
        body = JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [{ text: prompt }]
            }
          ],
          generationConfig: {
            temperature,
            maxOutputTokens: maxTokens
          }
        });
      } else if (model.provider === "deepseek") {
        url = DEEP_URL;
        headers = {
          Authorization: `Bearer ${DEEP_KEY}`,
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

        if ([429, 500, 502, 503, 504].includes(status) && attempt < maxAttempts) {
          await sleep(delay);
          delay = Math.min(delay * 2, 20000);
          attempt += 1;
          continue;
        }
        return null;
      }

      const data = await resp.json();
      let content = null;

      if (model.provider === "gemini") {
        const parts = data?.candidates?.[0]?.content?.parts || [];
        content = parts.map(p => p.text || "").join("\n").trim() || null;
      } else {
        content = data?.choices?.[0]?.message?.content || null;
      }

      if (!content) return null;

      await sleep(800);
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

// Health check
app.get("/", (_req, res) => {
  res.send("LLM Council backend is running.");
});

app.post("/council", async (req, res) => {
  const prompt = req.body?.prompt;
  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ error: "Missing prompt in body." });
  }

  try {
    const attemptedModels = MODELS.map(m => m.label);
    const answers = [];

    // Global warmup
    await sleep(1500);

    // 1) Collect answers sequentially with per-provider stagger
    for (const m of MODELS) {
      const answer = await callModelWithRetry(
        m,
        `User question:\n${prompt}\n\nRespond clearly in markdown.`,
        { temperature: 0.7, maxTokens: 450 }
      );
      if (answer) {
        answers.push({
          label: m.label,
          provider: m.provider,
          text: answer
        });
      }
      const extraDelay = BASE_DELAY[m.provider] || 1500;
      await sleep(extraDelay);
    }

    if (!answers.length) {
      return res.status(500).json({ error: "No models returned an answer." });
    }

    const n = answers.length;

    // 2) Ask chairman for JSON ranking + confidence
    let rankingIndices = [];
    let confByRank = [];
    let rankingReason = "No reason parsed.";
    let bestIndex = 1;

    const rankingPrompt =
      `You are judging answers from ${n} models to the same question.\n` +
      `Question:\n"${prompt}"\n\n` +
      `Answers:\n` +
      answers
        .map((a, i) => `${i + 1}. (${a.label})\n${a.text}`)
        .join("\n\n") +
      `\n\nReturn STRICT JSON ONLY in this format (no prose, no markdown fences):\n` +
      `{\n` +
      `  "ranking": [answer_numbers_best_to_worst],\n` +
      `  "confidence": [parallel_confidences_between_0_and_1],\n` +
      `  "reason": "one short sentence explaining why the top answer is best"\n` +
      `}`;

    const rankingRaw = await callModelWithRetry(
      CHAIRMAN,
      rankingPrompt,
      { temperature: 0.0, maxTokens: 260 }
    );

    if (rankingRaw) {
      try {
        const trimmed = rankingRaw.trim();
        const jsonStart = trimmed.indexOf("{");
        const jsonEnd = trimmed.lastIndexOf("}");
        const jsonText = jsonStart >= 0 && jsonEnd > jsonStart
          ? trimmed.slice(jsonStart, jsonEnd + 1)
          : trimmed;

        const parsed = JSON.parse(jsonText);

        if (Array.isArray(parsed.ranking)) {
          rankingIndices = parsed.ranking.map(v => parseInt(v, 10));
        }
        if (Array.isArray(parsed.confidence)) {
          confByRank = parsed.confidence.map(v => {
            const x = Number(v);
            if (Number.isNaN(x)) return 1;
            return Math.max(0, Math.min(1, x));
          });
        }
        if (typeof parsed.reason === "string" && parsed.reason.trim()) {
          rankingReason = parsed.reason.trim();
        }
      } catch (err) {
        console.error("Failed to parse ranking JSON:", err);
      }
    }

    // Fallback ranking / confidence
    if (!rankingIndices.length || rankingIndices.length !== n) {
      rankingIndices = Array.from({ length: n }, (_, i) => i + 1);
    }
    if (!confByRank.length || confByRank.length !== n) {
      confByRank = Array.from({ length: n }, () => 1);
    }

    // Validate ranking is permutation of 1..n
    const expectedSet = new Set(Array.from({ length: n }, (_, i) => i + 1));
    const gotSet = new Set(rankingIndices);
    if (expectedSet.size !== gotSet.size || [...expectedSet].some(v => !gotSet.has(v))) {
      rankingIndices = Array.from({ length: n }, (_, i) => i + 1);
    }

    // 3) Confidence-weighted scoring
    const pointsPerAnswer = new Array(n).fill(0);
    const avgConfPerAnswer = new Array(n).fill(0);

    for (let answerIdx = 0; answerIdx < n; answerIdx++) {
      const answerNumber = answerIdx + 1;
      const rankPos = rankingIndices.indexOf(answerNumber); // 0-based
      if (rankPos === -1) continue;

      const basePoints = n - rankPos;
      const conf = confByRank[rankPos] ?? 1;
      const clampedConf = Math.max(0, Math.min(1, conf));
      const weighted = basePoints * clampedConf;

      pointsPerAnswer[answerIdx] = weighted;
      avgConfPerAnswer[answerIdx] = clampedConf;
    }

    const maxPoints = pointsPerAnswer.reduce((m, v) => (v > m ? v : m), 0);
    const percentages = pointsPerAnswer.map(v => {
      if (maxPoints <= 0) return 0;
      return Math.round((v / maxPoints) * 100);
    });

    // Best answer index
    if (rankingIndices.length > 0) {
      bestIndex = rankingIndices[0];
    } else {
      const maxIdx = pointsPerAnswer.reduce(
        (idx, v, i) => (v > pointsPerAnswer[idx] ? i : idx),
        0
      );
      bestIndex = maxIdx + 1;
    }

    const bestAnswer = answers[bestIndex - 1] || answers[0];

    // 4) Chairman synthesizes final answer
    const finalPrompt =
      `User question:\n${prompt}\n\n` +
      `Here are answers from different models. The best answer is number ${bestIndex} from ${bestAnswer.label}.\n\n` +
      answers
        .map((a, i) => `${i + 1}. (${a.label})\n${a.text}`)
        .join("\n\n") +
      `\n\nWrite ONE final answer for the user in markdown. Focus on the best answer and improve it. Do not mention models, ranking, or that multiple answers were combined.`;

    const finalAnswer = await callModelWithRetry(
      CHAIRMAN,
      finalPrompt,
      { temperature: 0.6, maxTokens: 700 }
    );

    // Transcripts for frontend collapsible display
    const transcripts = answers.map(a => ({
      model: a.label,
      answer: a.text
    }));

    return res.json({
      chairman: CHAIRMAN.label,
      modelsAttempted: attemptedModels,
      modelsUsed: answers.map(a => a.label),
      bestIndex,
      bestModel: bestAnswer.label,
      ranking: rankingIndices,
      rankingReason,
      percentages,
      avgConfidences: avgConfPerAnswer,
      final: finalAnswer || bestAnswer.text || "Final answer unavailable.",
      transcripts
    });
  } catch (err) {
    console.error("Council error:", err);
    return res.status(500).json({
      error: "Council error: " + (err?.message || String(err))
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("LLM Council backend listening on port", PORT);
});
