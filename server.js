// server.js - LLM Council backend (Groq + OpenRouter + Gemini + DeepSeek)
// package.json must include:  "type": "module"

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

if (!GROQ_KEY)   console.warn("WARNING: GROQ_KEY not set.");
if (!OR_KEY)     console.warn("WARNING: OR_KEY not set.");
if (!GEMINI_KEY) console.warn("WARNING: GEMINI_KEY not set.");

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const OR_URL   = "https://openrouter.ai/api/v1/chat/completions";

// Council model set
const MODELS = [
  // Groq
  { id: "llama-3.1-8b-instant",             provider: "groq",       label: "Groq • Llama-3.1 8B" },
  { id: "llama-3.3-70b-versatile",          provider: "groq",       label: "Groq • Llama-3.3 70B" }, // chairman

  // OpenRouter (OSS + Llama + Kimi + DeepSeek)
  { id: "openai/gpt-oss-20b",               provider: "openrouter", label: "OR • GPT-OSS-20B" },
  { id: "meta-llama/llama-3.1-8b-instruct", provider: "openrouter", label: "OR • LLaMA-3.1 8B Instruct" },
  { id: "moonshotai/kimi-k2-instruct-0905", provider: "openrouter", label: "OR • Kimi-K2 Instruct 0905" },
  { id: "deepseek/deepseek-v3",             provider: "openrouter", label: "OR • DeepSeek V3" },

  // Extra DeepSeek via OR (free tier)
  { id: "deepseek/deepseek-chat",           provider: "openrouter", label: "OR • DeepSeek Chat" },
  { id: "deepseek/deepseek-r1",             provider: "openrouter", label: "OR • DeepSeek R1" },
  { id: "deepseek/deepseek-chat-v3.1",      provider: "openrouter", label: "OR • DeepSeek Chat v3.1" },

  // Gemini
  { id: "gemini-2.0-flash-exp",             provider: "gemini",     label: "Gemini • Flash 2.0 Exp" }
];

const CHAIRMAN = MODELS[1];

const BASE_DELAY = {
  groq: 1000,
  openrouter: 2000,
  gemini: 2000
};

function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

async function callModelWithRetry(model, prompt, {
  temperature = 0.7,
  maxTokens = 450,
  maxAttempts = 3
} = {}) {
  let attempt = 1;
  let delay = BASE_DELAY[model.provider] || 2000;

  while (attempt <= maxAttempts) {
    try {
      let url, headers, body;

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
      } else {
        return null;
      }

      const resp = await fetch(url, { method: "POST", headers, body });

      if (!resp.ok) {
        const status = resp.status;
        const txt = await resp.text().catch(() => "");
        console.error(`Model ${model.label} attempt ${attempt} failed: ${status} ${txt}`);

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

      await sleep(500);
      return content;
    } catch (err) {
      console.error(`Model ${model.label} network error attempt ${attempt}:`, err?.message || err);
      if (attempt >= maxAttempts) return null;
      await sleep(delay);
      delay = Math.min(delay * 2, 20000);
      attempt += 1;
    }
  }
  return null;
}

// Health
app.get("/", (_req, res) => {
  res.send("LLM Council backend is running.");
});

app.post("/council", async (req, res) => {
  const prompt = req.body?.prompt;
  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ error: "Missing prompt" });
  }

  try {
    const attemptedModels = MODELS.map(m => m.label);
    const answers = [];

    // Small warmup so models are less likely to collide on limits
    await sleep(800);

    // 1) Collect answers sequentially with provider-aware stagger
    for (const m of MODELS) {
      const answer = await callModelWithRetry(
        m,
        `User question:\n${prompt}\n\nRespond clearly in markdown.`,
        { temperature: 0.7, maxTokens: 400 }
      );
      if (answer) {
        answers.push({
          label: m.label,
          provider: m.provider,
          text: answer
        });
      }
      const extra = BASE_DELAY[m.provider] || 1500;
      await sleep(extra);
    }

    if (!answers.length) {
      return res.status(500).json({ error: "No models returned an answer." });
    }

    const n = answers.length;

    // 2) Chairman JSON ranking + confidence
    let rankingIndices = [];
    let confByRank = [];
    let rankingReason = "No reason provided.";
    let bestIndex = 1;

    const rankingPrompt =
      `You are judging answers from ${n} models to the same user question.\n` +
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
        const start = trimmed.indexOf("{");
        const end = trimmed.lastIndexOf("}");
        const jsonText = start >= 0 && end > start ? trimmed.slice(start, end + 1) : trimmed;
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

    if (!rankingIndices.length || rankingIndices.length !== n) {
      rankingIndices = Array.from({ length: n }, (_, i) => i + 1);
    }
    if (!confByRank.length || confByRank.length !== n) {
      confByRank = Array.from({ length: n }, () => 1);
    }

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
      const rankPos = rankingIndices.indexOf(answerNumber);
      if (rankPos === -1) continue;

      const basePoints = n - rankPos;
      const conf = confByRank[rankPos] ?? 1;
      const clamped = Math.max(0, Math.min(1, conf));
      const weighted = basePoints * clamped;

      pointsPerAnswer[answerIdx] = weighted;
      avgConfPerAnswer[answerIdx] = clamped;
    }

    const maxPoints = pointsPerAnswer.reduce((m, v) => (v > m ? v : m), 0);
    const percentages = pointsPerAnswer.map(v => {
      if (maxPoints <= 0) return 0;
      return Math.round((v / maxPoints) * 100);
    });

    if (rankingIndices.length > 0) {
      bestIndex = rankingIndices[0];
    } else {
      bestIndex = pointsPerAnswer.reduce(
        (idx, v, i) => (v > pointsPerAnswer[idx] ? i : idx),
        0
      ) + 1;
    }

    const bestAnswer = answers[bestIndex - 1] || answers[0];

    // 4) Chairman synthesis
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
    return res.status(500).json({ error: "Council error: " + (err?.message || String(err)) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("LLM Council backend listening on port", PORT);
});
