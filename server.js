// server.js — FINAL PATCHED VERSION
import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(cors());

/* -------------------------------------------------------
   LOAD ENVIRONMENT KEYS
--------------------------------------------------------*/
const GROQ_KEY = process.env.GROQ_KEY;
const OR_KEY = process.env.OR_KEY;
const GEMINI_KEY = process.env.GEMINI_KEY;
const DEEP_KEY = process.env.DEEP_KEY;

/* -------------------------------------------------------
   MODEL DEFINITIONS
--------------------------------------------------------*/

const MODELS = [
  {
    label: "Groq • Llama-3.1 8B",
    id: "llama-3.1-8b-instant",
    provider: "groq"
  },
  {
    label: "Groq • Llama-3.3 70B",
    id: "llama-3.3-70b-versatile",
    provider: "groq"
  },
  {
    label: "GPT-OSS-20B",
    id: "openai/gpt-oss-20b",
    provider: "openrouter"
  },
  {
    label: "LLaMA 3.1 8B Instruct",
    id: "meta-llama/llama-3.1-8b-instruct",
    provider: "openrouter"
  },
  {
    label: "Kimi-K2",
    id: "moonshotai/kimi-k2-instruct-0905",
    provider: "openrouter"
  },
  {
    label: "DeepSeek V3",
    id: "deepseek/deepseek-v3",
    provider: "openrouter"
  },
  {
    label: "Gemini Flash 2.0",
    id: "models/gemini-2.0-flash-exp",
    provider: "gemini"
  },
  {
    label: "DeepSeek Chat",
    id: "deepseek-chat",
    provider: "deepseek"
  },
  {
    label: "DeepSeek Reasoner",
    id: "deepseek-reasoner",
    provider: "deepseek"
  }
];

/* -------------------------------------------------------
   CALL HELPERS
--------------------------------------------------------*/

async function callGroq(modelId, userPrompt) {
  try {
    const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: userPrompt }],
        max_tokens: 600,
        temperature: 0.6
      })
    });

    if (!r.ok) return null;
    const j = await r.json();
    return j.choices?.[0]?.message?.content || null;
  } catch {
    return null;
  }
}

async function callOpenRouter(modelId, userPrompt) {
  try {
    const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OR_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: userPrompt }],
        max_tokens: 600,
        temperature: 0.6
      })
    });

    if (!r.ok) return null;
    const j = await r.json();
    return j.choices?.[0]?.message?.content || null;
  } catch {
    return null;
  }
}

async function callGemini(modelId, userPrompt) {
  try {
    const r = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/${modelId}:generateContent?key=${GEMINI_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: userPrompt }] }]
        })
      }
    );

    const j = await r.json();
    if (j.error?.code === 429) return null;
    return j.candidates?.[0]?.content?.parts?.[0]?.text || null;
  } catch {
    return null;
  }
}

async function callDeepseek(modelId, userPrompt) {
  try {
    const r = await fetch("https://api.deepseek.com/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${DEEP_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: userPrompt }],
        max_tokens: 600,
        temperature: 0.6
      })
    });

    if (!r.ok) return null;
    const j = await r.json();
    return j.choices?.[0]?.message?.content || null;
  } catch {
    return null;
  }
}

/* -------------------------------------------------------
   ROUTE: /council
--------------------------------------------------------*/

app.post("/council", async (req, res) => {
  const prompt = req.body.prompt || "";
  const attemptedModels = MODELS.map(m => m.label);

  const answers = [];
  const transcripts = [];
  const weights = [];

  /* CALL ALL MODELS SEQUENTIALLY WITH THROTTLING */
  for (const m of MODELS) {
    let out = null;

    if (m.provider === "groq") {
      out = await callGroq(m.id, prompt);
    } else if (m.provider === "openrouter") {
      out = await callOpenRouter(m.id, prompt);
    } else if (m.provider === "gemini") {
      out = await callGemini(m.id, prompt);
    } else if (m.provider === "deepseek") {
      out = await callDeepseek(m.id, prompt);
    }

    if (out) {
      answers.push({ label: m.label, text: out });
      transcripts.push({ model: m.label, answer: out });

      // naive confidence estimation (longer = more confident)
      const conf = Math.min(1, out.length / 800);
      weights.push(conf);
    }

    await new Promise(res => setTimeout(res, 1200));
  }

  /* IF NOTHING WORKED, RETURN FALLBACK */
  if (answers.length === 0) {
    return res.json({
      chairman: "Groq • Llama-3.3 70B",
      modelsAttempted: attemptedModels,
      modelsUsed: [],
      bestIndex: 1,
      bestModel: "Groq • Llama-3.3 70B",
      ranking: [1],
      rankingReason: "No models responded",
      percentages: [100],
      avgConfidences: [1],
      final: "No answer could be generated.",
      transcripts: []
    });
  }

  /* WEIGHTED VOTING */
  const total = weights.reduce((a, b) => a + b, 0);
  const percentages = weights.map(w => Math.round((w / total) * 100));

  let bestIndex = percentages.indexOf(Math.max(...percentages));
  let finalAnswer = answers[bestIndex].text;

  /* RETURN NORMAL FORMAT */
  return res.json({
    chairman: "Groq • Llama-3.3 70B",
    modelsAttempted: attemptedModels,
    modelsUsed: answers.map(a => a.label),
    bestIndex,
    bestModel: answers[bestIndex].label,
    ranking: answers.map((a, i) => i + 1),
    rankingReason: "Confidence-weighted voting",
    percentages,
    avgConfidences: weights,
    final: finalAnswer,
    transcripts
  });
});

/* -------------------------------------------------------
   START SERVER
--------------------------------------------------------*/

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Council backend running on " + PORT));
