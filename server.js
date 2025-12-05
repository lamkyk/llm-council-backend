// server.js — Council backend with batched parallel execution (safe, reliable)
import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(cors());

/* -------------------------------------------------------
   LOAD ENVIRONMENT KEYS
   (Render dashboard: GROQ_KEY, OR_KEY; GEMINI_KEY/DEEP_KEY optional)
--------------------------------------------------------*/
const GROQ_KEY = process.env.GROQ_KEY;
const OR_KEY = process.env.OR_KEY;
const GEMINI_KEY = process.env.GEMINI_KEY;  // not used if all via OpenRouter
const DEEP_KEY = process.env.DEEP_KEY;      // not used if all via OpenRouter

/* -------------------------------------------------------
   MODEL REGISTRY
   - Groq: 2 models
   - OpenRouter: free/subsidized stack
--------------------------------------------------------*/
const MODELS = [
  // Groq
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

  // OpenRouter — confirmed free/subsidized
  {
    label: "GPT-OSS-20B",
    id: "openai/gpt-oss-20b",
    provider: "openrouter"
  },
  {
    label: "LLaMA 3.1 8B Instruct",
    id: "meta-llama/llama-3.1-8b-instruct:free",
    provider: "openrouter"
  },
  {
    label: "Kimi-K2",
    id: "moonshotai/kimi-k2-instruct-0905:free",
    provider: "openrouter"
  },
  {
    label: "DeepSeek V3",
    id: "deepseek/deepseek-v3:free",
    provider: "openrouter"
  },
  {
    label: "DeepSeek Chat (OR)",
    id: "deepseek/deepseek-chat:free",
    provider: "openrouter"
  },
  {
    label: "DeepSeek R1 Reasoner (OR)",
    id: "deepseek/deepseek-r1:free",
    provider: "openrouter"
  },
  {
    label: "DeepSeek Chat v3.1 (OR)",
    id: "deepseek/deepseek-chat-v3.1:free",
    provider: "openrouter"
  },
  {
    label: "Gemini Flash 2.0",
    id: "google/gemini-2.0-flash-exp:free",
    provider: "openrouter"
  },
  {
    label: "Mistral Nemo",
    id: "mistralai/mistral-nemo:free",
    provider: "openrouter"
  }
];

// Chairman is Groq Llama-3.3 70B
const CHAIRMAN_ID = "llama-3.3-70b-versatile";
const CHAIRMAN_LABEL = "Groq • Llama-3.3 70B";

/* -------------------------------------------------------
   LOW-LEVEL CALL HELPERS
--------------------------------------------------------*/

async function callGroq(model, prompt, maxTokens = 500) {
  if (!GROQ_KEY) return null;
  try {
    const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens
      })
    });
    if (!r.ok) return null;
    const j = await r.json();
    return j.choices?.[0]?.message?.content || null;
  } catch {
    return null;
  }
}

async function callOpenRouter(model, prompt, maxTokens = 500) {
  if (!OR_KEY) return null;
  try {
    const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OR_KEY}`,
        "HTTP-Referer": "https://llm-council",
        "X-Title": "LLM Council",
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens
      })
    });
    if (!r.ok) return null;
    const j = await r.json();
    return j.choices?.[0]?.message?.content || null;
  } catch {
    return null;
  }
}

/* Generic wrapper: pick correct provider, add small per-call delay */
async function callModel(entry, prompt, maxTokens = 500) {
  let out = null;

  if (entry.provider === "groq") {
    out = await callGroq(entry.id, prompt, maxTokens);
  } else if (entry.provider === "openrouter") {
    out = await callOpenRouter(entry.id, prompt, maxTokens);
  } else {
    out = null;
  }

  // short pause to smooth bursts (used inside batch loop)
  await new Promise(res => setTimeout(res, 200));
  return out;
}

/* -------------------------------------------------------
   BATCHED PARALLEL EXECUTION
   - max concurrency to keep under rate limits
--------------------------------------------------------*/
async function runInBatches(items, batchSize, fn) {
  const results = new Array(items.length);
  let index = 0;

  while (index < items.length) {
    const batch = [];
    const batchIndices = [];
    for (let i = 0; i < batchSize && index < items.length; i++, index++) {
      batch.push(items[index]);
      batchIndices.push(index);
    }

    await Promise.all(
      batch.map((item, localIdx) =>
        fn(item, batchIndices[localIdx])
          .then(res => {
            results[batchIndices[localIdx]] = res;
          })
          .catch(() => {
            results[batchIndices[localIdx]] = null;
          })
      )
    );

    // small gap between batches so we never spike too hard
    await new Promise(res => setTimeout(res, 1200));
  }

  return results;
}

/* -------------------------------------------------------
   MAIN COUNCIL ENDPOINT
--------------------------------------------------------*/
app.post("/council", async (req, res) => {
  const prompt = (req.body?.prompt || "").trim();
  if (!prompt) {
    return res.json({ error: "Missing prompt" });
  }

  const modelsAttempted = MODELS.map(m => m.label);

  /* -------------------------
     PHASE 1: FIRST OPINIONS
  --------------------------*/
  try {
    const firstResults = await runInBatches(
      MODELS,
      3, // concurrency
      async (m) => {
        return await callModel(m, prompt, 500);
      }
    );

    const answers = [];
    firstResults.forEach((ans, idx) => {
      if (ans && typeof ans === "string" && ans.trim().length > 0) {
        answers.push({
          model: MODELS[idx].label,
          answer: ans.trim()
        });
      }
    });

    if (!answers.length) {
      return res.json({ error: "No models returned answers." });
    }

    /* -------------------------
       PHASE 2: REVIEW & RANK
       Confidence-weighted voting
    --------------------------*/
    const N = answers.length;
    const anonList = answers
      .map((a, i) => `A${i + 1}: ${a.answer.slice(0, 550)}`)
      .join("\n\n");

    const reviewPrompts = answers.map(() => {
      return (
        `You are an expert judge on an LLM council.\n\n` +
        `You will see multiple candidate answers A1..A${N} to the same question.\n` +
        `Rank them from best to worst in terms of accuracy, depth, clarity, and usefulness.\n` +
        `Also estimate your confidence that your ranking is correct (0.0–1.0).\n\n` +
        `Return ONLY valid JSON in this exact schema:\n` +
        `{"ranking":[1,2,3,...], "confidence":0.0}\n\n` +
        `Answers:\n${anonList}`
      );
    });

    const reviewResults = await runInBatches(
      answers,
      2, // review concurrency
      async (_ans, idx) => {
        const reviewerLabel = answers[idx].model;
        const modelEntry = MODELS.find(m => m.label === reviewerLabel) || MODELS[0];
        const raw = await callModel(modelEntry, reviewPrompts[idx], 220);
        if (!raw) return null;
        try {
          const parsed = JSON.parse(raw);
          if (!Array.isArray(parsed.ranking)) return null;
          const conf =
            typeof parsed.confidence === "number"
              ? Math.max(0, Math.min(1, parsed.confidence))
              : 0.5;
          return {
            ranking: parsed.ranking.map(x => Number(x)).filter(n => !Number.isNaN(n)),
            confidence: conf
          };
        } catch {
          return null;
        }
      }
    );

    // Confidence-weighted tallies
    const scores = new Array(N).fill(0);
    const confSum = new Array(N).fill(0);
    const confCount = new Array(N).fill(0);

    reviewResults.forEach((rev) => {
      if (!rev || !Array.isArray(rev.ranking) || !rev.ranking.length) return;
      const conf = rev.confidence ?? 0.5;
      const L = rev.ranking.length;

      rev.ranking.forEach((pos, idx) => {
        const answerIdx = pos - 1;
        if (answerIdx < 0 || answerIdx >= N) return;
        const rankWeight = L - idx; // higher for better rank
        const score = conf * rankWeight;
        scores[answerIdx] += score;
        confSum[answerIdx] += conf;
        confCount[answerIdx] += 1;
      });
    });

    let totalScore = scores.reduce((a, b) => a + b, 0);

    // Fallback if no valid reviews: uniform scores
    if (!totalScore || !Number.isFinite(totalScore)) {
      for (let i = 0; i < N; i++) {
        scores[i] = 1;
        confSum[i] = 0.5;
        confCount[i] = 1;
      }
      totalScore = scores.reduce((a, b) => a + b, 0);
    }

    const percentages = scores.map((s) =>
      totalScore ? Math.round((s / totalScore) * 100) : 0
    );
    const avgConfidences = confSum.map((c, i) =>
      confCount[i] ? c / confCount[i] : 0
    );

    // Ranking as 1-based indices sorted by score desc
    const ranking = [...Array(N).keys()]
      .sort((a, b) => scores[b] - scores[a])
      .map(i => i + 1);

    /* -------------------------
       PHASE 3: CHAIRMAN ANSWER
    --------------------------*/
    const chairmanPrompt =
      `You are the Chairman of the LLM Council.\n\n` +
      `User question:\n${prompt}\n\n` +
      `Here are the candidate answers from the council models:\n\n` +
      answers
        .map(
          (a, i) =>
            `A${i + 1} (${a.model}):\n${a.answer}\n`
        )
        .join("\n") +
      `\n\nHere is the confidence-weighted ranking (1 is best):\n${JSON.stringify(
        ranking
      )}\n\n` +
      `Use this information to write a single, high-quality final answer for the user. ` +
      `Do not mention the ranking process, just answer the question as well as possible.`;

    const finalAnswer =
      (await callGroq(CHAIRMAN_ID, chairmanPrompt, 700)) ||
      answers[ranking[0] - 1].answer ||
      answers[0].answer;

    const transcripts = answers.map(a => ({
      model: a.model,
      answer: a.answer
    }));

    return res.json({
      final: finalAnswer,
      chairman: CHAIRMAN_LABEL,
      ranking,
      rankingReason: "Confidence-weighted voting",
      percentages,
      avgConfidences,
      modelsUsed: answers.map(a => a.model),
      modelsAttempted,
      transcripts
    });
  } catch (e) {
    console.error("Council error:", e);
    return res.json({
      error: "Council failure",
      final: null
    });
  }
});

/* -------------------------------------------------------
   HEALTH CHECK (Upgraded for OpenRouter + Groq compatibility)
--------------------------------------------------------*/

app.get("/health", async (req, res) => {
  const checks = [];

  async function testGroq(model, name) {
    try {
      const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.GROQ_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model,
          max_tokens: 20,
          temperature: 0,
          messages: [
            { role: "system", content: "Health check. Reply with: pong" },
            { role: "user", content: "pong" }
          ]
        })
      });

      const j = await r.json();
      const reply = j?.choices?.[0]?.message?.content || null;

      checks.push({
        model: name,
        id: model,
        provider: "groq",
        ok: reply?.toLowerCase().includes("pong") || false,
        reply
      });
    } catch (err) {
      checks.push({
        model: name,
        id: model,
        provider: "groq",
        ok: false,
        reply: String(err)
      });
    }
  }

  async function testOR(model, name) {
    try {
      const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.OR_KEY}`,
          "HTTP-Referer": "https://your-site.com",  // optional but recommended
          "X-Title": "LLM Council Health Check",
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model,
          max_tokens: 20,
          temperature: 0,
          messages: [
            { role: "system", content: "Health probe. Respond exactly with: pong" },
            { role: "user", content: "pong" }
          ]
        })
      });

      const j = await r.json();
      const reply = j?.choices?.[0]?.message?.content || null;

      checks.push({
        model: name,
        id: model,
        provider: "openrouter",
        ok: reply?.toLowerCase().includes("pong") || false,
        reply
      });
    } catch (err) {
      checks.push({
        model: name,
        id: model,
        provider: "openrouter",
        ok: false,
        reply: String(err)
      });
    }
  }

  /* -------------------------
     MODELS TO HEALTH CHECK
  --------------------------*/

  const GROQ_MODELS = [
    ["llama-3.1-8b-instant", "Groq • Llama-3.1 8B"],
    ["llama-3.3-70b-versatile", "Groq • Llama-3.3 70B"]
  ];

  const OR_MODELS = [
    ["openai/gpt-oss-20b", "GPT-OSS-20B"],
    ["meta-llama/llama-3.1-8b-instruct:free", "LLaMA 3.1 8B Instruct"],
    ["moonshotai/kimi-k2-instruct-0905:free", "Kimi-K2"],
    ["deepseek/deepseek-v3:free", "DeepSeek V3"],
    ["deepseek/deepseek-chat:free", "DeepSeek Chat"],
    ["deepseek/deepseek-r1:free", "DeepSeek R1 Reasoner"],
    ["deepseek/deepseek-chat-v3.1:free", "DeepSeek Chat v3.1"],
    ["google/gemini-2.0-flash-exp:free", "Gemini Flash 2.0"],
    ["mistralai/mistral-nemo:free", "Mistral Nemo"]
  ];

  /* -------------------------
     RUN CHECKS IN PARALLEL
  --------------------------*/

  await Promise.all([
    ...GROQ_MODELS.map(([id, name]) => testGroq(id, name)),
    ...OR_MODELS.map(([id, name]) => testOR(id, name))
  ]);

  /* -------------------------
     SEND RESULT
  --------------------------*/

  res.json({
    status: "ok",
    modelsTested: checks.length,
    timestamp: new Date().toISOString(),
    checks
  });
});


/* -------------------------------------------------------
   START SERVER
--------------------------------------------------------*/
const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log("Council backend running on " + PORT)
);
