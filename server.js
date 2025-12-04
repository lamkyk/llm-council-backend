// server.js
import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(cors());
app.use(express.json());

// keys come from environment, not from code
const GROQ_KEY = process.env.GROQ_KEY;
const OR_KEY   = process.env.OR_KEY;

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const OR_URL   = "https://openrouter.ai/api/v1/chat/completions";

const MODELS = [
  { id: "llama-3.1-8b-instant",       provider: "groq" },
  { id: "llama-3.3-70b-versatile",    provider: "groq" },
  { id: "deepseek/deepseek-chat",     provider: "or" },
  { id: "google/gemini-1.5-pro",      provider: "or" },
  { id: "anthropic/claude-3.5-sonnet",provider: "or" },
  { id: "openai/gpt-4.1-mini",        provider: "or" }
];

async function callModel(model, prompt) {
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
          max_tokens: 400,
          temperature: 0.7
        })
      });
      const d = await r.json();
      return d.choices?.[0]?.message?.content;
    }

    if (model.provider === "or") {
      const r = await fetch(OR_URL, {
        method: "POST",
        headers: {
          "X-API-Key": OR_KEY,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: model.id,
          messages: [{ role: "user", content: prompt }],
          max_tokens: 400,
          temperature: 0.7
        })
      });
      const d = await r.json();
      return d.choices?.[0]?.message?.content;
    }
  } catch (e) {
    return null;
  }
}

app.post("/council", async (req, res) => {
  const prompt = req.body.prompt;
  if (!prompt) return res.json({ error: "Missing prompt" });

  const answers = [];

  // collect from all models
  for (const m of MODELS) {
    const out = await callModel(m, prompt);
    if (out) answers.push({ model: m.id, text: out });
  }

  if (!answers.length) {
    return res.json({ error: "No models responded." });
  }

  // pick strongest-like model
  function pickStrongModel() {
    const order = ["claude", "gpt", "gemini", "70b", "deepseek", "8b"];
    for (const key of order) {
      const found = answers.find(a => a.model.toLowerCase().includes(key));
      if (found) return found;
    }
    return answers[0];
  }

  const strongest = pickStrongModel(); // {model,text}

  const synthPrompt =
    `Combine the following answers into one unified final answer:\n\n` +
    answers.map((a, i) => `${i + 1}. (${a.model})\n${a.text}`).join("\n\n") +
    `\n\nDo NOT mention models or providers. Just give the best final answer.`;

  const finalModel = MODELS.find(m => m.id === strongest.model);
  const finalAnswer = await callModel(finalModel, synthPrompt);

  res.json({
    used: answers.length,
    final: finalAnswer,
    answers
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Backend running on port " + PORT));
