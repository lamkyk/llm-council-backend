import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(cors());
app.use(express.json());

const GROQ_KEY = process.env.GROQ_KEY;
const OR_KEY = process.env.OR_KEY;

app.post("/council", async (req, res) => {
  try {
    const prompt = req.body.prompt;

    // Groq call
    const groqRes = await fetch(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${GROQ_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "llama-3.3-70b-versatile",
          messages: [{ role: "user", content: prompt }],
          temperature: 0.7,
          max_tokens: 400
        })
      }
    );
    const groqData = await groqRes.json();

    // OR call (second model)
    const orRes = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OR_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lamkyk.github.io/llm-council-frontend/",
        "X-Title": "LLM Council"
      },
      body: JSON.stringify({
        model: "mistralai/mistral-nemo",
        messages: [{ role: "user", content: prompt }]
      })
    });
    const orData = await orRes.json();

    res.json({
      fromGroq: groqData.choices?.[0]?.message?.content || "",
      fromOR: orData.choices?.[0]?.message?.content || ""
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/", (req, res) => {
  res.send("LLM Council backend running");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log("Backend running on", PORT));
