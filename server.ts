import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Health Endpoint
  app.get("/api/health", (req, res) => {
    const keySet = !!(process.env.GROQ_API_KEY);
    res.json({
      status: "ok",
      keySet: keySet,
      message: "✓ Groq LLaMA 3.3 Connected & Secured"
    });
  });

  // API Ask RAG Endpoint
  app.post("/api/ask", async (req, res) => {
    try {
      const apiKey = process.env.GROQ_API_KEY ;
      if (!apiKey) {
        return res.status(500).json({
          error: "Groq API key is not configured. Please add GROQ_API_KEY."
        });
      }

      const { question, chunks } = req.body;
      if (!question || !chunks || chunks.length === 0) {
        return res.status(400).json({ error: "Missing question or context chunks." });
      }

      // Build context from chunks
      const context = chunks
        .map((c: any, i: number) => `[Chunk ${i + 1}]:\n${c.text || c}`)
        .join("\n\n");

      const systemPrompt = `You are a precise knowledge assistant. Answer questions strictly based on the provided context chunks.
Rules:
- Only use information from the provided context.
- If the answer is not in the context, say: "This information is not available in the provided document."
- Be concise, clear, and informative.
- Never make up or invent information.`;

      const userPrompt = `Context from document:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;

      const groqRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: "llama-3.3-70b-versatile",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt }
          ],
          max_tokens: 600,
          temperature: 0.2
        })
      });

      if (!groqRes.ok) {
        const errData = await groqRes.json().catch(() => ({}));
        const msg = errData?.error?.message || `Groq API returned HTTP ${groqRes.status}`;
        return res.status(502).json({ error: msg });
      }

      const data = await groqRes.json() as any;
      const answer = data.choices?.[0]?.message?.content?.trim() || "No response received from Groq LLaMA.";

      res.json({
        answer: answer,
        usage: data.usage || {
          prompt_tokens: undefined,
          completion_tokens: undefined
        },
        model: "llama-3.3-70b-versatile"
      });
    } catch (error: any) {
      console.error("Backend error:", error);
      res.status(500).json({ error: error.message || "Internal Server Error" });
    }
  });

  // Vite Middleware Setup for development, static assets in production
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();