import 'dotenv/config';
import express from 'express';
import type { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const MODEL_NAME = 'gemini-4-hydrogen';

const SYSTEM_INSTRUCTION =
  'Kamu adalah SolvexAI, asisten AI modern yang cerdas, akurat, ramah, dan membantu. Berikan jawaban terstruktur dan jelas menggunakan Bahasa Indonesia kecuali pengguna meminta bahasa lain. Gunakan format markdown sederhana seperti **tebal** dan blok kode (```) jika relevan.';

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('Server Gemini API key belum terkonfigurasi.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

interface IncomingPart {
  text?: string;
  inlineData?: {
    mimeType: string;
    data: string;
  };
  inline_data?: {
    mime_type: string;
    data: string;
  };
}

interface IncomingContent {
  role: 'user' | 'model';
  parts: IncomingPart[];
}

function normalizeContents(contents: IncomingContent[]) {
  return contents.map((turn) => ({
    role: turn.role === 'model' ? 'model' : 'user',
    parts: (Array.isArray(turn.parts) ? turn.parts : []).map((part) => {
      if (part.inlineData?.data && part.inlineData?.mimeType) {
        return {
          inlineData: {
            mimeType: part.inlineData.mimeType,
            data: part.inlineData.data,
          },
        };
      }
      if (part.inline_data?.data && part.inline_data?.mime_type) {
        return {
          inlineData: {
            mimeType: part.inline_data.mime_type,
            data: part.inline_data.data,
          },
        };
      }
      return { text: String(part.text ?? '') };
    }),
  }));
}

async function generateWithRetry(ai: GoogleGenAI, contents: ReturnType<typeof normalizeContents>, maxRetries = 2) {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await ai.models.generateContent({
        model: MODEL_NAME,
        contents,
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
        },
      });
    } catch (error: unknown) {
      lastError = error;
      const msg = error instanceof Error ? error.message : String(error);
      const isTransient = msg.includes('429') || msg.includes('503') || msg.includes('UNAVAILABLE');
      if (!isTransient || attempt === maxRetries) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 600 * Math.pow(2, attempt)));
    }
  }
  throw lastError;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '20mb' }));

  app.post('/api/chat', async (req: Request, res: Response) => {
    try {
      const body = req.body as { contents?: IncomingContent[] };
      if (!body || !Array.isArray(body.contents) || body.contents.length === 0) {
        res.status(400).json({ error: 'Riwayat percakapan tidak boleh kosong.' });
        return;
      }

      const ai = getGeminiClient();
      const normalized = normalizeContents(body.contents);
      const response = await generateWithRetry(ai, normalized);
      const replyText = (response.text ?? '').trim();

      if (!replyText) {
        res.status(502).json({ error: 'Model tidak mengembalikan teks balasan.' });
        return;
      }

      res.json({ reply: replyText });
    } catch (error: unknown) {
      console.error('SolvexAI /api/chat error:', error);
      const message = error instanceof Error ? error.message : 'Terjadi kesalahan pada server.';
      res.status(500).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`SolvexAI server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
