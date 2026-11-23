import { MORRIS_SYSTEM_PROMPT } from "@/lib/persona";

export const maxDuration = 1200;

type Msg = { role: "user" | "assistant"; content: string };

const MAX_TURNS = 16; // messages sent to the model, newest kept
const MAX_CHARS = 4000; // per message, defensive cap

function sanitise(body: unknown): Msg[] | null {
  if (!body || typeof body !== "object" || !Array.isArray((body as { messages?: unknown }).messages)) return null;
  const out: Msg[] = [];
  for (const m of (body as { messages: unknown[] }).messages) {
    if (!m || typeof m !== "object") return null;
    const { role, content } = m as { role?: unknown; content?: unknown };
    if ((role !== "user" && role !== "assistant") || typeof content !== "string" || !content.trim()) return null;
    out.push({ role, content: content.slice(0, MAX_CHARS) });
  }
  return out.slice(-MAX_TURNS);
}

/** Proxies one chat turn to the Modal vLLM server (OpenAI-compatible). */
export async function POST(req: Request) {
  const base = process.env.MORRIS_ENDPOINT;
  const model = process.env.MORRIS_MODEL;
  if (!base || !model) {
    return Response.json({ error: "Server is not configured" }, { status: 500 });
  }
  const messages = sanitise(await req.json().catch(() => null));
  if (!messages || messages.length === 0) {
    return Response.json({ error: "Bad request" }, { status: 400 });
  }
  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "system", content: MORRIS_SYSTEM_PROMPT }, ...messages],
        max_tokens: 1000,
        temperature: 0.9,
      }),
      signal: AbortSignal.timeout(1_200_000),
    });
    if (!res.ok) {
      // 503 means the server scaled down again: the client should re-wake it.
      return Response.json({ error: "model_unavailable" }, { status: res.status === 503 ? 503 : 502 });
    }
    const data = await res.json();
    const content: unknown = data?.choices?.[0]?.message?.content;
    if (typeof content !== "string" || !content.trim()) {
      return Response.json({ error: "Bad upstream reply" }, { status: 502 });
    }
    return Response.json({ content: content.trim() });
  } catch {
    return Response.json({ error: "timeout" }, { status: 504 });
  }
}
