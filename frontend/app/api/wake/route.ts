/** Probes the Modal vLLM server. 200 from /v1/models means the GPU is up;
 *  503 or a timeout means it is still cold-starting. The probe itself
 *  triggers the cold start, so the client polls this until ready. */
export async function GET() {
  const base = process.env.MORRIS_ENDPOINT;
  if (!base) {
    return Response.json({ ready: false, error: "MORRIS_ENDPOINT is not set" }, { status: 500 });
  }
  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/v1/models`, {
      signal: AbortSignal.timeout(8000),
      cache: "no-store",
    });
    return Response.json({ ready: res.ok });
  } catch {
    return Response.json({ ready: false });
  }
}
