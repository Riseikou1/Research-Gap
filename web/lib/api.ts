import {z} from "zod";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const status = z.enum(["pending", "running", "completed", "failed"]);
export const analysisSchema = z.object({
  analysis_id: z.string(), research_idea: z.string(), status, mode: z.enum(["quick", "full"]),
  stage: z.string(), created_at: z.string(), started_at: z.string().nullable().optional(),
  completed_at: z.string().nullable().optional(), configuration: z.record(z.string(), z.unknown()).optional(),
  progress: z.record(z.string(), z.unknown()).optional(), result: z.record(z.string(), z.unknown()).nullable().optional(),
  error_message: z.string().nullable().optional(), decomposer: z.string().optional(),
  query_generator: z.string().optional(), paper_limit: z.number().optional()
});
export type Analysis = z.infer<typeof analysisSchema>;
export const meSchema = z.object({
  kind: z.string(), signed_in: z.boolean(), verified: z.boolean(), role: z.string(), credits: z.number(),
  profile: z.object({display_name: z.string(), avatar_url: z.string().nullable(), email: z.string().nullable()}).nullable(),
  plan: z.object({test_mode: z.boolean(), price_usd: z.number(), credits_per_cycle: z.number()}).optional()
});
export type Me = z.infer<typeof meSchema>;

async function call(path: string, token?: string | null, init?: RequestInit): Promise<unknown> {
  const response = await fetch(`${API_URL}${path}`, {credentials: "include", ...init, headers: {
    ...(init?.body ? {"Content-Type": "application/json"} : {}), ...(token ? {Authorization: `Bearer ${token}`} : {}),
    ...init?.headers
  }});
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
  }
  if (response.status === 204) return null;
  return response.json();
}
export async function getMe(token?: string | null) { return meSchema.parse(await call("/me", token)); }
export async function createAnalysis(input: object, token?: string | null) {
  return z.object({analysis_id: z.string(), status}).parse(await call("/analyses", token, {method: "POST", body: JSON.stringify(input)}));
}
export async function getAnalysis(id: string, token?: string | null) { return analysisSchema.parse(await call(`/analyses/${encodeURIComponent(id)}`, token)); }
export async function getHistory(token?: string | null) { return z.array(analysisSchema.pick({analysis_id: true, research_idea: true, status: true, mode: true, stage: true, created_at: true, completed_at: true})).parse(await call("/analyses", token)); }
export async function post(path: string, token?: string | null, body?: object) { return call(path, token, {method: "POST", body: body ? JSON.stringify(body) : undefined}); }
export {API_URL};
