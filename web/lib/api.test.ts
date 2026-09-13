import {afterEach, describe, expect, it, vi} from "vitest";

import {analysisFailureMessage, getAnalysis, meSchema} from "./api";

afterEach(() => vi.restoreAllMocks());

describe("administrator-safe API presentation", () => {
  it("parses an explicit exemption without inventing an infinite credit balance", () => {
    const account = meSchema.parse({
      kind: "user", signed_in: true, verified: true, role: "admin",
      credits: 0, credit_exempt: true, profile: null,
    });
    expect(account.credits).toBe(0);
    expect(account.credit_exempt).toBe(true);
    expect(String(account.credits)).not.toBe("Infinity");
  });

  it("uses refund wording only when the backend supplied it", () => {
    expect(analysisFailureMessage(null)).toBe(
      "The analysis could not be completed. Please try again later.",
    );
    expect(analysisFailureMessage(null)).not.toMatch(/refund|reserved credit/i);
    expect(analysisFailureMessage(
      "The analysis could not be completed. Any reserved credit was returned.",
    )).toMatch(/reserved credit was returned/i);
  });

  it("keeps unexpected errors safe while exposing the request reference", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail:"An unexpected error occurred.", request_id:"support-123",
    }), {status:500, headers:{"Content-Type":"application/json", "X-Request-ID":"support-123"}})));
    await expect(getAnalysis("analysis-1")).rejects.toThrow(
      "The service is temporarily unavailable. Please try again later. Reference: support-123",
    );
  });
});
