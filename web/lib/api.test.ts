import {describe, expect, it} from "vitest";

import {analysisFailureMessage, meSchema} from "./api";

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
});
