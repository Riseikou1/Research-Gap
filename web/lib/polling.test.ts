import {describe, expect, it} from "vitest";
import {shouldContinuePolling} from "./polling";
describe("analysis polling", () => {it("stops at every terminal status", () => {
  expect(shouldContinuePolling("pending")).toBe(true); expect(shouldContinuePolling("running")).toBe(true);
  expect(shouldContinuePolling("completed")).toBe(false); expect(shouldContinuePolling("failed")).toBe(false);
});});
