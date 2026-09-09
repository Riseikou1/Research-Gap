import React from "react";
import {fireEvent, render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import {ResultView} from "./result-view";
import type {Analysis} from "@/lib/api";

const base: Analysis = {
  analysis_id: "a1", research_idea: "Example idea", status: "completed", mode: "full",
  stage: "completed", created_at: new Date().toISOString(), configuration: {}, progress: {},
  result: {
    candidate_count: 1,
    idea_assessment: {label: "uncertain", reason: "Coverage is sparse."},
    papers: [{id: "W1", title: "Real-shaped paper", publication_year: 2024,
      authors: ["Ada"], abstract: null, final_score: .8}],
    evidence: [{paper_id: "W1", title: "Real-shaped paper", missing_fields: ["datasets"],
      method_or_intervention: [], datasets: [], main_findings: [], limitations: []}],
    gaps: [], landscape: {total_papers: 1, frequencies: [], conflicts: []}, queries: [],
    stage_timings: {planning: .2}
  }
};

describe("ResultView", () => {
  it("renders a summary first and labels missing evidence", () => {
    render(<ResultView analysis={base}/>);
    expect(screen.getByText("Coverage is sparse.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", {name: "Evidence"}));
    expect(screen.getAllByText(/Not extracted/).length).toBeGreaterThan(0);
  });
  it("does not claim verification for Quick Search", () => {
    const quick: Analysis = {...base, mode: "quick", result: {...base.result, idea_assessment: null}};
    render(<ResultView analysis={quick}/>);
    expect(screen.getByText(/did not extract evidence, generate gaps, or perform verification/i)).toBeInTheDocument();
    expect(screen.queryByRole("tab", {name: "Research gaps"})).not.toBeInTheDocument();
  });
});
