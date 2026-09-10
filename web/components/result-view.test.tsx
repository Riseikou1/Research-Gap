import React from "react";
import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import {ResultView} from "./result-view";
import {realisticAnalysis} from "@/test-fixtures/analysis-result";
import {analysisSchema} from "@/lib/api";

describe("ResultView", () => {
  it("renders the real rationale, evidence fields, coverage, and verification in one report", () => {
    render(<ResultView analysis={realisticAnalysis}/>);
    expect(screen.getByText("Coverage is promising but bounded.")).toBeInTheDocument();
    expect(screen.getByText("Retrieval improved grounded answers")).toBeInTheDocument();
    expect(screen.getByText(/Grounded answer F1 improved/)).toBeInTheDocument();
    expect(screen.getByText("External multilingual validation")).toBeInTheDocument();
    expect(screen.getByText(/A counterexample needs closer review/)).toBeInTheDocument();
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("inspected at full-text level")).toBeInTheDocument();
  });

  it("suppresses empty scientific sections for Quick Search", () => {
    const quick = analysisSchema.parse({...realisticAnalysis, mode:"quick", result:{
      mode:"quick",candidate_count:1,papers:realisticAnalysis.result?.papers,queries:[],
      retrieved_paper_ids:["W1"],evidence:[],gaps:[],idea_assessment:null,landscape:null,
      notices:[],analysis_notices:[],coverage_messages:[],failure_summary:{retrieval:0,extraction:0},
      work_metrics:{},stage_timings:{},full_text_requested:false,
    }});
    render(<ResultView analysis={quick}/>);
    expect(screen.getByText(/did not extract structured evidence/i)).toBeInTheDocument();
    expect(screen.queryByRole("heading", {name:"Candidate research gaps"})).not.toBeInTheDocument();
    expect(screen.queryByText("Not extracted")).not.toBeInTheDocument();
  });

  it("never displays raw provider errors", () => {
    const safe = analysisSchema.parse({...realisticAnalysis,result:{...realisticAnalysis.result,
      coverage_messages:["Two literature routes were unavailable."],failure_summary:{retrieval:2,extraction:0}}});
    render(<ResultView analysis={safe}/>);
    expect(screen.getByText("Two literature routes were unavailable.")).toBeInTheDocument();
    expect(screen.queryByText(/OpenAIError|Traceback|internal\.local/i)).not.toBeInTheDocument();
  });
});
