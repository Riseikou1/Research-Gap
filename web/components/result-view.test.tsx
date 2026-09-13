import React from "react";
import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import {reportDuration, ResultView} from "./result-view";
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
    expect(screen.getByText("final full-text evidence records")).toBeInTheDocument();
    expect(screen.getByText("Per-paper coverage")).toBeInTheDocument();
    expect(screen.getByText(/attempted as PDF/)).toBeInTheDocument();
    expect(screen.getByText("full-text access attempts")).toBeInTheDocument();
    expect(screen.getByText("12.5s")).toBeInTheDocument();
    expect(screen.getByText("total elapsed wall-clock time")).toBeInTheDocument();
    expect(screen.getByText("papers requested for structured extraction")).toBeInTheDocument();
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
    expect(screen.queryByRole("heading", {name:"Citation context"})).not.toBeInTheDocument();
  });

  it("renders bounded citation relationships and loads legacy graph-less results", () => {
    const legacy = analysisSchema.parse({...realisticAnalysis, result:{...realisticAnalysis.result}});
    expect(legacy.result?.citation_graph.status).toBe("unavailable");
    const withGraph = analysisSchema.parse({...realisticAnalysis, result:{...realisticAnalysis.result,
      citation_graph:{status:"available",scope:"canonical_retrieved_pool",node_count:2,edge_count:1,
        component_count:1,isolated_node_count:0,external_reference_count:4,
        limitation:"Only the bounded pool is shown.",
        nodes:[
          {paper_id:"W1",aliases:["https://openalex.org/W1"],title:"Multilingual clinical retrieval",publication_year:2025,citation_count:3,relevance_score:.91,selected_for_analysis:true,in_degree:0,out_degree:1,external_reference_count:3},
          {paper_id:"W2",aliases:["https://openalex.org/W2"],title:"A counterexample study",publication_year:2024,citation_count:1,relevance_score:.72,selected_for_analysis:true,in_degree:1,out_degree:0,external_reference_count:1},
        ],edges:[{citing_paper_id:"W1",cited_paper_id:"W2"}]}
    }});
    render(<ResultView analysis={withGraph}/>);
    expect(screen.getByRole("heading", {name:"Citation context"})).toBeInTheDocument();
    expect(screen.getByText("internal citation relationships")).toBeInTheDocument();
    expect(screen.getByText("cites")).toBeInTheDocument();
    expect(screen.getByText(/descriptive only/)).toBeInTheDocument();
  });

  it("suppresses an empty graph view and bounds the accessible relationship list", () => {
    const empty = analysisSchema.parse({...realisticAnalysis, result:{...realisticAnalysis.result,
      citation_graph:{status:"available",scope:"canonical_retrieved_pool",nodes:[],edges:[],node_count:0,
        edge_count:0,component_count:0,isolated_node_count:0,external_reference_count:0,limitation:"Bounded."}
    }});
    const emptyRender = render(<ResultView analysis={empty}/>);
    expect(screen.queryByRole("heading", {name:"Citation context"})).not.toBeInTheDocument();
    emptyRender.unmount();

    const nodes = Array.from({length:52}, (_,index) => ({
      paper_id:`G${index}`,aliases:[],title:`Graph paper ${index}`,publication_year:2025,
      citation_count:0,relevance_score:.5,selected_for_analysis:index<10,in_degree:index?1:0,
      out_degree:index<51?1:0,external_reference_count:0,
    }));
    const edges = Array.from({length:51}, (_,index) => ({
      citing_paper_id:`G${index}`,cited_paper_id:`G${index+1}`,
    }));
    const bounded = analysisSchema.parse({...realisticAnalysis, result:{...realisticAnalysis.result,
      citation_graph:{status:"available",scope:"canonical_retrieved_pool",nodes,edges,node_count:52,
        edge_count:51,component_count:1,isolated_node_count:0,external_reference_count:0,limitation:"Bounded."}
    }});
    const {container} = render(<ResultView analysis={bounded}/>);
    expect(container.querySelectorAll(".citation-relationships li")).toHaveLength(50);
  });

  it("never displays raw provider errors", () => {
    const safe = analysisSchema.parse({...realisticAnalysis,result:{...realisticAnalysis.result,
      coverage_messages:["Two literature routes were unavailable."],failure_summary:{retrieval:2,extraction:0}}});
    render(<ResultView analysis={safe}/>);
    expect(screen.getByText("Two literature routes were unavailable.")).toBeInTheDocument();
    expect(screen.queryByText(/OpenAIError|Traceback|internal\.local/i)).not.toBeInTheDocument();
  });

  it("uses wall-clock duration and only top-level stages for legacy approximations", () => {
    expect(reportDuration({duration_seconds:12,stage_timings:{direct_verification:8,direct_verification_evidence_extraction:7}}).value).toBe("12.0s");
    expect(reportDuration({stage_timings:{direct_verification:8,direct_verification_evidence_extraction:7,candidate_verification:4}})).toEqual({value:"≈12.0s",label:"approx. top-level stage time (legacy result)"});
  });

  it("suppresses empty evidence sections and deduplicates identical method labels", () => {
    const methodOnly = analysisSchema.parse({...realisticAnalysis,result:{...realisticAnalysis.result,
      evidence:[{...realisticAnalysis.result!.evidence[0],research_objective:null,population_or_setting:[],comparison_or_baseline:[],data_or_modality:[],datasets:[],sample_size:null,evaluation_metrics:[],main_findings:[],constraints:[],limitations:[],future_work:[]}],
      landscape:{...realisticAnalysis.result!.landscape!,frequencies:[
        {dimension:"method",value:"Retrieval-augmented generation",count:1,prevalence:1,paper_ids:["W1"]},
        {dimension:"method_family",value:"retrieval augmented generation",count:1,prevalence:1,paper_ids:["W1"]},
      ]},gaps:[],
    }});
    render(<ResultView analysis={methodOnly}/>);
    expect(screen.queryByRole("heading",{name:"Populations, settings, datasets, and modalities"})).not.toBeInTheDocument();
    expect(screen.queryByRole("heading",{name:"Baselines and evaluation metrics"})).not.toBeInTheDocument();
    expect(screen.queryByRole("heading",{name:"Main findings in the selected literature"})).not.toBeInTheDocument();
    expect(screen.getAllByText("Retrieval-augmented generation")).toHaveLength(2);
  });

  it("shows readable paper titles as primary citations and IDs as secondary details", () => {
    const {container}=render(<ResultView analysis={realisticAnalysis}/>);
    const citation=container.querySelector(".citation");
    expect(citation).toHaveTextContent("Multilingual clinical retrieval");
    expect(citation?.querySelector("code")).toHaveTextContent("W1");
  });

  it("renders one relevant-paper card for DOI aliases", () => {
    const duplicate = {
      ...realisticAnalysis.result!.papers[0],
      id: "W1-alias",
      title: "Multilingual-clinical retrieval",
      doi: "https://doi.org/10.1/EXAMPLE",
    };
    const analysis = analysisSchema.parse({
      ...realisticAnalysis,
      result: {
        ...realisticAnalysis.result,
        papers: [...realisticAnalysis.result!.papers, duplicate],
      },
    });
    const {container} = render(<ResultView analysis={analysis}/>);
    expect(container.querySelectorAll(".paper-list .paper")).toHaveLength(2);
  });

  it("does not expand strongly authored version aliases with different DOIs", () => {
    const title="Reducing hallucination in structured outputs via Retrieval-Augmented Generation";
    const base={
      ...realisticAnalysis.result!.papers[0],title,publication_year:2024,
      authors:["Patrice Béchard","Orlando Marquez Ayala"],
    };
    const papers=[
      {...base,id:"https://openalex.org/W4394838812",openalex_id:"https://openalex.org/W4394838812",doi:"https://doi.org/10.48550/arxiv.2404.08189"},
      {...base,id:"https://openalex.org/W6966460441",openalex_id:"https://openalex.org/W6966460441",doi:"https://doi.org/10.48448/p2d8-gv20",authors:["Bechard, Patrice","Marquez, Orlando"]},
      {...base,id:"https://openalex.org/W4401042735",openalex_id:"https://openalex.org/W4401042735",doi:"https://doi.org/10.18653/v1/2024.naacl-industry.19",authors:["Orlando Ayala","Patrice Bechard"]},
    ];
    const analysis=analysisSchema.parse({...realisticAnalysis,result:{
      ...realisticAnalysis.result,papers,
    }});
    const {container}=render(<ResultView analysis={analysis}/>);
    expect(container.querySelectorAll(".paper-list .paper")).toHaveLength(1);
  });
});
