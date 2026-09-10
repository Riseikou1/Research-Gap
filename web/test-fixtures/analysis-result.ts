import {analysisSchema} from "@/lib/api";

export const realisticAnalysis = analysisSchema.parse({
  analysis_id: "analysis-1", research_idea: "Can a small clinical RAG model improve multilingual answers?",
  status: "completed", mode: "full", stage: "completed", created_at: "2026-09-10T00:00:00Z",
  completed_at: "2026-09-10T00:01:00Z", configuration: {full_text: true}, progress: {},
  result: {
    mode: "full", full_text_requested: true, candidate_count: 12, retrieved_paper_ids: ["W1","W2"],
    papers: [
      {id:"W1",title:"Multilingual clinical retrieval",abstract:"A clinical study.",authors:["Ada Kim"],publication_year:2025,doi:"10.1/example",url:"https://example.test/paper",citation_count:3,full_text_locations:[],provenance:[],final_score:.91},
      {id:"W2",title:"A counterexample study",abstract:null,authors:[],publication_year:2024,doi:null,url:null,citation_count:1,full_text_locations:[],provenance:[],final_score:.72},
    ],
    evidence: [{paper_id:"W1",title:"Multilingual clinical retrieval",study_type:"empirical",
      research_objective:{value:"Evaluate multilingual clinical retrieval",evidence_text:"We evaluate retrieval for three clinical languages.",source:"full_text",confidence:.9,section_type:"introduction",section_heading:"Introduction",section_id:"s1"},
      population_or_setting:[{value:"Hospital questions",evidence_text:"Questions were collected in a hospital.",source:"full_text",confidence:.8,section_type:"methods",section_heading:"Methods",section_id:"s2"}],
      method_or_intervention:[{value:"Retrieval-augmented generation",evidence_text:"We use a retrieval-augmented generator.",source:"abstract",confidence:.9}],
      comparison_or_baseline:[{value:"No-retrieval baseline",evidence_text:"We compare against generation without retrieval.",source:"full_text",confidence:.8,section_type:"experimental_setup",section_heading:"Evaluation",section_id:"s3"}],
      data_or_modality:[{value:"Clinical text",evidence_text:"The input is clinical text.",source:"abstract",confidence:.8}],
      datasets:[{value:"ClinicQA",evidence_text:"Experiments use ClinicQA.",source:"full_text",confidence:.9,section_type:"dataset",section_heading:"Dataset",section_id:"s4"}],
      sample_size:{value:"1,200 questions",evidence_text:"The set contains 1,200 questions.",source:"full_text",confidence:.9,section_type:"dataset",section_heading:"Dataset",section_id:"s4"},
      evaluation_metrics:[{value:"F1",evidence_text:"We report macro F1.",source:"full_text",confidence:.9,section_type:"results",section_heading:"Results",section_id:"s5"}],
      main_findings:[{value:"Retrieval improved grounded answers",evidence_text:"Grounded answer F1 improved by six points.",source:"full_text",confidence:.88,section_type:"results",section_heading:"Results",section_id:"s5"}],
      constraints:[], limitations:[{value:"Single hospital",evidence_text:"This study is limited to one hospital.",source:"full_text",confidence:.9,section_type:"limitations",section_heading:"Limitations",section_id:"s6",author_stated:true}],
      future_work:[{value:"External validation",evidence_text:"Future work should validate other hospitals.",source:"full_text",confidence:.9,section_type:"future_work",section_heading:"Future work",section_id:"s7"}],
      extraction_confidence:.88,missing_fields:["constraints"],coverage:{source_level:"full_text",full_text_status:"usable",inspected_section_types:["introduction","methods","dataset","results","limitations","future_work"],structure_available:true,truncated:false,notices:[]}
    }],
    gaps:[{id:"gap-1",title:"External multilingual validation",description:"Validation across hospitals and languages remains limited.",category:"population_or_setting",rationale:"The selected evidence uses one hospital.",supporting_paper_ids:["W1"],supporting_evidence:[{paper_id:"W1",evidence_type:"limitation",value:"Single hospital",evidence_text:"This study is limited to one hospital.",study_type:"empirical",role:"direct_support"}],pattern_type:"limited_external_validation",landscape_basis:[{dimension:"population_or_setting",value:"Hospital questions",count:1,total:1,prevalence:1,paper_ids:["W1"]}],verification_queries:[{candidate_id:"gap-1",query:"multilingual clinical RAG external validation",pattern_type:"limited_external_validation",strategy:"counterexample",source:"deterministic"}],verification:{candidate_id:"gap-1",verification_queries:[],searched_paper_ids:["W2"],supporting_paper_ids:["W1"],potential_contradiction_paper_ids:["W2"],contradicting_paper_ids:["W2"],evidence:[],coverage_notes:["One targeted query completed."],label:"uncertain",reason:"A counterexample needs closer review."},final_label:"uncertain",potentially_contradicting_paper_ids:["W2"],contradicting_paper_ids:["W2"],potentially_contradicting_evidence:[],contradicting_evidence:[],verification_status:"verified"}],
    idea_assessment:{label:"uncertain",rationale:"Coverage is promising but bounded.",supporting_paper_ids:["W1"],supporting_evidence:[{paper_id:"W1",evidence_type:"method",value:"RAG",evidence_text:"We use a retrieval-augmented generator.",study_type:"empirical",role:"contextual_support"}],counterexample_paper_ids:["W2"],counterexample_evidence:[{paper_id:"W2",evidence_type:"direct_match",value:"Similar evaluation",evidence_text:"A similar evaluation was reported.",study_type:"empirical",role:"confirmed_direct_match"}],partial_match_paper_ids:["W1"],potential_match_paper_ids:["W2"],matched_facets:{W1:["method","population"]},verification_queries:[],searched_paper_ids:["W1","W2"],coverage_notes:["The search was bounded."]},
    landscape:{total_papers:1,papers:[],frequencies:[{dimension:"method",value:"Retrieval-augmented generation",count:1,prevalence:1,paper_ids:["W1"]}],combinations:[],missing_field_counts:{constraints:1},source_coverage:{source_levels:{full_text:1,abstract:0,metadata_only:0},full_text_outcomes:{usable:1,unavailable:0,fetch_failed:0,parse_failed:0,not_attempted:0},truncated_full_text_documents:0},conflicts:[]},
    notices:[],analysis_notices:[],coverage_messages:[],failure_summary:{retrieval:0,extraction:0},queries:[{text:"multilingual clinical RAG",strategy:"original"}],ranking_mode:"hybrid",work_metrics:{},stage_timings:{planning:.2,ranking:.3}
  }
});
