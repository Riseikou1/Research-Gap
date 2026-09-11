import {z} from "zod";

const stringList = z.array(z.string()).default([]);

export const evidenceItemSchema = z.object({
  value: z.string(),
  canonical_value: z.string().nullable().optional(),
  evidence_text: z.string(),
  source: z.enum(["title", "abstract", "full_text"]),
  confidence: z.number(),
  section_type: z.string().nullable().optional(),
  section_heading: z.string().nullable().optional(),
  section_id: z.string().nullable().optional(),
  author_stated: z.literal(true).optional(),
});

export const coverageSchema = z.object({
  source_level: z.enum(["metadata_only", "abstract", "full_text"]),
  full_text_status: z.enum(["not_attempted", "unavailable", "fetch_failed", "parse_failed", "usable"]),
  inspected_section_types: stringList,
  structure_available: z.boolean().default(false),
  truncated: z.boolean().default(false),
  notices: stringList,
});

const evidenceFields = {
  population_or_setting: z.array(evidenceItemSchema).default([]),
  method_or_intervention: z.array(evidenceItemSchema).default([]),
  comparison_or_baseline: z.array(evidenceItemSchema).default([]),
  data_or_modality: z.array(evidenceItemSchema).default([]),
  datasets: z.array(evidenceItemSchema).default([]),
  evaluation_metrics: z.array(evidenceItemSchema).default([]),
  main_findings: z.array(evidenceItemSchema).default([]),
  constraints: z.array(evidenceItemSchema).default([]),
  limitations: z.array(evidenceItemSchema).default([]),
  future_work: z.array(evidenceItemSchema).default([]),
};

export const paperEvidenceSchema = z.object({
  paper_id: z.string(), title: z.string(),
  study_type: z.enum(["empirical", "review", "survey", "methodological", "other"]),
  research_objective: evidenceItemSchema.nullable().default(null),
  ...evidenceFields,
  sample_size: evidenceItemSchema.nullable().default(null),
  extraction_confidence: z.number().default(0),
  missing_fields: stringList,
  coverage: coverageSchema.nullable().default(null),
});

export const paperSchema = z.object({
  id: z.string(), title: z.string(), abstract: z.string().nullable().default(null),
  authors: stringList, publication_year: z.number().nullable().default(null),
  publication_date: z.string().nullable().optional(), doi: z.string().nullable().default(null),
  openalex_id: z.string().nullable().optional(), citation_count: z.number().default(0),
  source: z.string().nullable().optional(), url: z.string().nullable().default(null),
  full_text_locations: z.array(z.object({url: z.string(), source_format: z.string(), is_open_access: z.boolean()})).default([]),
  provenance: z.array(z.unknown()).default([]), lexical_score: z.number().nullable().optional(),
  semantic_score: z.number().nullable().optional(), final_score: z.number().nullable().optional(),
  ranking_mode: z.string().nullable().optional(), matched_queries: stringList.optional(),
  retrieval_modes: stringList.optional(), retrieved_by: stringList.optional(),
});

const gapEvidenceSchema = z.object({
  paper_id: z.string(), evidence_type: z.string(), value: z.string(), evidence_text: z.string(),
  study_type: z.string().nullable().optional(), role: z.string(),
});
const verificationQuerySchema = z.object({
  candidate_id: z.string(), query: z.string(), pattern_type: z.string(), strategy: z.string(), source: z.string(),
});
const verificationSchema = z.object({
  candidate_id: z.string(), verification_queries: z.array(verificationQuerySchema).default([]),
  searched_paper_ids: stringList, supporting_paper_ids: stringList,
  potential_contradiction_paper_ids: stringList, contradicting_paper_ids: stringList,
  evidence: z.array(gapEvidenceSchema).default([]), coverage_notes: stringList,
  label: z.enum(["well_studied", "uncertain", "promising_gap"]), reason: z.string(),
}).nullable().default(null);

export const gapSchema = z.object({
  id: z.string(), title: z.string(), description: z.string(), category: z.string(), rationale: z.string(),
  supporting_paper_ids: stringList, supporting_evidence: z.array(gapEvidenceSchema).default([]),
  pattern_type: z.string(), landscape_basis: z.array(z.object({
    dimension: z.string(), value: z.string(), count: z.number(), total: z.number(),
    prevalence: z.number(), paper_ids: stringList,
  })).default([]),
  verification_queries: z.array(verificationQuerySchema).default([]), verification: verificationSchema,
  final_label: z.enum(["well_studied", "uncertain", "promising_gap"]).nullable().default(null),
  potentially_contradicting_paper_ids: stringList, contradicting_paper_ids: stringList,
  potentially_contradicting_evidence: z.array(gapEvidenceSchema).default([]),
  contradicting_evidence: z.array(gapEvidenceSchema).default([]), verification_status: z.string(),
});

export const ideaAssessmentSchema = z.object({
  label: z.enum(["well_studied", "uncertain", "promising_gap"]), rationale: z.string(),
  supporting_paper_ids: stringList, supporting_evidence: z.array(gapEvidenceSchema).default([]),
  counterexample_paper_ids: stringList, counterexample_evidence: z.array(gapEvidenceSchema).default([]),
  partial_match_paper_ids: stringList, potential_match_paper_ids: stringList,
  matched_facets: z.record(z.string(), z.array(z.string())).default({}),
  verification_queries: z.array(verificationQuerySchema).default([]), searched_paper_ids: stringList,
  coverage_notes: stringList,
});

const landscapeSchema = z.object({
  total_papers: z.number(),
  frequencies: z.array(z.object({dimension: z.string(), value: z.string(), count: z.number(), prevalence: z.number(), paper_ids: stringList})).default([]),
  combinations: z.array(z.object({dimensions: z.record(z.string(), z.string()), count: z.number(), prevalence: z.number(), paper_ids: stringList})).default([]),
  conflicts: z.array(z.object({paper_ids: stringList, topic: z.string(), status: z.string(), reason: z.string()})).default([]),
  missing_field_counts: z.record(z.string(), z.number()).default({}),
  source_coverage: z.object({
    source_levels: z.record(z.string(), z.number()).default({}),
    full_text_outcomes: z.record(z.string(), z.number()).default({}),
    truncated_full_text_documents: z.number().default(0),
  }).default({source_levels: {}, full_text_outcomes: {}, truncated_full_text_documents: 0}),
}).nullable().default(null);

export const analysisResultSchema = z.object({
  mode: z.enum(["quick", "full"]).optional(), full_text_requested: z.boolean().default(false),
  candidate_count: z.number().default(0), retrieved_paper_ids: stringList,
  papers: z.array(paperSchema).default([]), evidence: z.array(paperEvidenceSchema).default([]),
  gaps: z.array(gapSchema).default([]), idea_assessment: ideaAssessmentSchema.nullable().default(null),
  landscape: landscapeSchema, notices: stringList, analysis_notices: stringList,
  coverage_messages: stringList,
  failure_summary: z.object({retrieval: z.number().default(0), extraction: z.number().default(0)}).default({retrieval: 0, extraction: 0}),
  extraction_coverage: z.object({
    selected_for_report: z.number().default(0),
    requested_for_extraction: z.number().default(0),
    successful_evidence_records: z.number().default(0),
    failed_extractions: z.number().default(0),
    not_requested_for_extraction: z.number().default(0),
    partial: z.boolean().default(false),
  }).default({selected_for_report: 0, requested_for_extraction: 0, successful_evidence_records: 0, failed_extractions: 0, not_requested_for_extraction: 0, partial: false}),
  queries: z.array(z.object({text: z.string(), strategy: z.string().optional(), source: z.string().optional()})).default([]),
  ranking_mode: z.string().optional(), work_metrics: z.record(z.string(), z.number()).default({}),
  stage_timings: z.record(z.string(), z.number()).default({}),
  duration_seconds: z.number().nonnegative().nullable().optional(),
});

export type AnalysisResult = z.infer<typeof analysisResultSchema>;
export type PaperEvidence = z.infer<typeof paperEvidenceSchema>;
export type EvidenceItem = z.infer<typeof evidenceItemSchema>;
export type Gap = z.infer<typeof gapSchema>;
