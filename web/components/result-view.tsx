"use client";

import {useState, type ReactNode} from "react";
import type {Analysis} from "@/lib/api";
import type {EvidenceItem, Gap, PaperEvidence} from "@/lib/result-schema";
import {StatusPill} from "./status-pill";

const sectionLabels: Record<string, string> = {
  research_objective: "Research objective", population_or_setting: "Population or setting",
  method_or_intervention: "Method or intervention", comparison_or_baseline: "Comparison or baseline",
  data_or_modality: "Data or modality", datasets: "Dataset", sample_size: "Sample size",
  evaluation_metrics: "Evaluation metric", main_findings: "Finding", constraints: "Constraint",
  limitations: "Limitation", future_work: "Future work",
};

type Result = NonNullable<Analysis["result"]>;
type Paper = Result["papers"][number];
type PaperMap = Map<string, Paper>;
type EvidenceField = keyof Pick<PaperEvidence,"research_objective"|"population_or_setting"|"method_or_intervention"|"comparison_or_baseline"|"data_or_modality"|"datasets"|"sample_size"|"evaluation_metrics"|"main_findings"|"constraints"|"limitations"|"future_work">;

export function ResultView({analysis}: {analysis: Analysis}) {
  const result = analysis.result;
  if (!result) return null;
  const isQuick = analysis.mode === "quick";
  const assessment = result.idea_assessment;
  const evidence = result.evidence;
  const landscape = result.landscape;
  const dimensions = groupLandscape(landscape?.frequencies ?? []);
  const coverage = landscape?.source_coverage;
  const extraction = result.extraction_coverage;
  const fullTextRequested = result.full_text_requested || analysis.configuration?.full_text === true;
  const papers = new Map(result.papers.map(paper => [paper.id, paper]));
  const duration = reportDuration(result);

  const hasExisting = hasEvidence(evidence, ["research_objective"])
    || Boolean(dimensions.problem?.length || dimensions.study_type?.length);
  const hasMethods = hasEvidence(evidence, ["method_or_intervention"])
    || Boolean(dimensions.method?.length || dimensions.method_family?.length);
  const hasContexts = hasEvidence(evidence, ["population_or_setting", "data_or_modality", "datasets", "sample_size", "constraints"])
    || Boolean(dimensions.population_or_setting?.length || dimensions.dataset?.length || dimensions.dataset_type?.length);
  const hasEvaluation = hasEvidence(evidence, ["comparison_or_baseline", "evaluation_metrics"])
    || Boolean(dimensions.baseline?.length || dimensions.performance_metric?.length || dimensions.efficiency_metric?.length);
  const hasFindings = hasEvidence(evidence, ["main_findings"]);
  const hasLimitations = hasEvidence(evidence, ["limitations", "future_work"]);

  const sections = isQuick
    ? [["summary", "Summary"], ["papers", "Relevant papers"], ["methodology", "Methodology"]]
    : [
        ["summary", "Verdict"],
        ...(hasExisting ? [["existing", "Existing literature"]] : []),
        ...(hasMethods ? [["methods", "Methods"]] : []),
        ...(hasContexts ? [["contexts", "Contexts & data"]] : []),
        ...(hasEvaluation ? [["evaluation", "Evaluation"]] : []),
        ...(hasFindings ? [["findings", "Findings"]] : []),
        ...(result.gaps.length ? [["gaps", "Candidate gaps"]] : []),
        ...(hasLimitations ? [["limitations", "Limitations"]] : []),
        ["coverage", "Coverage"], ["papers", "Papers"], ["methodology", "Methodology"],
      ];

  return <div className="report-layout"><aside className="report-toc"><p>In this report</p><nav aria-label="Report sections">{sections.map(([id,label]) => <a key={id} href={`#${id}`}>{label}</a>)}</nav></aside>
    <article className="report">
      <ReportSection id="summary" eyebrow="Research idea and verdict" title={isQuick ? "Quick Search results" : pretty(assessment?.label ?? "uncertain")}>
        {!isQuick && <div className="verdict"><StatusPill status={assessment?.label ?? "uncertain"}/><p>{assessment?.rationale ?? "The available evidence did not include a direct idea assessment."}</p></div>}
        {isQuick && <p className="callout">Quick Search retrieved and ranked papers only. It did not extract structured evidence, build a landscape, generate gaps, perform verification, or inspect full text.</p>}
        <div className="metrics">
          <Metric value={result.candidate_count} label="retrieved candidates"/>
          <Metric value={result.papers.length} label="papers selected for this report"/>
          {!isQuick && <Metric value={extraction.requested_for_extraction} label="papers requested for structured extraction"/>}
          {!isQuick && <Metric value={extraction.successful_evidence_records} label="successful evidence records"/>}
          <Metric value={duration.value} label={duration.label}/>
        </div>
        {!isQuick && assessment && <AssessmentDetails assessment={assessment} papers={papers}/>}
        <p className="coverage-caveat">This is a bounded, evidence-backed investigation—not proof of global novelty and not a systematic review.</p>
      </ReportSection>

      {!isQuick && hasExisting && <ReportSection id="existing" eyebrow="Executive landscape" title="What the literature already studies well">
        <FrequencyList values={dedupeFrequencyRows([...(dimensions.problem??[]), ...(dimensions.study_type??[])])} papers={papers}/>
        <EvidenceGroup records={evidence} fields={["research_objective"]}/>
      </ReportSection>}

      {!isQuick && hasMethods && <ReportSection id="methods" eyebrow="Approaches" title="Common methods and method families">
        <FrequencyList values={dedupeFrequencyRows([...(dimensions.method??[]), ...(dimensions.method_family??[])])} papers={papers}/>
        <EvidenceGroup records={evidence} fields={["method_or_intervention"]}/>
      </ReportSection>}

      {!isQuick && hasContexts && <ReportSection id="contexts" eyebrow="Study context" title="Populations, settings, datasets, and modalities">
        <FrequencyList values={dedupeFrequencyRows([...(dimensions.population_or_setting??[]), ...(dimensions.dataset??[]), ...(dimensions.dataset_type??[])])} papers={papers}/>
        <EvidenceGroup records={evidence} fields={["population_or_setting","data_or_modality","datasets","sample_size","constraints"]}/>
      </ReportSection>}

      {!isQuick && hasEvaluation && <ReportSection id="evaluation" eyebrow="Comparisons" title="Baselines and evaluation metrics">
        <FrequencyList values={dedupeFrequencyRows([...(dimensions.baseline??[]), ...(dimensions.performance_metric??[]), ...(dimensions.efficiency_metric??[])])} papers={papers}/>
        <EvidenceGroup records={evidence} fields={["comparison_or_baseline","evaluation_metrics"]}/>
      </ReportSection>}

      {!isQuick && hasFindings && <ReportSection id="findings" eyebrow="Observed outcomes" title="Main findings in the selected literature"><EvidenceGroup records={evidence} fields={["main_findings"]}/></ReportSection>}

      {!isQuick && result.gaps.length > 0 && <ReportSection id="gaps" eyebrow="Qualified opportunities" title="Candidate research gaps"><div className="gap-list">{result.gaps.map(gap => <GapReport key={gap.id} gap={gap} papers={papers}/>)}</div></ReportSection>}

      {!isQuick && hasLimitations && <ReportSection id="limitations" eyebrow="Author-stated boundaries" title="Important limitations and future work"><EvidenceGroup records={evidence} fields={["limitations","future_work"]}/></ReportSection>}

      {!isQuick && <ReportSection id="coverage" eyebrow="Reading depth" title="Evidence coverage">
        <p><strong>Full text requested:</strong> {fullTextRequested ? "Yes" : "No"}</p>
        <div className="metrics coverage-metrics">
          <Metric value={result.candidate_count} label="retrieved candidates"/>
          <Metric value={extraction.selected_for_report} label="selected for ranking/reporting"/>
          <Metric value={extraction.requested_for_extraction} label="requested for structured extraction"/>
          <Metric value={extraction.successful_evidence_records} label="successful evidence records"/>
          <Metric value={extraction.failed_extractions} label="failed extractions"/>
          <Metric value={extraction.not_requested_for_extraction} label="selected but outside extraction limit"/>
          <Metric value={coverage?.source_levels.full_text??0} label="inspected at full-text level"/>
          <Metric value={coverage?.source_levels.abstract??0} label={fullTextRequested ? "abstract fallback" : "abstract sources"}/>
        </div>
        {inspectedSections(evidence).length > 0 && <p><strong>Semantic sections inspected:</strong> {inspectedSections(evidence).map(pretty).join(", ")}.</p>}
        {(result.coverage_messages.length > 0 || fullTextRequested) && <div className="callout"><strong>{extraction.partial ? "Partial evidence coverage" : "How source coverage works"}</strong>{result.coverage_messages.map(message=><p key={message}>{message}</p>)}{fullTextRequested&&<p>When open full text could not be accessed, the system used the abstract or metadata available for that paper. This lowers coverage; it is not presented as a full read.</p>}</div>}
        {evidence.length > 0 && <details><summary>Paper-by-paper evidence provenance</summary><div className="evidence-records">{evidence.map(record=><PaperEvidenceCard key={record.paper_id} record={record}/>)}</div></details>}
      </ReportSection>}

      <ReportSection id="papers" eyebrow="Source set" title="Relevant papers"><Papers papers={result.papers}/></ReportSection>
      <ReportSection id="methodology" eyebrow="Progressive disclosure" title="Methodology and technical details">
        <details><summary>Search queries</summary><ol>{result.queries.map(query=><li key={`${query.strategy}-${query.text}`}>{query.text} {query.strategy&&<span className="hint">({pretty(query.strategy)})</span>}</li>)}</ol></details>
        {!isQuick && landscape?.conflicts.length ? <details><summary>Counterexamples and literature conflicts</summary>{landscape.conflicts.map(conflict=><div key={`${conflict.topic}-${conflict.paper_ids.join()}`}><strong>{conflict.topic}</strong><p>{conflict.reason}</p><Citations ids={conflict.paper_ids} papers={papers}/></div>)}</details> : null}
        <details><summary>Configuration and work metrics</summary><p className="hint">Stage diagnostics may overlap their parent stages and must not be added together.</p><pre>{JSON.stringify({configuration:analysis.configuration, work_metrics:result.work_metrics, stage_timings:result.stage_timings, duration_seconds:result.duration_seconds},null,2)}</pre></details>
      </ReportSection>
    </article>
  </div>;
}

function ReportSection({id, eyebrow, title, children}: {id:string; eyebrow:string; title:string; children:ReactNode}) {return <section className="report-section" id={id}><p className="eyebrow">{eyebrow}</p><h2>{title}</h2>{children}</section>}
function Metric({value,label}:{value:string|number;label:string}) {return <div className="metric"><strong>{value}</strong><span>{label}</span></div>}
function pretty(value:string){return value.replaceAll("_"," ").replace(/^./,letter=>letter.toUpperCase())}
function canonical(value:string){return value.toLocaleLowerCase().replaceAll(/[-_]/g," ").replaceAll(/[^\p{L}\p{N}\s]/gu,"").replaceAll(/\s+/g," ").trim()}

const topLevelStages = ["planning","initial_retrieval","ranking_embeddings","evidence_lookup_extraction","landscape","direct_verification","candidate_generation","candidate_verification"];
export function reportDuration(result: Pick<Result,"duration_seconds"|"stage_timings">) {
  if (typeof result.duration_seconds === "number") return {value:`${result.duration_seconds.toFixed(1)}s`,label:"total elapsed wall-clock time"};
  const seconds = topLevelStages.reduce((sum,key)=>sum+(result.stage_timings[key]??0),0);
  return seconds > 0 ? {value:`≈${seconds.toFixed(1)}s`,label:"approx. top-level stage time (legacy result)"} : {value:"—",label:"total elapsed time unavailable"};
}

function groupLandscape(rows:{dimension:string;value:string;count:number;prevalence:number;paper_ids:string[]}[]){return rows.reduce<Record<string,typeof rows>>((group,row)=>{(group[row.dimension]??=[]).push(row);return group}, {})}
function dedupeFrequencyRows<T extends {value:string;paper_ids:string[]}>(rows:T[]):T[]{const seen=new Map<string,T>();for(const row of rows){const key=canonical(row.value);const existing=seen.get(key);if(!existing){seen.set(key,{...row,paper_ids:[...row.paper_ids]});continue}existing.paper_ids=[...new Set([...existing.paper_ids,...row.paper_ids])]}return [...seen.values()]}
function FrequencyList({values,papers}:{values:{dimension:string;value:string;count:number;prevalence:number;paper_ids:string[]}[];papers:PaperMap}){if(!values.length)return null;return <div className="frequency-list">{values.slice(0,12).map(row=><div key={`${row.dimension}-${row.value}`}><span><strong>{row.value}</strong><small>{pretty(row.dimension)} · {row.count} paper{row.count===1?"":"s"}</small></span><span className="frequency-track" aria-label={`${Math.round(row.prevalence*100)} percent`}><i style={{width:`${Math.max(3,Math.round(row.prevalence*100))}%`}}/></span><Citations ids={row.paper_ids} papers={papers}/></div>)}</div>}

function itemsFor(record:PaperEvidence, field:EvidenceField):EvidenceItem[]{const value=record[field];return Array.isArray(value)?value:value?[value]:[]}
function hasEvidence(records:PaperEvidence[], fields:EvidenceField[]){return records.some(record=>fields.some(field=>itemsFor(record,field).length>0))}
function EvidenceGroup({records,fields}:{records:PaperEvidence[];fields:EvidenceField[]}){const entries=records.flatMap(record=>fields.flatMap(field=>itemsFor(record,field).map(item=>({record,field,item}))));if(!entries.length)return null;return <div className="claim-list">{entries.map(({record,field,item},index)=><article key={`${record.paper_id}-${field}-${index}`}><p className="claim-type">{sectionLabels[field]} · {record.study_type}</p><h3>{item.value}</h3><blockquote>{item.evidence_text}</blockquote><p className="provenance"><strong>{record.title}</strong> <span>· {pretty(item.source)}{item.section_heading?` · ${item.section_heading}`:""}{item.section_type?` (${pretty(item.section_type)})`:""}</span><code>{record.paper_id}</code></p></article>)}</div>}

function AssessmentDetails({assessment,papers}:{assessment:NonNullable<Result["idea_assessment"]>;papers:PaperMap}){return <details className="assessment-details"><summary>Evidence behind the verdict</summary><IdGroup label="Supporting papers" ids={assessment.supporting_paper_ids} papers={papers}/><IdGroup label="Counterexample papers" ids={assessment.counterexample_paper_ids} papers={papers}/>{assessment.supporting_evidence.length>0&&<div><h3>Supporting evidence</h3>{assessment.supporting_evidence.map((item,index)=><GapEvidenceRow key={`support-${index}`} item={item} papers={papers}/>)}</div>}{assessment.counterexample_evidence.length>0&&<div><h3>Counterexample evidence</h3>{assessment.counterexample_evidence.map((item,index)=><GapEvidenceRow key={`counter-${index}`} item={item} papers={papers}/>)}</div>}<IdGroup label="Partial matches" ids={assessment.partial_match_paper_ids} papers={papers}/><IdGroup label="Potential matches" ids={assessment.potential_match_paper_ids} papers={papers}/>{Object.entries(assessment.matched_facets).length>0&&<div><h3>Matched facets</h3>{Object.entries(assessment.matched_facets).map(([id,facets])=><p key={id}><strong>{papers.get(id)?.title ?? "Paper record"}</strong> <code>{id}</code>: {facets.join(", ")}</p>)}</div>}{assessment.coverage_notes.length>0&&<div><h3>Coverage notes</h3><ul>{assessment.coverage_notes.map(note=><li key={note}>{note}</li>)}</ul></div>}</details>}
function GapReport({gap,papers}:{gap:Gap;papers:PaperMap}){const verification=gap.verification;return <article className="gap-card"><div className="gap-heading"><StatusPill status={gap.final_label??verification?.label??"uncertain"}/><span>{pretty(gap.category)}</span></div><h3>{gap.title}</h3><p>{gap.description}</p><p><strong>Why it was suggested:</strong> {gap.rationale}</p><Citations ids={gap.supporting_paper_ids} papers={papers}/>{gap.landscape_basis.length>0&&<p><strong>Landscape basis:</strong> {gap.landscape_basis.map(item=>`${item.value} (${item.count}/${item.total} papers)`).join("; ")}</p>}{gap.supporting_evidence.length>0&&<details><summary>Supporting evidence</summary>{gap.supporting_evidence.map((item,index)=><GapEvidenceRow key={index} item={item} papers={papers}/>)}</details>}<details><summary>Verification and counterexamples</summary><p><strong>Status:</strong> {pretty(gap.verification_status)} · <strong>pattern:</strong> {pretty(gap.pattern_type)}</p>{verification&&<><p>{verification.reason}</p><IdGroup label="Potentially contradicting papers" ids={gap.potentially_contradicting_paper_ids} papers={papers}/><IdGroup label="Confirmed counterexamples" ids={gap.contradicting_paper_ids} papers={papers}/>{gap.potentially_contradicting_evidence.map((item,index)=><GapEvidenceRow key={`potential-${index}`} item={item} papers={papers}/>)}{gap.contradicting_evidence.map((item,index)=><GapEvidenceRow key={`confirmed-${index}`} item={item} papers={papers}/>)}{verification.coverage_notes.length>0&&<ul>{verification.coverage_notes.map(note=><li key={note}>{note}</li>)}</ul>}</>}{gap.verification_queries.length>0&&<><h4>Verification queries</h4><ul>{gap.verification_queries.map(query=><li key={query.query}>{query.query}</li>)}</ul></>}</details></article>}
function GapEvidenceRow({item,papers}:{item:Gap["supporting_evidence"][number];papers:PaperMap}){return <div className="gap-evidence"><p><strong>{item.value}</strong> <span className="hint">({pretty(item.role)})</span></p><blockquote>{item.evidence_text}</blockquote><p className="provenance"><strong>{papers.get(item.paper_id)?.title ?? "Paper record"}</strong><code>{item.paper_id}</code></p></div>}
function IdGroup({label,ids,papers}:{label:string;ids:string[];papers:PaperMap}){return ids.length?<div><strong>{label}:</strong><Citations ids={ids} papers={papers}/></div>:null}
function Citations({ids,papers}:{ids:string[];papers:PaperMap}){return ids.length?<p className="citation-ids"><span>Sources</span>{ids.map(id=>{const paper=papers.get(id);return <span className="citation" key={id}>{paper?.url?<a href={paper.url} target="_blank" rel="noreferrer">{paper.title}</a>:<strong>{paper?.title??"Paper record"}</strong>}<code>{id}</code></span>})}</p>:null}
function inspectedSections(records:PaperEvidence[]){return [...new Set(records.flatMap(record=>record.coverage?.inspected_section_types??[]))]}
function PaperEvidenceCard({record}:{record:PaperEvidence}){const coverage=record.coverage;return <article className="evidence-card"><h3>{record.title}</h3><p><code>{record.paper_id}</code> · {pretty(record.study_type)}</p>{coverage&&<p><strong>Coverage:</strong> {pretty(coverage.source_level)} · full-text status: {pretty(coverage.full_text_status)}{coverage.truncated?" · truncated":""}</p>}{coverage?.inspected_section_types.length?<p><strong>Inspected sections:</strong> {coverage.inspected_section_types.map(pretty).join(", ")}</p>:null}{record.missing_fields.length>0&&<p><strong>No validated evidence found in the inspected source for:</strong> {record.missing_fields.map(pretty).join(", ")}.</p>}</article>}
function Papers({papers}:{papers:Paper[]}){const[query,setQuery]=useState("");const[sort,setSort]=useState("relevance");const visible=[...papers].filter(p=>p.title.toLowerCase().includes(query.toLowerCase())).sort((a,b)=>sort==="year"?(b.publication_year??0)-(a.publication_year??0):(b.final_score??0)-(a.final_score??0));return <><div className="paper-tools"><label>Filter papers<input value={query} onChange={event=>setQuery(event.target.value)} placeholder="Search titles"/></label><label>Sort by<select value={sort} onChange={event=>setSort(event.target.value)}><option value="relevance">Relevance</option><option value="year">Newest</option></select></label></div><div className="paper-list">{visible.map(paper=><article className="paper" key={paper.id}><p className="paper-meta">{paper.publication_year??"Year unavailable"} · relevance {typeof paper.final_score==="number"?paper.final_score.toFixed(2):"not scored"}</p><h3>{paper.title}</h3><p>{paper.authors.join(", ")||"Authors unavailable"}</p>{paper.abstract&&<p>{paper.abstract.slice(0,560)}</p>}<div className="actions">{paper.url&&<a className="button secondary" href={paper.url} target="_blank" rel="noreferrer">Open paper</a>}{paper.doi&&<a className="button quiet" href={`https://doi.org/${paper.doi.replace("https://doi.org/","")}`} target="_blank" rel="noreferrer">View DOI</a>}</div><details><summary>Retrieval provenance</summary><pre>{JSON.stringify(paper.provenance,null,2)}</pre></details></article>)}</div></>}
