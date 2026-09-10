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

export function ResultView({analysis}: {analysis: Analysis}) {
  const result = analysis.result;
  if (!result) return null;
  const isQuick = analysis.mode === "quick"; const assessment = result.idea_assessment;
  const evidence = result.evidence; const landscape = result.landscape;
  const dimensions = groupLandscape(landscape?.frequencies ?? []);
  const coverage = landscape?.source_coverage;
  const fullTextRequested = result.full_text_requested || analysis.configuration?.full_text === true;
  const sections = isQuick ? [["summary", "Summary"], ["papers", "Relevant papers"], ["methodology", "Methodology"]] : [
    ["summary", "Verdict"], ["existing", "Existing literature"], ["methods", "Methods"], ["contexts", "Contexts & data"],
    ["evaluation", "Evaluation"], ["findings", "Findings"], ["gaps", "Candidate gaps"], ["limitations", "Limitations"],
    ["coverage", "Coverage"], ["papers", "Papers"], ["methodology", "Methodology"],
  ];
  return <div className="report-layout"><aside className="report-toc"><p>In this report</p><nav aria-label="Report sections">{sections.map(([id,label]) => <a key={id} href={`#${id}`}>{label}</a>)}</nav></aside>
    <article className="report">
      <ReportSection id="summary" eyebrow="Research idea and verdict" title={isQuick ? "Quick Search results" : pretty(assessment?.label ?? "uncertain")}>
        {!isQuick && <div className="verdict"><StatusPill status={assessment?.label ?? "uncertain"}/><p>{assessment?.rationale ?? "The available evidence did not include a direct idea assessment."}</p></div>}
        {isQuick && <p className="callout">Quick Search retrieved and ranked papers only. It did not extract structured evidence, build a landscape, generate gaps, perform verification, or inspect full text.</p>}
        <div className="metrics"><Metric value={result.candidate_count} label="candidates retrieved"/><Metric value={result.papers.length} label="papers selected"/>{!isQuick&&<Metric value={evidence.length} label="evidence records"/>}<Metric value={formatTime(result.stage_timings)} label="recorded stage time"/></div>
        {!isQuick && assessment && <AssessmentDetails assessment={assessment}/>}
        <p className="coverage-caveat">This is a bounded, evidence-backed investigation—not proof of global novelty and not a systematic review.</p>
      </ReportSection>

      {!isQuick && evidence.length > 0 && <>
        <ReportSection id="existing" eyebrow="Executive landscape" title="What the literature already studies well">
          <FrequencyList values={[...(dimensions.problem??[]), ...(dimensions.study_type??[])]}/>
          <EvidenceGroup records={evidence} fields={["research_objective"]}/>
        </ReportSection>
        <ReportSection id="methods" eyebrow="Approaches" title="Common methods and method families">
          <FrequencyList values={[...(dimensions.method??[]), ...(dimensions.method_family??[])]}/>
          <EvidenceGroup records={evidence} fields={["method_or_intervention"]}/>
        </ReportSection>
        <ReportSection id="contexts" eyebrow="Study context" title="Populations, settings, datasets, and modalities">
          <FrequencyList values={[...(dimensions.population_or_setting??[]), ...(dimensions.dataset??[]), ...(dimensions.dataset_type??[])]}/>
          <EvidenceGroup records={evidence} fields={["population_or_setting","data_or_modality","datasets","sample_size","constraints"]}/>
        </ReportSection>
        <ReportSection id="evaluation" eyebrow="Comparisons" title="Baselines and evaluation metrics">
          <FrequencyList values={[...(dimensions.baseline??[]), ...(dimensions.performance_metric??[]), ...(dimensions.efficiency_metric??[])]}/>
          <EvidenceGroup records={evidence} fields={["comparison_or_baseline","evaluation_metrics"]}/>
        </ReportSection>
        <ReportSection id="findings" eyebrow="Observed outcomes" title="Main findings in the selected literature"><EvidenceGroup records={evidence} fields={["main_findings"]}/></ReportSection>
      </>}

      {!isQuick && result.gaps.length > 0 && <ReportSection id="gaps" eyebrow="Qualified opportunities" title="Candidate research gaps">
        <div className="gap-list">{result.gaps.map(gap => <GapReport key={gap.id} gap={gap}/>)}</div>
      </ReportSection>}

      {!isQuick && hasEvidence(evidence, ["limitations","future_work"]) && <ReportSection id="limitations" eyebrow="Author-stated boundaries" title="Important limitations and future work"><EvidenceGroup records={evidence} fields={["limitations","future_work"]}/></ReportSection>}

      {!isQuick && <ReportSection id="coverage" eyebrow="Reading depth" title="Full-text and evidence coverage">
        <p><strong>Full text requested:</strong> {fullTextRequested ? "Yes" : "No"}</p>
        <div className="metrics coverage-metrics"><Metric value={result.papers.length} label="selected papers"/><Metric value={coverage?.source_levels.full_text??0} label="inspected at full-text level"/><Metric value={coverage?.source_levels.abstract??0} label={fullTextRequested ? "abstract fallback" : "abstract sources"}/><Metric value={coverage?.source_levels.metadata_only??0} label="metadata only"/><Metric value={coverage?.full_text_outcomes.unavailable??0} label="full text unavailable"/><Metric value={coverage?.full_text_outcomes.fetch_failed??0} label="fetch failures"/><Metric value={coverage?.full_text_outcomes.parse_failed??0} label="parse failures"/><Metric value={coverage?.truncated_full_text_documents??0} label="truncated"/></div>
        {inspectedSections(evidence).length > 0 && <p><strong>Semantic sections inspected:</strong> {inspectedSections(evidence).map(pretty).join(", ")}.</p>}
        {(result.coverage_messages.length > 0 || fullTextRequested) && <div className="callout"><strong>How fallback works</strong>{result.coverage_messages.map(message=><p key={message}>{message}</p>)}{fullTextRequested&&<p>When open full text could not be accessed, the system used the abstract or metadata available for that paper. This lowers coverage; it is not presented as a full read.</p>}</div>}
        <details><summary>Paper-by-paper evidence provenance</summary><div className="evidence-records">{evidence.map(record=><PaperEvidenceCard key={record.paper_id} record={record}/>)}</div></details>
      </ReportSection>}

      <ReportSection id="papers" eyebrow="Source set" title="Relevant papers"><Papers papers={result.papers}/></ReportSection>
      <ReportSection id="methodology" eyebrow="Progressive disclosure" title="Methodology and technical details">
        <details><summary>Search queries</summary><ol>{result.queries.map(query=><li key={`${query.strategy}-${query.text}`}>{query.text} {query.strategy&&<span className="hint">({pretty(query.strategy)})</span>}</li>)}</ol></details>
        {!isQuick && landscape?.conflicts.length ? <details><summary>Counterexamples and literature conflicts</summary>{landscape.conflicts.map(conflict=><div key={`${conflict.topic}-${conflict.paper_ids.join()}`}><strong>{conflict.topic}</strong><p>{conflict.reason}</p><CitationIds ids={conflict.paper_ids}/></div>)}</details> : null}
        <details><summary>Configuration and work metrics</summary><pre>{JSON.stringify({configuration:analysis.configuration, work_metrics:result.work_metrics, stage_timings:result.stage_timings},null,2)}</pre></details>
      </ReportSection>
    </article>
  </div>;
}

function ReportSection({id, eyebrow, title, children}: {id:string; eyebrow:string; title:string; children:ReactNode}) {return <section className="report-section" id={id}><p className="eyebrow">{eyebrow}</p><h2>{title}</h2>{children}</section>}
function Metric({value,label}:{value:string|number;label:string}) {return <div className="metric"><strong>{value}</strong><span>{label}</span></div>}
function pretty(value:string){return value.replaceAll("_"," ").replace(/^./,letter=>letter.toUpperCase())}
function formatTime(values:Record<string,number>){const total=Object.values(values).reduce((sum,value)=>sum+value,0);return `${total.toFixed(1)}s`}
function groupLandscape(rows:{dimension:string;value:string;count:number;prevalence:number;paper_ids:string[]}[]){return rows.reduce<Record<string,typeof rows>>((group,row)=>{(group[row.dimension]??=[]).push(row);return group}, {})}
function FrequencyList({values}:{values:{dimension:string;value:string;count:number;prevalence:number;paper_ids:string[]}[]}){if(!values.length)return null;return <div className="frequency-list">{values.slice(0,12).map(row=><div key={`${row.dimension}-${row.value}`}><span><strong>{row.value}</strong><small>{pretty(row.dimension)} · {row.count} paper{row.count===1?"":"s"}</small></span><span className="frequency-track" aria-label={`${Math.round(row.prevalence*100)} percent`}><i style={{width:`${Math.max(3,Math.round(row.prevalence*100))}%`}}/></span><CitationIds ids={row.paper_ids}/></div>)}</div>}

type EvidenceField = keyof Pick<PaperEvidence,"research_objective"|"population_or_setting"|"method_or_intervention"|"comparison_or_baseline"|"data_or_modality"|"datasets"|"sample_size"|"evaluation_metrics"|"main_findings"|"constraints"|"limitations"|"future_work">;
function itemsFor(record:PaperEvidence, field:EvidenceField):EvidenceItem[]{const value=record[field];return Array.isArray(value)?value:value?[value]:[]}
function hasEvidence(records:PaperEvidence[], fields:EvidenceField[]){return records.some(record=>fields.some(field=>itemsFor(record,field).length>0))}
function EvidenceGroup({records,fields}:{records:PaperEvidence[];fields:EvidenceField[]}){const entries=records.flatMap(record=>fields.flatMap(field=>itemsFor(record,field).map(item=>({record,field,item}))));if(!entries.length)return <p className="hint">The selected sources did not provide usable structured evidence for this part of the report.</p>;return <div className="claim-list">{entries.map(({record,field,item},index)=><article key={`${record.paper_id}-${field}-${index}`}><p className="claim-type">{sectionLabels[field]} · {record.study_type}</p><h3>{item.value}</h3><blockquote>{item.evidence_text}</blockquote><p className="provenance"><strong>{record.title}</strong> · <code>{record.paper_id}</code> · {pretty(item.source)}{item.section_heading?` · ${item.section_heading}`:""}{item.section_type?` (${pretty(item.section_type)})`:""}</p></article>)}</div>}

function AssessmentDetails({assessment}:{assessment:NonNullable<Analysis["result"]>["idea_assessment"]}){if(!assessment)return null;return <details className="assessment-details"><summary>Evidence behind the verdict</summary><IdGroup label="Supporting papers" ids={assessment.supporting_paper_ids}/><IdGroup label="Counterexample papers" ids={assessment.counterexample_paper_ids}/>{assessment.supporting_evidence.length>0&&<div><h3>Supporting evidence</h3>{assessment.supporting_evidence.map((item,index)=><GapEvidenceRow key={`support-${index}`} item={item}/>)}</div>}{assessment.counterexample_evidence.length>0&&<div><h3>Counterexample evidence</h3>{assessment.counterexample_evidence.map((item,index)=><GapEvidenceRow key={`counter-${index}`} item={item}/>)}</div>}<IdGroup label="Partial matches" ids={assessment.partial_match_paper_ids}/><IdGroup label="Potential matches" ids={assessment.potential_match_paper_ids}/>{Object.entries(assessment.matched_facets).length>0&&<div><h3>Matched facets</h3>{Object.entries(assessment.matched_facets).map(([id,facets])=><p key={id}><code>{id}</code>: {facets.join(", ")}</p>)}</div>}{assessment.coverage_notes.length>0&&<div><h3>Coverage notes</h3><ul>{assessment.coverage_notes.map(note=><li key={note}>{note}</li>)}</ul></div>}</details>}
function GapReport({gap}:{gap:Gap}){const verification=gap.verification;return <article className="gap-card"><div className="gap-heading"><StatusPill status={gap.final_label??verification?.label??"uncertain"}/><span>{pretty(gap.category)}</span></div><h3>{gap.title}</h3><p>{gap.description}</p><p><strong>Why it was suggested:</strong> {gap.rationale}</p><CitationIds ids={gap.supporting_paper_ids}/>{gap.landscape_basis.length>0&&<p><strong>Landscape basis:</strong> {gap.landscape_basis.map(item=>`${item.value} (${item.count}/${item.total} papers)`).join("; ")}</p>}{gap.supporting_evidence.length>0&&<details><summary>Supporting evidence</summary>{gap.supporting_evidence.map((item,index)=><GapEvidenceRow key={index} item={item}/>)}</details>}<details><summary>Verification and counterexamples</summary><p><strong>Status:</strong> {pretty(gap.verification_status)} · <strong>pattern:</strong> {pretty(gap.pattern_type)}</p>{verification&&<><p>{verification.reason}</p><IdGroup label="Potentially contradicting papers" ids={gap.potentially_contradicting_paper_ids}/><IdGroup label="Confirmed counterexamples" ids={gap.contradicting_paper_ids}/>{gap.potentially_contradicting_evidence.map((item,index)=><GapEvidenceRow key={`potential-${index}`} item={item}/>)}{gap.contradicting_evidence.map((item,index)=><GapEvidenceRow key={`confirmed-${index}`} item={item}/>)}{verification.coverage_notes.length>0&&<ul>{verification.coverage_notes.map(note=><li key={note}>{note}</li>)}</ul>}</>}{gap.verification_queries.length>0&&<><h4>Verification queries</h4><ul>{gap.verification_queries.map(query=><li key={query.query}>{query.query}</li>)}</ul></>}</details></article>}
function GapEvidenceRow({item}:{item:Gap["supporting_evidence"][number]}){return <div className="gap-evidence"><p><strong>{item.value}</strong> <span className="hint">({pretty(item.role)})</span></p><blockquote>{item.evidence_text}</blockquote><code>{item.paper_id}</code></div>}
function IdGroup({label,ids}:{label:string;ids:string[]}){return ids.length?<p><strong>{label}:</strong> {ids.map(id=><code key={id}>{id}</code>)}</p>:null}
function CitationIds({ids}:{ids:string[]}){return ids.length?<p className="citation-ids"><span>Sources</span>{ids.map(id=><code key={id}>{id}</code>)}</p>:null}
function inspectedSections(records:PaperEvidence[]){return [...new Set(records.flatMap(record=>record.coverage?.inspected_section_types??[]))]}
function PaperEvidenceCard({record}:{record:PaperEvidence}){const coverage=record.coverage;return <article className="evidence-card"><h3>{record.title}</h3><p><code>{record.paper_id}</code> · {pretty(record.study_type)}</p>{coverage&&<p><strong>Coverage:</strong> {pretty(coverage.source_level)} · full-text status: {pretty(coverage.full_text_status)}{coverage.truncated?" · truncated":""}</p>}{coverage?.inspected_section_types.length?<p><strong>Inspected sections:</strong> {coverage.inspected_section_types.map(pretty).join(", ")}</p>:null}{record.missing_fields.length>0&&<p><strong>Fields without extracted evidence:</strong> {record.missing_fields.map(pretty).join(", ")}.</p>}{Object.keys(sectionLabels).map(field=>itemsFor(record,field as EvidenceField)).flat().length===0&&<p className="hint">No structured claims were extracted from the available source.</p>}</article>}
function Papers({papers}:{papers:NonNullable<Analysis["result"]>["papers"]}){const[query,setQuery]=useState("");const[sort,setSort]=useState("relevance");const visible=[...papers].filter(p=>p.title.toLowerCase().includes(query.toLowerCase())).sort((a,b)=>sort==="year"?(b.publication_year??0)-(a.publication_year??0):(b.final_score??0)-(a.final_score??0));return <><div className="paper-tools"><label>Filter papers<input value={query} onChange={event=>setQuery(event.target.value)} placeholder="Search titles"/></label><label>Sort by<select value={sort} onChange={event=>setSort(event.target.value)}><option value="relevance">Relevance</option><option value="year">Newest</option></select></label></div><div className="paper-list">{visible.map(paper=><article className="paper" key={paper.id}><p className="paper-meta">{paper.publication_year??"Year unavailable"} · relevance {typeof paper.final_score==="number"?paper.final_score.toFixed(2):"not scored"}</p><h3>{paper.title}</h3><p>{paper.authors.join(", ")||"Authors unavailable"}</p>{paper.abstract&&<p>{paper.abstract.slice(0,560)}</p>}<div className="actions">{paper.url&&<a className="button secondary" href={paper.url} target="_blank" rel="noreferrer">Open paper</a>}{paper.doi&&<a className="button quiet" href={`https://doi.org/${paper.doi.replace("https://doi.org/","")}`} target="_blank" rel="noreferrer">View DOI</a>}</div><details><summary>Retrieval provenance</summary><pre>{JSON.stringify(paper.provenance,null,2)}</pre></details></article>)}</div></>}
