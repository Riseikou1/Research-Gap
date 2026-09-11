"""Private analysis creation, history, retrieval, export, and deletion routes."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from src.api.dependencies import principal_for
from src.api.models import AnalysisCreated, AnalysisDetail, AnalysisSummary, CreateAnalysisRequest
from src.auth import network_rate_key
from src.persistence.models import CreditBillingDecision, NewAnalysis
from src.persistence.security import QuotaError, SecurityRepository
from src.api.safety import public_analysis_result
from src.extraction.evidence import canonical_evidence_key
from src.retrieval.deduplication import normalize_doi, normalize_title

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisCreated, status_code=status.HTTP_201_CREATED)
def create_analysis(payload: CreateAnalysisRequest, request: Request) -> AnalysisCreated:
    components = request.app.state.components
    mode = payload.mode or "full"
    principal = principal_for(request, force_guest=payload.mode is not None)
    if payload.paper_limit > components.settings.openalex.max_candidates:
        raise HTTPException(status_code=422, detail="paper_limit exceeds the configured candidate limit.")
    if components.repository.count_active_for_owner(principal.kind, principal.principal_id) >= components.settings.web.max_active_analyses_per_principal:
        raise HTTPException(status_code=429, detail="Too many analyses are already active. Try again after one finishes.")
    if mode == "full" and principal.kind == "guest":
        raise HTTPException(status_code=401, detail="Sign in with a verified email to run a Full Gap Analysis.")
    if mode == "full" and principal.kind == "user" and not principal.email_verified:
        raise HTTPException(status_code=403, detail="Verify your email before spending a Full Gap Analysis credit.")
    if mode == "quick" and (payload.full_text or payload.decomposer != "deterministic" or payload.query_generator != "deterministic"):
        raise HTTPException(status_code=422, detail="Quick Search uses deterministic retrieval and cannot enable full text.")

    settings = components.settings.web
    if mode == "quick":
        limit = settings.guest_quick_limit if principal.kind == "guest" else settings.user_quick_daily_limit
        keys = [f"{principal.kind}:{principal.principal_id}"]
        if principal.kind == "guest":
            address = request.client.host if request.client else "unknown"
            keys.append("network:" + network_rate_key(address, settings.guest_cookie_secret))
        for key in keys:
            if not components.security.record_rate_event(key, "quick_search", limit=limit):
                raise HTTPException(status_code=429, detail="Quick Search limit reached. Try again after the rolling 24-hour window.")

    analysis_id = str(uuid4())
    reservation_id = None
    billing_decision = _credit_billing_decision(
        mode,
        principal.kind,
        principal.principal_id,
        components.security,
    )
    if billing_decision == "paid_credit":
        try:
            reservation_id = components.security.reserve_credit(principal.principal_id, analysis_id)
        except QuotaError as exc:
            raise HTTPException(status_code=402, detail=str(exc))
    configuration = components.service.configuration_snapshot(
        decomposer="deterministic" if mode == "quick" else payload.decomposer,
        query_generator="deterministic" if mode == "quick" else payload.query_generator,
        paper_limit=payload.paper_limit, full_text=payload.full_text and mode == "full", mode=mode,
    )
    configuration["credit_billing_decision"] = billing_decision
    try:
        record = components.repository.create(NewAnalysis(
            analysis_id=analysis_id, research_idea=payload.research_idea,
            decomposer="deterministic" if mode == "quick" else payload.decomposer,
            query_generator="deterministic" if mode == "quick" else payload.query_generator,
            paper_limit=payload.paper_limit, configuration=configuration,
            owner_kind=principal.kind, owner_id=principal.principal_id, mode=mode,
            reservation_id=reservation_id,
        ))
    except Exception:
        if reservation_id:
            components.security.release_credit(analysis_id)
        raise
    try:
        components.runner.submit(analysis_id)
    except Exception:
        components.repository.mark_failed(analysis_id, "Analysis worker is unavailable.")
        if reservation_id:
            components.security.release_credit(analysis_id)
        raise HTTPException(status_code=503, detail="Analysis worker is unavailable.")
    return AnalysisCreated(analysis_id=record.analysis_id, status=record.status)


@router.get("", response_model=list[AnalysisSummary])
def list_analyses(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> list[AnalysisSummary]:
    principal = principal_for(request)
    records = request.app.state.components.repository.list_for_owner(principal.kind, principal.principal_id, limit)
    return [AnalysisSummary.from_record(record) for record in records]


def _owned_record(request: Request, analysis_id: str):
    principal = principal_for(request)
    record = request.app.state.components.repository.get_for_owner(analysis_id, principal.kind, principal.principal_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return principal, record


@router.get("/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: str, request: Request) -> AnalysisDetail:
    _, record = _owned_record(request, analysis_id)
    return AnalysisDetail.from_record(record)


@router.get("/{analysis_id}/export")
def export_analysis(analysis_id: str, request: Request,
                    format: str = Query(default="json", pattern="^(json|markdown)$")) -> Response:
    principal, record = _owned_record(request, analysis_id)
    if record.status != "completed" or record.result is None:
        raise HTTPException(status_code=409, detail="Only completed analyses can be exported.")
    if not request.app.state.components.security.record_rate_event(
        f"{principal.kind}:{principal.principal_id}", "export", limit=30,
    ):
        raise HTTPException(status_code=429, detail="Export rate limit reached. Try again later.")
    if format == "json":
        return Response(json.dumps(public_analysis_result(record.result), ensure_ascii=False, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="research-gap-{analysis_id}.json"'})
    return Response(_markdown_report(record.research_idea, record.mode, record.result),
                    media_type="text/markdown; charset=utf-8", headers={
                        "Content-Disposition": f'attachment; filename="research-gap-{analysis_id}.md"'})


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: str, request: Request) -> None:
    principal, record = _owned_record(request, analysis_id)
    components = request.app.state.components
    if not record.terminal and not components.runner.cancel(analysis_id):
        raise HTTPException(status_code=409, detail="A running analysis cannot be deleted until it finishes.")
    if record.status == "pending":
        components.repository.mark_failed(analysis_id, "Analysis cancelled before it started.")
        if record.reservation_id:
            components.security.release_credit(analysis_id)
    if not components.repository.delete_for_owner(analysis_id, principal.kind, principal.principal_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")


def _credit_billing_decision(
    mode: str,
    principal_kind: str,
    principal_id: str,
    security: SecurityRepository,
) -> CreditBillingDecision:
    """Choose ledger behavior from server-owned account state only."""

    if mode == "full" and principal_kind == "user":
        if security.administrator_credit_exempt(principal_id):
            return "administrator_credit_exempt"
        return "paid_credit"
    if mode == "quick" and principal_kind == "guest":
        return "guest_quick_search"
    return "credit_not_applicable"


def _markdown_report(idea: str, mode: str, result: dict[str, object]) -> str:
    public = public_analysis_result(result)
    assessment = public.get("idea_assessment") or {}
    label = assessment.get("label", "not performed") if isinstance(assessment, dict) else "not performed"
    rationale = assessment.get("rationale") if isinstance(assessment, dict) else None
    lines = ["# Research GAP report", "", f"**Idea:** {idea}", f"**Mode:** {mode}",
             f"**Assessment:** {str(label).replace('_', ' ')}"]
    if rationale:
        lines.extend(["", "## Executive summary", "", str(rationale)])

    evidence = public.get("evidence", [])
    papers = public.get("papers", [])
    paper_titles = {
        str(paper.get("id")): str(paper.get("title") or "Paper record")
        for paper in papers if isinstance(papers, list) and isinstance(paper, dict)
    }
    if isinstance(evidence, list) and evidence:
        for heading, fields in (
            ("Research objectives", ("research_objective",)),
            ("Populations and settings", ("population_or_setting",)),
            ("Methods", ("method_or_intervention",)),
            ("Datasets and data modalities", ("datasets", "data_or_modality")),
            ("Sample sizes", ("sample_size",)),
            ("Comparisons and baselines", ("comparison_or_baseline",)),
            ("Evaluation metrics", ("evaluation_metrics",)),
            ("Main findings", ("main_findings",)),
            ("Limitations and future work", ("limitations", "future_work")),
        ):
            _markdown_evidence_section(lines, heading, evidence, paper_titles, *fields)

    provenance_index = _evidence_provenance_index(evidence)

    gaps = public.get("gaps", [])
    if mode == "full" and isinstance(gaps, list) and gaps:
        lines.extend(["", "## Candidate research gaps", ""])
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            lines.append(f"### {gap.get('title') or 'Candidate gap'}")
            lines.extend(["", str(gap.get("description") or ""), "", str(gap.get("rationale") or "")])
            paper_ids = gap.get("supporting_paper_ids") or []
            if isinstance(paper_ids, list) and paper_ids:
                citations = [f"{paper_titles.get(str(item), 'Paper record')} ({item})" for item in paper_ids]
                lines.extend(["", "Supporting papers: " + ", ".join(citations)])
            supporting = gap.get("supporting_evidence") or []
            if isinstance(supporting, list) and supporting:
                lines.extend(["", "Supporting evidence:"])
                _markdown_gap_evidence(lines, supporting, paper_titles, provenance_index)
            verification = gap.get("verification") or {}
            if isinstance(verification, dict) and verification.get("reason"):
                lines.extend(["", "Verification: " + str(verification["reason"])])
                verification_evidence = verification.get("evidence") or []
                if isinstance(verification_evidence, list) and verification_evidence:
                    lines.extend(["", "Verification evidence:"])
                    _markdown_gap_evidence(
                        lines, verification_evidence, paper_titles, provenance_index
                    )

    extraction = public.get("extraction_coverage") or {}
    lines.extend(["", "## Full-text and evidence coverage", "",
                  f"Full text requested: {'yes' if public.get('full_text_requested') else 'no'}."])
    if isinstance(extraction, dict):
        lines.append(
            "Retrieved candidates: " + str(public.get("candidate_count", 0))
            + "; selected for reporting: " + str(extraction.get("selected_for_report", 0))
            + "; requested for structured extraction: " + str(extraction.get("requested_for_extraction", 0))
            + "; successful evidence records: " + str(extraction.get("successful_evidence_records", 0))
            + "; failed extractions: " + str(extraction.get("failed_extractions", 0)) + "."
        )
        if "accounted_papers" in extraction:
            lines.append(
                "Final outcomes — full text: " + str(extraction.get("full_text_successes", 0))
                + "; abstract: " + str(extraction.get("abstract_successes", 0))
                + "; abstract fallback: " + str(extraction.get("abstract_fallback_successes", 0))
                + "; metadata only: " + str(extraction.get("metadata_only_successes", 0))
                + "; final failures: " + str(extraction.get("final_failures", 0))
                + "; accounted: " + str(extraction.get("accounted_papers", 0)) + "."
            )
    attempts = public.get("full_text_attempt_summary") or {}
    if isinstance(attempts, dict) and attempts:
        lines.append(
            "Attempt diagnostics — full-text attempts: " + str(attempts.get("full_text_attempts", 0))
            + "; fetch failures: " + str(attempts.get("fetch_failures", 0))
            + "; parse failures: " + str(attempts.get("parse_failures", 0))
            + "; truncations: " + str(attempts.get("truncations", 0))
            + "; successful full-text extraction calls: "
            + str(attempts.get("full_text_extraction_successes", 0))
            + "; model/schema/evidence-validation final failures: "
            + str(attempts.get("model_schema_evidence_validation_failures", 0))
            + "; provider final failures: "
            + str(attempts.get("provider_failures", 0)) + "."
        )

    paper_coverage = public.get("paper_coverage") or []
    if isinstance(paper_coverage, list) and paper_coverage:
        lines.extend(["", "## Per-paper coverage", ""])
        for item in paper_coverage:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or "paper ID unavailable")
            title = str(item.get("title") or paper_titles.get(paper_id, "Paper record"))
            state = "success" if item.get("final_state") == "success" else "final failure"
            lines.append(
                f"- **{title}** ({paper_id}) — {state}; final evidence: "
                f"{str(item.get('final_evidence_level', 'none')).replace('_', ' ')}; "
                f"full text: {str(item.get('full_text_status', 'not_attempted')).replace('_', ' ')}"
                + (f" ({str(item['full_text_source_format']).upper()})" if item.get("full_text_source_format") else "")
                + "."
            )
            sections = item.get("inspected_section_types") or []
            if isinstance(sections, list) and sections:
                lines.append("  Full-text sections supplied to the extraction attempt: " + ", ".join(str(value).replace("_", " ") for value in sections) + ".")
            if item.get("fallback_explanation"):
                lines.append("  " + str(item["fallback_explanation"]))

    paper_records = [
        paper for paper in papers
        if isinstance(paper, dict)
    ] if isinstance(papers, list) else []
    if paper_records:
        lines.extend(["", "## Relevant papers", ""])
        seen_papers: set[tuple[str, str]] = set()
        for paper in paper_records:
            doi = normalize_doi(str(paper.get("doi") or ""))
            identity = (
                "doi", doi
            ) if doi else (
                normalize_title(str(paper.get("title") or "")),
                str(paper.get("publication_year") or ""),
            )
            if identity in seen_papers:
                continue
            seen_papers.add(identity)
            lines.append(f"- {paper.get('title', 'Untitled')} ({paper.get('publication_year') or 'year unavailable'})")
    lines.extend(["", "## Coverage limitations", "",
                  "This bounded analysis helps investigate possible gaps; it does not prove global novelty or replace a systematic review.", ""])
    return "\n".join(lines)


def _markdown_evidence_section(lines: list[str], heading: str,
                               evidence: list[object], paper_titles: dict[str, str], *fields: str) -> None:
    values: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for record in evidence:
        if not isinstance(record, dict):
            continue
        for field in fields:
            raw = record.get(field)
            items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
            for item in items:
                if isinstance(item, dict) and item.get("value"):
                    paper_id = str(record.get("paper_id", "paper ID unavailable"))
                    key = (paper_id, field, canonical_evidence_key(str(item["value"])))
                    if key in seen:
                        continue
                    seen.add(key)
                    values.extend([
                        f"- **{item['value']}** — {paper_titles.get(paper_id, 'Paper record')} ({paper_id})",
                        f"  > {str(item.get('evidence_text') or '').strip()}",
                        "  " + _markdown_provenance(item),
                    ])
    if values:
        lines.extend(["", f"## {heading}", "", *values])


def _markdown_provenance(item: dict[str, object]) -> str:
    source = str(item.get("source") or "unknown")
    if source != "full_text":
        return f"Evidence source: {source.replace('_', ' ')}."
    details = ["Evidence source: full text"]
    if item.get("section_heading"):
        details.append(f"heading: {item['section_heading']}")
    if item.get("section_type"):
        details.append(f"section type: {str(item['section_type']).replace('_', ' ')}")
    if item.get("section_id"):
        details.append(f"section ID: {item['section_id']}")
    return "; ".join(details) + "."


def _evidence_provenance_index(evidence: object) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    if not isinstance(evidence, list):
        return result
    fields = (
        "research_objective", "population_or_setting", "method_or_intervention",
        "comparison_or_baseline", "data_or_modality", "datasets", "sample_size",
        "evaluation_metrics", "main_findings", "constraints", "limitations", "future_work",
    )
    for record in evidence:
        if not isinstance(record, dict):
            continue
        paper_id = str(record.get("paper_id") or "")
        for field in fields:
            raw = record.get(field)
            items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
            for item in items:
                if isinstance(item, dict) and item.get("evidence_text"):
                    result[(paper_id, canonical_evidence_key(str(item["evidence_text"])))] = item
    return result


def _markdown_gap_evidence(
    lines: list[str],
    items: list[object],
    paper_titles: dict[str, str],
    provenance_index: dict[tuple[str, str], dict[str, object]],
) -> None:
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict) or not item.get("evidence_text"):
            continue
        paper_id = str(item.get("paper_id") or "paper ID unavailable")
        evidence_text = str(item["evidence_text"])
        key = (paper_id, canonical_evidence_key(evidence_text))
        if key in seen:
            continue
        seen.add(key)
        lines.extend([
            f"- **{item.get('value') or 'Supporting claim'}** — {paper_titles.get(paper_id, 'Paper record')} ({paper_id})",
            f"  > {evidence_text.strip()}",
            "  " + _markdown_provenance(provenance_index.get(key) or item),
        ])
