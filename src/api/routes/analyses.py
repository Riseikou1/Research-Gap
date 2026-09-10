"""Private analysis creation, history, retrieval, export, and deletion routes."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from src.api.dependencies import principal_for
from src.api.models import AnalysisCreated, AnalysisDetail, AnalysisSummary, CreateAnalysisRequest
from src.auth import network_rate_key
from src.persistence.models import NewAnalysis
from src.persistence.security import QuotaError
from src.api.safety import public_analysis_result

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
    if mode == "full" and principal.kind == "user":
        try:
            reservation_id = components.security.reserve_credit(principal.principal_id, analysis_id)
        except QuotaError as exc:
            raise HTTPException(status_code=402, detail=str(exc))
    configuration = components.service.configuration_snapshot(
        decomposer="deterministic" if mode == "quick" else payload.decomposer,
        query_generator="deterministic" if mode == "quick" else payload.query_generator,
        paper_limit=payload.paper_limit, full_text=payload.full_text and mode == "full", mode=mode,
    )
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
        components.security.release_credit(analysis_id)
    if not components.repository.delete_for_owner(analysis_id, principal.kind, principal.principal_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")


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
    if isinstance(evidence, list) and evidence:
        _markdown_evidence_section(lines, "What the literature already studies well", evidence, "research_objective")
        _markdown_evidence_section(lines, "Common methods", evidence, "method_or_intervention")
        _markdown_evidence_section(lines, "Main findings", evidence, "main_findings")
        _markdown_evidence_section(lines, "Important limitations and future work", evidence, "limitations", "future_work")

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
                lines.extend(["", "Supporting papers: " + ", ".join(map(str, paper_ids))])
            verification = gap.get("verification") or {}
            if isinstance(verification, dict) and verification.get("reason"):
                lines.extend(["", "Verification: " + str(verification["reason"])])

    landscape = public.get("landscape") or {}
    coverage = landscape.get("source_coverage") or {} if isinstance(landscape, dict) else {}
    levels = coverage.get("source_levels") or {} if isinstance(coverage, dict) else {}
    outcomes = coverage.get("full_text_outcomes") or {} if isinstance(coverage, dict) else {}
    lines.extend(["", "## Full-text and evidence coverage", "",
                  f"Full text requested: {'yes' if public.get('full_text_requested') else 'no'}.",
                  f"Selected evidence records: {len(evidence) if isinstance(evidence, list) else 0}."])
    if isinstance(levels, dict):
        lines.append(
            f"Full text: {levels.get('full_text', 0)}; abstract fallback: {levels.get('abstract', 0)}; metadata only: {levels.get('metadata_only', 0)}."
        )
    if isinstance(outcomes, dict):
        lines.append(
            f"Unavailable: {outcomes.get('unavailable', 0)}; fetch failed: {outcomes.get('fetch_failed', 0)}; parse failed: {outcomes.get('parse_failed', 0)}."
        )

    lines.extend(["", "## Relevant papers", ""])
    papers = public.get("papers", [])
    for paper in papers if isinstance(papers, list) else []:
        if isinstance(paper, dict):
            lines.append(f"- {paper.get('title', 'Untitled')} ({paper.get('publication_year') or 'year unavailable'})")
    lines.extend(["", "## Coverage limitations", "",
                  "This bounded analysis helps investigate possible gaps; it does not prove global novelty or replace a systematic review.", ""])
    return "\n".join(lines)


def _markdown_evidence_section(lines: list[str], heading: str,
                               evidence: list[object], *fields: str) -> None:
    values: list[str] = []
    for record in evidence:
        if not isinstance(record, dict):
            continue
        for field in fields:
            raw = record.get(field)
            items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
            for item in items:
                if isinstance(item, dict) and item.get("value"):
                    values.append(f"- {item['value']} — {record.get('paper_id', 'paper ID unavailable')}")
    if values:
        lines.extend(["", f"## {heading}", "", *values])
