"""Public response sanitizers for provider-backed analysis data."""

from __future__ import annotations

from copy import deepcopy


def public_analysis_result(result: dict[str, object]) -> dict[str, object]:
    """Remove provider exception strings while retaining useful coverage counts."""
    public = deepcopy(result)
    retrieval = public.pop("retrieval_failures", [])
    extraction = public.pop("extraction_failures", [])
    retrieval_count = len(retrieval) if isinstance(retrieval, list) else 0
    extraction_count = len(extraction) if isinstance(extraction, list) else 0
    papers = public.get("papers")
    evidence = public.get("evidence")
    work_metrics = public.get("work_metrics")
    selected_count = len(papers) if isinstance(papers, list) else 0
    successful_count = len(evidence) if isinstance(evidence, list) else 0
    requested_count = successful_count + extraction_count
    if isinstance(work_metrics, dict):
        configured_requested = work_metrics.get("papers_requested_for_initial_extraction")
        configured_successful = work_metrics.get("successful_initial_evidence_records")
        configured_failed = work_metrics.get("failed_initial_extractions")
        if isinstance(configured_requested, int) and configured_requested >= 0:
            requested_count = configured_requested
        if isinstance(configured_successful, int) and configured_successful >= 0:
            successful_count = configured_successful
        if isinstance(configured_failed, int) and configured_failed >= 0:
            extraction_count = configured_failed
    public["failure_summary"] = {
        "retrieval": retrieval_count,
        "extraction": extraction_count,
    }
    public["extraction_coverage"] = {
        "selected_for_report": selected_count,
        "requested_for_extraction": requested_count,
        "successful_evidence_records": successful_count,
        "failed_extractions": extraction_count,
        "not_requested_for_extraction": max(0, selected_count - requested_count),
        "partial": extraction_count > 0,
    }
    messages: list[str] = []
    if retrieval_count:
        messages.append(
            f"{retrieval_count} literature search route(s) were unavailable; other completed routes were retained."
        )
    if extraction_count:
        messages.append(
            f"Structured extraction failed for {extraction_count} requested paper(s). "
            "Fields from those papers could not be evaluated; this is partial coverage, "
            "not evidence that the fields were unreported."
        )
    public["coverage_messages"] = messages
    if isinstance(evidence, list):
        for record in evidence:
            if not isinstance(record, dict):
                continue
            coverage = record.get("coverage")
            if isinstance(coverage, dict):
                status = coverage.get("full_text_status")
                coverage["notices"] = _coverage_notices(status, bool(coverage.get("truncated")))
    assessment = public.get("idea_assessment")
    if isinstance(assessment, dict):
        assessment.pop("failures", None)
    gaps = public.get("gaps")
    if isinstance(gaps, list):
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            verification = gap.get("verification")
            if isinstance(verification, dict):
                verification.pop("failures", None)
    return public


def _coverage_notices(status: object, truncated: bool) -> list[str]:
    messages: list[str] = []
    if status == "unavailable":
        messages.append("Open full text was unavailable; available abstract or metadata was used.")
    elif status in {"fetch_failed", "parse_failed"}:
        messages.append("Open full text could not be inspected; available abstract or metadata was used.")
    if truncated:
        messages.append("Part of this document was truncated to remain within analysis limits.")
    return messages


def public_job_error(_message: str | None) -> str:
    return "The analysis could not be completed. Any reserved credit was returned. Please try again later."
