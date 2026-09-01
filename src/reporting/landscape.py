"""Concise CLI formatting for a LiteratureLandscape."""

from __future__ import annotations

from src.analysis.gap_candidates import is_concrete_entity
from src.models.landscape import LiteratureLandscape


def format_landscape(landscape: LiteratureLandscape) -> str:
    """Render a landscape without exposing its full serialized structure."""

    lines = [
        "Literature Landscape",
        "====================",
        "",
        f"Evidence papers analyzed: {landscape.total_papers}",
        "",
        "Evidence Coverage",
        "-----------------",
    ]
    for label, field_name in (
        ("Research objective", "research_objective"),
        ("Population / setting", "population_or_setting"),
        ("Methods", "methods"),
        ("Datasets", "datasets"),
        ("Sample size", "sample_size"),
        ("Baselines", "baselines"),
        ("Metrics", "evaluation_metrics"),
        ("Outcomes", "outcomes"),
        ("Constraints", "constraints"),
        ("Limitations", "limitations"),
        ("Future work", "future_work"),
    ):
        missing = landscape.missing_field_counts.get(field_name, 0)
        available = landscape.total_papers - missing
        lines.append(f"{label}: {available}/{landscape.total_papers}")

    for heading, dimension in (
        ("Problems", "problem"),
        ("Method Families", "method_family"),
        ("Datasets", "dataset"),
        ("Performance Metrics", "performance_metric"),
        ("Efficiency Metrics", "efficiency_metric"),
        ("Constraints", "constraint"),
    ):
        lines.extend(["", heading, "-" * len(heading)])
        values = [
            item for item in landscape.frequencies
            if item.dimension == dimension and is_concrete_entity(item.value)
        ][:10]
        lines.extend([f"{item.value}: {item.count}" for item in values] or ["None observed."])

    lines.extend(["", "Study Types", "-----------"])
    study_types = [item for item in landscape.frequencies if item.dimension == "study_type"]
    lines.extend([f"{item.value}: {item.count}" for item in study_types] or ["None observed."])

    lines.extend(["", "Dataset Settings", "----------------"])
    dataset_types = [
        item for item in landscape.frequencies
        if item.dimension == "dataset_type" and is_concrete_entity(item.value)
    ][:10]
    lines.extend([f"{item.value}: {item.count}" for item in dataset_types] or ["None observed."])

    lines.extend(["", "Frequent Combinations", "---------------------"])
    frequent = [item for item in landscape.combinations if item.count >= 2][:10]
    lines.extend([f"{' + '.join(item.dimensions.values())}: {item.count}" for item in frequent] or ["None observed."])

    lines.extend(["", "Sparse Observed Combinations", "----------------------------"])
    sparse = [item for item in landscape.combinations if item.count == 1][:10]
    lines.extend([f"{' + '.join(item.dimensions.values())}: {item.count}" for item in sparse] or ["None observed."])

    lines.extend(["", "Potential Conflicts", "-------------------"])
    if landscape.conflicts:
        lines.extend([f"{item.topic} ({', '.join(item.paper_ids)}): {item.status}" for item in landscape.conflicts])
    else:
        lines.append("None supported by sufficiently comparable evidence.")
    return "\n".join(lines)
