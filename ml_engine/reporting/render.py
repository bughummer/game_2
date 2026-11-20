"""Render a ``Report`` to self-contained HTML. No templates, no network, no dependencies.

The HTML is deliberately plain and inline-styled so it is a valid standalone file and a clean
input for a PDF renderer (WeasyPrint, when installed). Every value shown comes from the report
document, which copied it from a deterministic artifact — the renderer invents nothing.
"""

from html import escape

from ml_engine.contracts.report import Report, ReportSection

_STYLE = "".join(
    [
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;",
        "color:#1a1a1a;margin:2rem;}",
        "h1{font-size:1.6rem;margin:0 0 .25rem;}",
        "h2{font-size:1.15rem;margin:1.5rem 0 .4rem;border-bottom:1px solid #ddd;",
        "padding-bottom:.2rem;}",
        ".meta{color:#666;font-size:.85rem;margin-bottom:1rem;}",
        "table{border-collapse:collapse;width:100%;margin:.3rem 0 .8rem;font-size:.85rem;}",
        "th,td{text-align:left;padding:.3rem .5rem;border-bottom:1px solid #eee;",
        "vertical-align:top;}",
        "th{background:#f6f6f6;font-weight:600;}",
        ".summary{color:#333;margin:.2rem 0 .4rem;}",
        ".note{margin:.15rem 0;font-size:.85rem;}",
        ".narrative{background:#f0f6ff;border-left:3px solid #4a90d9;padding:.6rem .8rem;",
        "margin:1rem 0;}",
        ".sources{color:#888;font-size:.75rem;margin-top:2rem;}",
    ]
)


def render_html(report: Report) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{escape(report.title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{escape(report.title)}</h1>",
        f"<div class='meta'>Experiment {escape(report.experiment_id)} · "
        f"generated {report.generated_at.isoformat()}</div>",
    ]
    if report.narrative:
        source = f" (source: {escape(report.narrative_source)})" if report.narrative_source else ""
        parts.append(f"<div class='narrative'>{escape(report.narrative)}{source}</div>")
    for section in report.sections:
        parts.append(_render_section(section))
    if report.source_artifacts:
        parts.append(
            "<div class='sources'>Assembled from: "
            + escape(", ".join(report.source_artifacts))
            + "</div>"
        )
    parts.append("</body></html>")
    return "".join(parts)


def _render_section(section: ReportSection) -> str:
    out = [f"<h2>{escape(section.title)}</h2>"]
    if section.summary:
        out.append(f"<div class='summary'>{escape(section.summary)}</div>")
    if section.rows:
        out.append(_render_table(section.rows))
    for note in section.notes:
        out.append(f"<div class='note'>{escape(note)}</div>")
    return "".join(out)


def _render_table(rows: list[dict[str, str]]) -> str:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
