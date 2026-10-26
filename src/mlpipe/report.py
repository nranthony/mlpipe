"""Agent reports (ADR-0010): ad-hoc HTML pages built from stored artifacts.

Read side only. A report script finds artifacts with `mlpipe show --json`, loads
them from the printed store paths, builds plotly figures and calls `write_report`.
It never writes to `store/` or `manifests/`, and every page lists the artifact
hashes it read, so two reports can be checked for comparability (AGENTS.md).

The page is a fragment (title, style, body content, no html/head/body tags): it
opens as-is in a browser and is also a valid claude.ai Artifact page, light and
dark. plotly.js is inlined once, so the page works offline.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

STYLE = """
/* layout: one reading column; facts and sources as quiet tables, charts full width */
:root {
  --bg: #f7f9fb; --surface: #ffffff; --fg: #17212b; --muted: #5b6b7b;
  --rule: #dfe5eb; --accent: #1f5f8b;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #10161c; --surface: #161e26; --fg: #e3e9ef; --muted: #93a3b3;
  --rule: #2a3540; --accent: #6fb1e0; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #10161c; --surface: #161e26; --fg: #e3e9ef; --muted: #93a3b3;
  --rule: #2a3540; --accent: #6fb1e0; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 16px/1.55 var(--body); }
.page { max-width: 1100px; margin: 0 auto; padding-block: 32px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 28px; }
.eyebrow { font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); }
h1 { font: 700 clamp(26px, 4vw, 36px)/1.15 var(--display); margin: 8px 0 0; text-wrap: balance; }
h2 { font: 600 20px/1.3 var(--display); margin: 0 0 4px; text-wrap: balance; }
.lede { max-width: 65ch; color: var(--muted); margin: 10px 0 0; }
section { display: grid; gap: 8px; min-width: 0; }
.note { max-width: 65ch; margin: 0; color: var(--muted); }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 8px; min-width: 0; overflow-x: auto; }
.scroll { overflow-x: auto; min-width: 0; }
table { border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 14px 6px 0; border-bottom: 1px solid var(--rule);
  vertical-align: top; }
th { color: var(--muted); font-weight: 500; }
code, .mono { font-family: var(--mono); font-size: 13px; }
"""

# Re-colour plotly text from the page tokens, now and whenever the theme flips.
THEME_JS = """
<script>
(function () {
  function paint() {
    var css = getComputedStyle(document.documentElement);
    var fg = css.getPropertyValue('--fg').trim(), rule = css.getPropertyValue('--rule').trim();
    document.querySelectorAll('.js-plotly-plot').forEach(function (el) {
      var upd = {'font.color': fg, 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'};
      Object.keys(el.layout || {}).forEach(function (k) {
        if (/^[xy]axis\\d*$/.test(k)) { upd[k + '.gridcolor'] = rule; upd[k + '.zerolinecolor'] = rule; }
      });
      if (window.Plotly) { Plotly.relayout(el, upd); }
    });
  }
  window.addEventListener('load', paint);
  try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paint); } catch (e) {}
  new MutationObserver(paint).observe(document.documentElement,
    {attributes: true, attributeFilter: ['data-theme']});
})();
</script>
"""

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700'
         '&family=JetBrains+Mono:wght@400;600&family=Source+Sans+3:wght@400;600&display=swap">')


def table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """A plain HTML table from dicts (first row's keys unless `columns` given)."""
    if not rows:
        return "<p class='note'>No rows.</p>"
    cols = columns or list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def write_report(
    path: Path | str,
    title: str,
    sections: list[dict[str, Any]],
    sources: dict[str, str],
    lede: str = "",
    eyebrow: str = "mlpipe report",
) -> Path:
    """Write the page. Each section is {"heading", optional "note", and "figure"
    (plotly) and/or "html" (e.g. from `table`)}. `sources` maps what was read
    (artifact key, plan hash) to its full hash, and is printed at the end."""
    parts, first = [], True
    for i, s in enumerate(sections):
        body = []
        if s.get("note"):
            body.append(f"<p class='note'>{html.escape(s['note'])}</p>")
        if s.get("figure") is not None:
            fig = s["figure"]
            fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_family="Source Sans 3, system-ui, sans-serif",
                              margin={"l": 50, "r": 20, "t": 40, "b": 40})
            # fixed div ids keep a re-render byte-identical
            body.append("<div class='chart'>" + fig.to_html(
                include_plotlyjs=first, full_html=False, div_id=f"fig{i}",
                # no PNG download: a published page cannot hand the viewer a file
                config={"displaylogo": False, "responsive": True,
                        "modeBarButtonsToRemove": ["toImage"]}) + "</div>")
            first = False
        if s.get("html"):
            body.append(s["html"])
        parts.append(f"<section><h2>{html.escape(s['heading'])}</h2>{''.join(body)}</section>")
    sources_rows = [{"input": k, "hash": v} for k, v in sources.items()]
    page = (
        f"<title>{html.escape(title)}</title>\n"
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        f"{FONTS}\n<style>{STYLE}</style>\n"
        f"<main class='page'><header><div class='eyebrow'>{html.escape(eyebrow)}</div>"
        f"<h1>{html.escape(title)}</h1>"
        + (f"<p class='lede'>{html.escape(lede)}</p>" if lede else "")
        + "</header>" + "".join(parts)
        + "<section><h2>Sources</h2><p class='note'>Artifacts this report read, by content hash."
        "</p>" + table(sources_rows) + "</section></main>\n" + THEME_JS
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
