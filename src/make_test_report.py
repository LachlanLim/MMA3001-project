"""
Turn the automated test results into a readable web page.

pytest can save its results as an XML file, which is complete but hard
to read. This script turns that file into a single HTML page with:

- a pass/fail summary at the top
- each test explained in plain English (taken from its docstring)
- for the invalid-label tests, each bad label and why it was rejected

It is the test report that a reviewer opens, at
``results/test_reports/test_summary.html``.

## Usage

Run the tests first, so the XML file exists, then build the page:

```text
$ pytest --junitxml=results/test_reports/all_tests.xml
..................                                         [100%]
18 passed in 0.37s

$ python src/make_test_report.py
Saved report to results/test_reports/test_summary.html
```

Open the HTML file in any web browser.
"""

import ast
import html
import platform
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from validate_annotations import validate_annotation


XML_PATH = Path("results/test_reports/all_tests.xml")
OUTPUT_PATH = Path("results/test_reports/test_summary.html")
TESTS_DIR = Path("tests")

# Friendly titles for each test module
SUITE_TITLES = {
    "tests.test_dataset": (
        "Dataset configuration",
        "Checks that data.yaml, the class list and the train/valid/test "
        "folders are set up correctly.",
    ),
    "tests.test_annotations": (
        "Annotation validation",
        "Checks the YOLO label validator against known good and bad "
        "annotations, then runs it over every label file in the dataset.",
    ),
    "src.validate_annotations": (
        "Documentation examples",
        "Runs the examples written in the code documentation, to prove "
        "they give the output shown.",
    ),
}


# --------------------------------------------------
# Collect test docstrings and expected rejection reasons
# --------------------------------------------------

def read_docstrings():
    """
    Read the first docstring line of every test function.

    Returns:
        dict[str, str]: Maps ``"tests.module.function"`` to its summary line.
    """
    docstrings = {}

    for path in TESTS_DIR.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = f"tests.{path.stem}"

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                if doc:
                    docstrings[f"{module}.{node.name}"] = doc.splitlines()[0]

    return docstrings


def rejection_reason(annotation):
    """
    Explain why the validator rejects an annotation.

    Args:
        annotation (str): YOLO annotation line used as a test input.

    Returns:
        str: The validator's error message, or a note if it was accepted.
    """
    try:
        validate_annotation(annotation)
    except ValueError as exc:
        return str(exc)
    return "Accepted (unexpected)"


def humanise(name):
    """
    Turn a test function name into a readable title.

    Args:
        name (str): Function name, e.g. ``"test_dataset_has_images"``.

    Returns:
        str: Title, e.g. ``"Dataset has images"``.
    """
    words = name.removeprefix("test_").replace("_", " ")
    return words[:1].upper() + words[1:]


# --------------------------------------------------
# Read the JUnit XML
# --------------------------------------------------

def load_results():
    """
    Parse the JUnit XML report written by pytest.

    Returns:
        tuple[dict, list[dict]]: Overall summary (counts, duration and
        timestamp) and one dict per test case with its module, function,
        parameter, description, status, duration and failure message.

    Raises:
        FileNotFoundError: If the XML report does not exist.
    """
    root = ET.parse(XML_PATH).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    docstrings = read_docstrings()

    cases = []
    for case in suite.iter("testcase"):
        module = case.get("classname")
        full_name = case.get("name")
        match = re.match(r"([\w.]+)(?:\[(.*)\])?$", full_name)
        function, param = match.group(1), match.group(2)

        # Doctests are named after the documented function,
        # e.g. "src.validate_annotations.validate_annotation"
        is_doctest = "." in function
        function = function.rsplit(".", 1)[-1]

        status, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            element = case.find(tag)
            if element is not None:
                status = {"failure": "failed"}.get(tag, tag)
                message = element.get("message") or element.text or ""
                break

        cases.append({
            "module": module,
            "function": function,
            "param": param,
            "title": (
                f"Examples for {function}()" if is_doctest else humanise(function)
            ),
            "description": (
                "Check that each example in the function's documentation "
                "gives the result shown."
                if is_doctest
                else docstrings.get(f"{module}.{function}", "")
            ),
            "time": float(case.get("time", 0)),
            "status": status,
            "message": message,
        })

    summary = {
        "tests": int(suite.get("tests", len(cases))),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "time": float(suite.get("time", 0)),
        "timestamp": suite.get("timestamp", ""),
        "hostname": suite.get("hostname", ""),
    }
    summary["passed"] = (
        summary["tests"] - summary["failures"]
        - summary["errors"] - summary["skipped"]
    )

    return summary, cases


# --------------------------------------------------
# Render HTML
# --------------------------------------------------

def format_time(seconds):
    """Format a duration in seconds as milliseconds or seconds."""
    if seconds < 0.001:
        return "&lt;1 ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def status_badge(status):
    """Return a coloured HTML badge for a test status."""
    labels = {"passed": "Passed", "failed": "Failed",
              "error": "Error", "skipped": "Skipped"}
    return f'<span class="badge {status}">{labels[status]}</span>'


def render_param_rows(group):
    """Render a parametrised test as a sub-table of its inputs."""
    rows = []
    for case in group:
        shown = case["param"] if case["param"] else "(empty line)"
        reason = (
            rejection_reason(case["param"])
            if case["function"] == "test_invalid_annotations" else ""
        )
        rows.append(
            f"<tr><td><code>{html.escape(shown)}</code></td>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{status_badge(case['status'])}</td></tr>"
        )
    return (
        '<table class="params"><thead><tr><th>Input annotation</th>'
        "<th>Rejected because</th><th>Result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_test(group):
    """
    Render one test function as a card.

    Args:
        group (list[dict]): Every case of the function (more than one if
            it is parametrised).

    Returns:
        str: HTML for the card.
    """
    first = group[0]
    statuses = {case["status"] for case in group}
    overall = next(
        (s for s in ("failed", "error", "skipped") if s in statuses),
        "passed",
    )
    total_time = sum(case["time"] for case in group)
    count = (
        f'<span class="count">{len(group)} cases</span>'
        if len(group) > 1 else ""
    )

    failures = "".join(
        f'<pre class="failure">{html.escape(case["message"])}</pre>'
        for case in group if case["message"]
    )
    params = render_param_rows(group) if first["param"] is not None else ""

    return f"""
    <div class="test {overall}">
      <div class="test-head">
        <div>
          <h3>{html.escape(first["title"])} {count}</h3>
          <p class="desc">{html.escape(first["description"])}</p>
          <code class="fn">{first["function"]}</code>
        </div>
        <div class="test-meta">
          {status_badge(overall)}
          <span class="time">{format_time(total_time)}</span>
        </div>
      </div>
      {params}
      {failures}
    </div>"""


def render_suite(module, cases):
    """Render all tests from one test module as a page section."""
    title, blurb = SUITE_TITLES.get(module, (module, ""))
    passed = sum(case["status"] == "passed" for case in cases)

    # Group parametrised cases under their test function
    groups = {}
    for case in cases:
        groups.setdefault(case["function"], []).append(case)

    tests_html = "".join(render_test(group) for group in groups.values())

    return f"""
  <section class="suite">
    <div class="suite-head">
      <h2>{html.escape(title)}</h2>
      <span class="suite-score">{passed} / {len(cases)} passed</span>
    </div>
    <p class="blurb">{html.escape(blurb)}</p>
    {tests_html}
  </section>"""


def render_page(summary, cases):
    """
    Build the full HTML page.

    Args:
        summary (dict): Overall results from ``load_results``.
        cases (list[dict]): Test cases from ``load_results``.

    Returns:
        str: Complete HTML document.
    """
    modules = {}
    for case in cases:
        modules.setdefault(case["module"], []).append(case)

    failed = summary["failures"] + summary["errors"]
    pass_rate = 100 * summary["passed"] / summary["tests"] if summary["tests"] else 0
    verdict = "All tests passed" if failed == 0 else f"{failed} test(s) failed"
    verdict_class = "ok" if failed == 0 else "bad"

    try:
        run_time = datetime.fromisoformat(summary["timestamp"])
        run_time = run_time.strftime("%d %B %Y, %I:%M %p")
    except ValueError:
        run_time = summary["timestamp"]

    suites_html = "".join(
        render_suite(module, module_cases)
        for module, module_cases in modules.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test Report</title>
<style>
  :root {{
    --bg: #f6f7f9; --card: #ffffff; --text: #1d2330; --muted: #5f6b7a;
    --border: #e2e6ec; --code-bg: #f1f3f6;
    --pass: #1a7f4b; --pass-bg: #e3f5eb;
    --fail: #c0392b; --fail-bg: #fbe6e3;
    --skip: #a46a00; --skip-bg: #fdf1d8;
    --accent: #2f5bd3;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #12151b; --card: #1b2029; --text: #e6e9ef; --muted: #9aa4b2;
      --border: #2b323e; --code-bg: #232a35;
      --pass: #4fd18b; --pass-bg: #163626;
      --fail: #ff7a6b; --fail-bg: #3d1d1a;
      --skip: #f0b84a; --skip-bg: #3a2d12;
      --accent: #7c9cff;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif;
  }}
  main {{ max-width: 960px; margin: 0 auto; padding: 40px 16px 64px; }}
  header h1 {{ margin: 0 0 4px; font-size: 28px; }}
  header .sub {{ color: var(--muted); margin: 0; }}
  .verdict {{
    display: inline-flex; align-items: center; gap: 8px; margin-top: 16px;
    padding: 6px 14px; border-radius: 999px; font-weight: 600;
  }}
  .verdict.ok {{ background: var(--pass-bg); color: var(--pass); }}
  .verdict.bad {{ background: var(--fail-bg); color: var(--fail); }}

  .tiles {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin: 28px 0 12px;
  }}
  .tile {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }}
  .tile .label {{ color: var(--muted); font-size: 13px; }}
  .tile .value {{ font-size: 26px; font-weight: 700; }}
  .tile.pass .value {{ color: var(--pass); }}
  .tile.fail .value {{ color: var(--fail); }}
  .tile.skip .value {{ color: var(--skip); }}

  .bar {{
    height: 8px; border-radius: 4px; background: var(--fail-bg);
    overflow: hidden; margin-bottom: 8px;
  }}
  .bar > div {{ height: 100%; background: var(--pass); }}
  .env {{ color: var(--muted); font-size: 13px; margin: 0 0 36px; }}

  .suite {{ margin-bottom: 40px; }}
  .suite-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 2px solid var(--border); padding-bottom: 6px;
  }}
  .suite-head h2 {{ margin: 0; font-size: 20px; }}
  .suite-score {{ color: var(--muted); font-weight: 600; }}
  .blurb {{ color: var(--muted); margin: 8px 0 16px; }}

  .test {{
    background: var(--card); border: 1px solid var(--border);
    border-left: 4px solid var(--pass); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 10px;
  }}
  .test.failed, .test.error {{ border-left-color: var(--fail); }}
  .test.skipped {{ border-left-color: var(--skip); }}
  .test-head {{ display: flex; justify-content: space-between; gap: 16px; }}
  .test h3 {{ margin: 0; font-size: 16px; }}
  .count {{
    font-size: 12px; font-weight: 500; color: var(--muted);
    margin-left: 6px;
  }}
  .desc {{ margin: 2px 0 6px; }}
  code {{
    font-family: Consolas, "Cascadia Mono", monospace; font-size: 13px;
    background: var(--code-bg); padding: 1px 6px; border-radius: 4px;
  }}
  .fn {{ color: var(--muted); font-size: 12px; }}
  .test-meta {{
    display: flex; flex-direction: column; align-items: flex-end;
    gap: 4px; flex-shrink: 0;
  }}
  .time {{ color: var(--muted); font-size: 13px; }}

  .badge {{
    font-size: 12px; font-weight: 600; padding: 2px 10px;
    border-radius: 999px; white-space: nowrap;
  }}
  .badge.passed {{ background: var(--pass-bg); color: var(--pass); }}
  .badge.failed, .badge.error {{ background: var(--fail-bg); color: var(--fail); }}
  .badge.skipped {{ background: var(--skip-bg); color: var(--skip); }}

  .params {{
    width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px;
  }}
  .params th {{
    text-align: left; color: var(--muted); font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: .04em;
    border-bottom: 1px solid var(--border); padding: 6px 8px;
  }}
  .params td {{ border-bottom: 1px solid var(--border); padding: 7px 8px; }}
  .params tr:last-child td {{ border-bottom: none; }}
  .params td:last-child, .params th:last-child {{ text-align: right; }}

  .failure {{
    background: var(--fail-bg); color: var(--fail); padding: 10px 12px;
    border-radius: 6px; white-space: pre-wrap; font-size: 13px;
    margin: 12px 0 0;
  }}
  footer {{ color: var(--muted); font-size: 13px; text-align: center; }}

  @media (max-width: 600px) {{
    .test-head {{ flex-direction: column; }}
    .test-meta {{ flex-direction: row; align-items: center; }}
    .params th:nth-child(2), .params td:nth-child(2) {{ display: none; }}
  }}
  @media print {{
    body {{ background: #fff; }}
    .test {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Automated Test Report</h1>
    <p class="sub">Pork packaging defect detection &middot; dataset and annotation checks</p>
    <div class="verdict {verdict_class}">{verdict}</div>
  </header>

  <div class="tiles">
    <div class="tile"><div class="label">Total tests</div><div class="value">{summary["tests"]}</div></div>
    <div class="tile pass"><div class="label">Passed</div><div class="value">{summary["passed"]}</div></div>
    <div class="tile fail"><div class="label">Failed</div><div class="value">{failed}</div></div>
    <div class="tile skip"><div class="label">Skipped</div><div class="value">{summary["skipped"]}</div></div>
    <div class="tile"><div class="label">Duration</div><div class="value">{format_time(summary["time"])}</div></div>
  </div>

  <div class="bar"><div style="width: {pass_rate:.1f}%"></div></div>
  <p class="env">
    {pass_rate:.0f}% pass rate &middot; Run {html.escape(run_time)}
    &middot; Python {platform.python_version()} on {platform.system()} {platform.release()}
  </p>

  {suites_html}

  <footer>Generated from {XML_PATH.as_posix()} by src/make_test_report.py</footer>
</main>
</body>
</html>
"""


def main():
    """Read the XML report and save the HTML summary."""
    summary, cases = load_results()
    OUTPUT_PATH.write_text(render_page(summary, cases), encoding="utf-8")
    print(f"Saved report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
