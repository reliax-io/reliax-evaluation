"""Regenerate the section 7.3 tables of PAPER.md and website/whitepaper.html
from results/expansion/tableshift.json and martingale_diag.json, so that no
number in either document is hand-typed.

Usage:
  .venv/bin/python eval/expansion/make_section7_tables.py            # rewrite both files in place
  .venv/bin/python eval/expansion/make_section7_tables.py --print    # print the blocks only
"""
import json
import pathlib
import re
import sys

BASE = pathlib.Path(__file__).resolve().parents[2]
E = BASE / "results" / "expansion"
R = json.load(open(E / "tableshift.json"))
D = json.load(open(E / "martingale_diag.json"))
ORDER = ["acsincome", "acsfoodstamps", "acsunemployment", "brfss_diabetes",
         "diabetes_readmission", "college_scorecard", "assistments", "anes"]
LABEL = {"acsincome": "ACS income (finance) · region", "acsfoodstamps": "ACS food stamps (public policy) · region",
         "acsunemployment": "ACS unemployment (labour) · education", "brfss_diabetes": "BRFSS diabetes (health) · race",
         "diabetes_readmission": "Hospital readmission (health) · admission source",
         "college_scorecard": "College Scorecard (education) · institution type",
         "assistments": "ASSISTments (education) · school",
         "anes": "ANES voting (civic) · region"}
TASKS = [t for t in ORDER if t in R["tasks"] and R["tasks"][t].get("seeds_ok")]


def rows_main():
    out = []
    for t in TASKS:
        S = R["tasks"][t]; I, O = S["id_test"], S["ood_test"]
        out.append([LABEL[t], f"{S['n']['cal']:,}", f"{I['accuracy']['mean']:.3f} / {O['accuracy']['mean']:.3f}",
                    f"{I['coverage']['0.05']['mean']:.3f}", f"{O['coverage']['0.05']['mean']:.3f}",
                    f"{I['review_rate']['mean']:.3f} / {O['review_rate']['mean']:.3f}",
                    f"{O['martingale']['alarm_rate']:.0%}",
                    f"{I['martingale_extra_id']['watch_rate']:.0%} / {I['martingale_extra_id']['alarm_rate']:.0%}",
                    f"{S['claimed_d']['mean']:.3f} / {O['realized_d']['mean']:.3f}"])
    return out


def rows_cal():
    out = []
    for t in TASKS:
        S = R["tasks"][t]; I, O = S["id_test"], S["ood_test"]
        out.append([LABEL[t].split(" ·")[0], f"{I['raw_ece']['mean']:.3f}", f"{I['va_ece']['mean']:.3f}",
                    f"{O['raw_ece']['mean']:.3f}", f"{O['va_ece']['mean']:.3f}", f"{S['claimed_d']['mean']:.3f}",
                    f"{I['realized_d']['mean']:.3f}", f"{O['realized_d']['mean']:.3f}"])
    return out


def rows_diag():
    out = []
    for t in TASKS:
        if t not in D:
            continue
        v = D[t]["variants"]
        out.append([LABEL[t].split(" ·")[0], f"{D[t]['n_cal']:,}", f"{D[t]['n_binary_features']} / {D[t]['n_features']}",
                    f"{v['frozen']['ks_p_uniform']:.3f}",
                    f"{v['frozen']['id_streams']['watch']:.0%} / {v['frozen']['id_streams']['alarm']:.0%}",
                    f"{v['no_scaler']['ks_p_uniform']:.3f}",
                    f"{v['no_scaler']['id_streams']['watch']:.0%} / {v['no_scaler']['id_streams']['alarm']:.0%}",
                    f"{v['frozen']['ood_streams']['alarm']:.0%} / {v['no_scaler']['ood_streams']['alarm']:.0%}"])
    return out


HEADS = {
    "main": ["task (domain) · shift", "cal n", "acc ID / OOD", "cov ID", "cov OOD", "REVIEW ID / OOD",
             "ALARM on OOD", "false WATCH / ALARM (100 ID streams)", "disbelief claimed / realised OOD"],
    "cal": ["task", "raw ECE ID", "Venn-Abers ECE ID", "raw ECE OOD", "Venn-Abers ECE OOD",
            "disbelief claimed", "realised ID", "realised OOD"],
    "diag": ["task", "cal n", "binary / all features", "KS p, shipped", "false WATCH / ALARM, shipped",
             "KS p, no scaler", "false WATCH / ALARM, no scaler", "OOD ALARM shipped / no scaler"],
}
ROWS = {"main": rows_main, "cal": rows_cal, "diag": rows_diag}


def md(kind):
    h = HEADS[kind]
    return "| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n" + \
        "\n".join("| " + " | ".join(r) + " |" for r in ROWS[kind]())


def html(kind):
    h = HEADS[kind]
    body = "\n".join("      <tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in ROWS[kind]())
    return ('  <div class="tbl-scroll"><table>\n    <thead><tr>' + "".join(f"<th>{c}</th>" for c in h)
            + "</tr></thead>\n    <tbody>\n" + body + "\n    </tbody>\n  </table></div>")


def replace_md(text, kind):
    header = "| " + " | ".join(HEADS[kind]) + " |"
    pat = re.compile(re.escape(header) + r".*?\n(?=\n)", re.S)
    assert pat.search(text), f"markdown table {kind} not found"
    return pat.sub(lambda m: md(kind) + "\n", text, count=1)


def replace_html(text, kind):
    header = "".join(f"<th>{c}</th>" for c in HEADS[kind])
    pat = re.compile(r'  <div class="tbl-scroll"><table>\n    <thead><tr>' + re.escape(header) + r"</tr></thead>.*?</table></div>", re.S)
    assert pat.search(text), f"html table {kind} not found"
    return pat.sub(lambda m: html(kind), text, count=1)


def main():
    if "--print" in sys.argv:
        for k in ("main", "cal", "diag"):
            print(md(k), "\n")
        return
    p = BASE / "PAPER.md"; t = p.read_text()
    for k in ("main", "cal", "diag"):
        t = replace_md(t, k)
    p.write_text(t)
    w = BASE.parent / "website" / "whitepaper.html"
    if w.exists():
        t = w.read_text()
        for k in ("main", "cal", "diag"):
            t = replace_html(t, k)
        w.write_text(t)
    print("tables rewritten for", TASKS)


if __name__ == "__main__":
    main()
