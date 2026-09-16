"""Small, CPU-only manuscript helpers. These checks never authorize a release."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def bad(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number: {value}")
        return parsed
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique,
                      parse_constant=bad, parse_float=finite_float)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{label}: out of range")
    return value


def integer(value, label, low=0, high=None):
    if type(value) is not int:
        raise ValueError(f"{label}: expected an integer")
    return number(value, label, low, high)


def probability(value, label="probability"):
    return number(value, label, 0, 1)


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def pvalue(value):
    probability(value)
    # Do not round a nonzero p-value to zero, or across the .05 threshold.
    if value == 0:
        return "0"
    if value < .001:
        return r"\ensuremath{" + f"{value:.2e}".replace("e", r"\times10^{") + "}}"
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    if float(rendered) == .05 and value != .05:
        return repr(value)
    return rendered


def command(name, body):
    if not name.isascii() or not name.isalpha():
        raise ValueError(f"invalid TeX macro name: {name}")
    return rf"\newcommand{{\{name}}}{{{body}}}" + "\n"


def table(headers, rows, caption, label, alignment=None):
    alignment = alignment or ("l" + "r" * (len(headers) - 1))
    return ("\n\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
            + rf"\begin{{tabular}}{{{alignment}}}\toprule" + "\n"
            + " & ".join(headers) + r"\\ \midrule" + "\n"
            + "\n".join(" & ".join(map(str, row)) + r"\\" for row in rows)
            + "\n\\bottomrule\\end{tabular}\n"
            + rf"\caption{{{caption}}}\label{{{label}}}" + "\n\\end{table}\n")


def atomic_text(path, text, check=False):
    path = Path(path)
    if check:
        if not path.is_file() or path.read_text() != text:
            raise ValueError(f"missing or stale generated file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    os.replace(temporary, path)


def preparation_notice():
    """Private preparation builds are labelled on every page, including title pages."""
    return r"""
\setlength{\headheight}{15pt}
\makeatletter
\def\ps@preparation{%
  \def\@oddhead{\hfil\small PREPARATION DRAFT --- RESULTS PENDING --- NOT FOR RELEASE\hfil}%
  \let\@evenhead\@oddhead
  \def\@oddfoot{\hfil\thepage\hfil}%
  \let\@evenfoot\@oddfoot}
\let\ps@plain\ps@preparation
\pagestyle{preparation}
\makeatother
"""
