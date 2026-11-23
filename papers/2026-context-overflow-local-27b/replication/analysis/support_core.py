"""Allowlisted, unchanged presentation definitions; no private input loading."""
# Full source SHA256: b239457ceffe01aaf186e7a30d7ce00fddc3686b33c2e02537bfbe7e45e61bb8
import math
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

