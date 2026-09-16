"""Allowlisted, unchanged presentation definitions; no private input loading."""
# Full source SHA256: 4f51a5686da1289ab965428a7b1b658e2738ab6dcd2605da1fd26fbd37b623b4
from reporting import ARM_TEX, EVIDENCE_METRICS, secondary_tag
from draft_support import pvalue
BUDGET_TEX = {"1024": "One", "4096": "Four", "None": "No", "none": "No"}


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def signed(x, digits=1):
    """Signed percentage points for text mode, with a typographic minus."""
    return f"{100 * x:+.{digits}f}".replace("-", "$-$")


def p_tex(p):
    if p is None:
        return "n/a"
    return pvalue(p)


def macros(claims: dict) -> list[str]:
    out = []
    used = set()

    def add(name, value):
        if name in used or not name.isascii() or not name.isalpha():
            raise ValueError(f"duplicate or invalid macro name: {name}")
        used.add(name)
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for key, cell in sorted(claims["cells"].items()):
        arm, budget = key.split("@")
        a, b = ARM_TEX.get(arm, arm), BUDGET_TEX.get(str(budget), str(budget))
        add(f"PfAcc{a}{b}", pct(cell["accuracy"]))
        add(f"PfAccLo{a}{b}", pct(cell["wilson95"][0]))
        add(f"PfAccHi{a}{b}", pct(cell["wilson95"][1]))
        add(f"PfN{a}{b}", cell["n"])
        add(f"PfCorrect{a}{b}", cell["correct"])
        if cell.get("delivered_prompt_tokens_mean") is not None:
            add(f"PfTok{a}{b}", f"{cell['delivered_prompt_tokens_mean']:,.0f}")
        if cell.get("completion_tokens_mean") is not None:
            add(f"PfCompTok{a}{b}", f"{cell['completion_tokens_mean']:,.0f}")
        if cell.get("generation_seconds_mean") is not None:
            add(f"PfSecs{a}{b}", f"{cell['generation_seconds_mean']:.1f}")
        add(f"PfCapHits{a}{b}", cell["cap_hits"])
        add(f"PfUnclosed{a}{b}", cell["unterminated_thinking"])
        add(f"PfMissing{a}{b}", cell["missing_or_failed_generations"])
        ja = cell.get("judge_agreement") or {}
        if ja.get("raw") is not None:
            add(f"PfJudgeAgree{a}{b}", pct(ja["raw"]))
        if ja.get("kappa") is not None:
            add(f"PfJudgeKappa{a}{b}", f"{ja['kappa']:.2f}")
        cv = cell.get("containment_vs_judge") or {}
        if cv.get("containment_accuracy") is not None:
            add(f"PfContAcc{a}{b}", pct(cv["containment_accuracy"]))
        if cv.get("agreement") is not None:
            add(f"PfContAgree{a}{b}", pct(cv["agreement"]))
        for k, v in (cell.get("diag_means") or {}).items():
            name = "".join(part.capitalize() for part in k.split("_"))
            if k.endswith("_seconds"):
                add(f"PfDiag{name}{a}{b}", f"{v:.2f}")
            elif k in {metric for metric, _, _ in EVIDENCE_METRICS} or k.endswith("_fraction"):
                add(f"PfDiag{name}{a}{b}", pct(v))
            else:
                add(f"PfDiag{name}{a}{b}", f"{v:,.1f}")
        for k, v in (cell.get("diag_counts") or {}).items():
            name = "".join(part.capitalize() for part in k.split("_"))
            add(f"PfDiagCount{name}{a}{b}", v)

    for arm, c in (claims["primary"].get("contrasts_vs_T_at_4096") or {}).items():
        a = ARM_TEX.get(arm, arm)
        add(f"PfDiff{a}vT", signed(c["difference"]))
        add(f"PfDiffLo{a}vT", signed(c["newcombe95"][0]))
        add(f"PfDiffHi{a}vT", signed(c["newcombe95"][1]))
        add(f"PfP{a}vT", p_tex(c["mcnemar_p"]))
        add(f"PfHolm{a}vT", p_tex(c["holm_p"]))
        add(f"PfRejected{a}vT", "rejected" if c["rejected_at_0.05"] else "not rejected")
        add(f"PfArmOnly{a}vT", c["arm_only"])
        add(f"PfRefOnly{a}vT", c["reference_only"])

    for name, c in (claims.get("secondary") or {}).items():
        if not isinstance(c, dict) or "difference" not in c:
            continue
        tag = secondary_tag(name)
        add(f"PfSec{tag}", signed(c["difference"]))
        add(f"PfSecLo{tag}", signed(c["newcombe95"][0]))
        add(f"PfSecHi{tag}", signed(c["newcombe95"][1]))
        add(f"PfSecP{tag}", p_tex(c["mcnemar_p"]))

    rank = (claims.get("secondary") or {}).get("ranking") or {}
    for b, order in rank.items():
        add(f"PfRank{BUDGET_TEX.get(str(b), str(b))}", ", ".join(order))
    og = (claims.get("secondary") or {}).get("oracle_gap_on_fit_subset") or {}
    if og:
        add("PfiveOFitN", og["n_fit"])
        for arm, c in (og.get("O_minus_arm") or {}).items():
            add(f"PfOracleGap{ARM_TEX.get(arm, arm)}", signed(c["difference"]))
    return out

