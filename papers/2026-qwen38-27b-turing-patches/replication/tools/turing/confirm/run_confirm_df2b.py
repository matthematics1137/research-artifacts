#!/usr/bin/env python3
"""Paper 8 confirmation run, follow-up step x-p8-confirm-df2b (Amendment 3 of
research/prereg/2026-11-04-p8-confirmation-speed.md).

Amendment 2's session ran, but its DFlash2 arm failed at its first start ("Qwen3.5 config: missing component
dflash2"). NInferno serves a local GGUF from its own text-only conversion (<dir>/.ninferno-cache/), and a start never
attaches a local GGUF's separate drafters; NInferno's published 2080 Ti command reaches them through its hub flow,
which converts with them. This step runs Amendment 2's session again with one change: the DFlash2 arm serves
NInferno's conversion of the registered UD-IQ2_XXS GGUF with the registry's DFlash2 drafter attached, made by NInferno's
own converter (--with-drafters) before the step and checked here by size and SHA-256. Its other arguments are the
registered ones. run_confirm_df2.py and plan-df2.json are unchanged (checked against their SHA-256 and imported).

  python3 run_confirm_df2b.py --plan plan.json --df2 plan-df2b.json --out DIR [--dry-run]

Rows, journal and logs go to DIR as in run_confirm.py. Stdlib only. Nothing is deleted.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_confirm as RC  # noqa: E402
import run_confirm_df2 as D  # noqa: E402
import run_confirm_fit as F  # noqa: E402

L = RC.L


def artifact_path(df2, lab=L):
    return df2["dflash2_artifact"]["path"].replace("{L}", lab)


def dflash2_args(plan, df2, lab=L):
    """The registered arguments with the GGUF replaced by the converted artifact; nothing else may change."""
    registered = plan["conditional"]["nif-dflash2"]["config_args"]
    if registered[0] != plan["models"]["iq2xxs"]:
        raise ValueError("the registered DFlash2 arm no longer starts from the UD-IQ2_XXS GGUF")
    return [artifact_path(df2, lab)] + registered[1:]


class DF2bRun(D.DF2Run):
    def __init__(self, plan, df2, out, dry):
        super().__init__(plan, df2, out, dry)
        self.plan["configs"] = dict(self.plan["configs"],
                                    **{"nif-xxs-df2": {"drafted": True, "args": dflash2_args(plan, df2)}})

    # -- the DFlash2 arm's context: NInferno's own choice at an empty context, as its published command runs it, and
    #    at 28K the smallest context that holds the filled prompt (Amendment 1's R2), since 32768 does not fit
    def server_cmd(self, c, extra=()):
        cmd, env = super().server_cmd(c, extra)
        if c.get("config") == "nif-xxs-df2":
            i = cmd.index("--max-context")
            ctx = self.df2["dflash2_max_context"]["28k" if c["fill"] else "0"]
            cmd = cmd[:i] + (["--max-context", str(ctx)] if ctx else []) + cmd[i + 2:]
        return cmd, env

    def check_artifact(self, lab=L):
        """The converted artifact, by size and SHA-256, as recorded when it was made; otherwise the arm is dropped."""
        a = self.df2["dflash2_artifact"]
        path = artifact_path(self.df2, lab)
        size = os.path.getsize(path) if os.path.isfile(path) else None
        digest = D.file_sha256(path) if size == a["bytes"] else None
        ok = digest == a["sha256"]
        self.log("artifact", path=path, bytes=size, sha256=digest, ok=ok)
        if not ok:
            for c in self.plan["conditions"]:
                if c["config"] == "nif-xxs-df2" and c["id"] not in self.dropped:
                    self.dropped[c["id"]] = "the converted DFlash2 artifact is missing or differs from the recorded one"
                    self.log("drop", condition=c["id"], reason=self.dropped[c["id"]])
        return ok

    def check_drafter(self, lab=L):
        ok = super().check_drafter(lab)
        return self.check_artifact(lab) and ok

    def prepare(self):
        """As registered, then: if neither DFlash2 condition survived its first start, the session has no purpose
        (Amendment 2 already ran the anchors), so every condition is dropped before round 1."""
        super().prepare()
        df2 = [c["id"] for c in self.plan["conditions"] if c["config"] == "nif-xxs-df2"]
        if df2 and all(cid in self.dropped for cid in df2):
            for c in self.plan["conditions"]:
                if c["id"] not in self.dropped:
                    self.dropped[c["id"]] = "the DFlash2 arm did not start, so the session stops before round 1"
                    self.log("drop", condition=c["id"], reason=self.dropped[c["id"]])

    def main(self):
        # run_confirm_df2's session event keeps Amendment 2's label; this event names the step and the DFlash2 arm
        self.log("amendment", plan="Amendment 3: x-p8-confirm-df2b",
                 dflash2_args=self.plan["configs"]["nif-xxs-df2"]["args"])
        super().main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--df2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with open(args.df2) as f:
        df2 = json.load(f)
    bad = [k for k, ok in F.frozen_ok(df2, os.path.dirname(os.path.abspath(args.plan))).items() if not ok]
    if bad:
        raise SystemExit(f"refusing: {', '.join(bad)} differ from the registration")
    plan = RC.load_plan(args.plan, L)
    run = DF2bRun(plan, df2, args.out, args.dry_run)
    if args.dry_run:
        run.baseline_mib = 0
        for c in run.plan["conditions"]:
            tag = f"confirm-{c['id']}-r1"
            if run.plan["engines"][c["engine"]]["kind"] == "census":
                cmd, _ = run.census_cmd(c, tag)
            else:
                cmd, _ = run.server_cmd(c)
            print(tag, " ".join(cmd))
        print("drafter", D.drafter_path(df2["drafter"]))
        print("artifact", artifact_path(df2))
        return
    run.main()


if __name__ == "__main__":
    sys.exit(main())
