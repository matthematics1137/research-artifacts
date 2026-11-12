#!/usr/bin/env python3
"""Paper 8 confirmation run, follow-up step x-p8-confirm-df2 (Amendment 2 of
research/prereg/2026-11-04-p8-confirmation-speed.md).

One session, three rotated rounds, of the conditions in plan-df2.json: the registered conditional arm NInferno +
DFlash2 (nif-xxs-df2-0 and -28k, built as run_confirm.py's conditional() builds it) beside same-session anchors. It
then meets the standard every other competitor met, faster or slower in every round of one session, instead of the
single end-of-session run that the registered conditional gives it. It reuses run_confirm.py and plan.json unchanged
(they are checked against their registered SHA-256 and imported, never edited) and run_confirm_fit.py's port chooser.
- **the drafter:** z-lab/Qwen3.8-27B-DFlash2 at NInferno's pinned revision must be in the lab's Hugging Face cache
  with the registered size and SHA-256, or both DFlash2 conditions are dropped and the anchors run alone;
- **not run:** the 240 W conditional (the desktop has no sudoers rule for it), the buun GSM8K check (done in the
  fit session) and every other competitor (measured in the registered and fit sessions).

  python3 run_confirm_df2.py --plan plan.json --df2 plan-df2.json --out DIR [--dry-run]

Rows, journal and logs go to DIR as in run_confirm.py. Stdlib only. Nothing is deleted.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_confirm as RC  # noqa: E402
import run_confirm_fit as F  # noqa: E402

L = RC.L
DF2_IDS = ("nif-xxs-df2-0", "nif-xxs-df2-28k")


def file_sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def drafter_path(d, lab=L):
    """The hub cache's snapshot path, where NInferno looks with HF_HUB_OFFLINE=1."""
    return os.path.join(lab, "cache/huggingface/hub", "models--" + d["repo"].replace("/", "--"),
                        "snapshots", d["revision"], d["file"])


def df2_conditions(plan, df2):
    """The session's conditions in plan-df2's order, with plan-df2's priorities. The DFlash2 arm takes the id, engine,
    config and arguments that run_confirm.py's conditional() gives it."""
    d = plan["conditional"]["nif-dflash2"]
    registered = {c["id"]: c for c in plan["conditions"]}
    configs = dict(plan["configs"], **{"nif-xxs-df2": {"drafted": True, "args": d["config_args"]}})
    out = []
    for item in df2["conditions"]:
        cid = item["id"]
        if cid in registered:
            c = dict(registered[cid], priority=item["priority"])
        elif cid in DF2_IDS:
            fill = plan["fill_long"] if cid.endswith("-28k") else 0
            if fill not in d["fills"]:
                raise ValueError(f"{cid}: fill {fill} is not a registered DFlash2 fill")
            c = {"id": cid, "engine": "ninferno", "config": "nif-xxs-df2", "fill": fill, "priority": item["priority"]}
        else:
            raise ValueError(f"{cid}: not a registered condition")
        out.append(c)
    return out, configs


class DF2Run(RC.Run):
    def __init__(self, plan, df2, out, dry):
        super().__init__(plan, out, dry)
        self.df2 = df2
        self.next_port = df2["port_range"][0]
        conditions, configs = df2_conditions(plan, df2)
        self.plan = dict(plan, rounds=df2["rounds"], cap_hours=df2["cap_hours"], configs=configs,
                         conditions=conditions)

    # -- ports: run_confirm_fit's chooser over this step's own range (Amendment 1's fix)
    def server_cmd(self, c, extra=()):
        e = self.plan["engines"][c["engine"]]
        lo, hi = self.df2["port_range"]
        port = self.next_port if self.dry else F.choose_port(lo, hi, self.next_port, self.df2["port_wait_s"])
        if port is None:
            raise RuntimeError("no bindable port in the range after the wait")
        self.next_port = lo + (port - lo + 1) % (hi - lo + 1)
        self.base = f"http://localhost:{port}"
        args = RC.subst(self.plan["configs"][c["config"]]["args"] + e["common"] + list(extra), {"port": port})
        env = dict(os.environ, XDG_CACHE_HOME=os.path.join(L, "cache"))
        env.update(e["env"])
        if "HOME" in e["env"] and not self.dry:  # a dry run touches nothing
            os.makedirs(e["env"]["HOME"], exist_ok=True)
        return [e["bin"]] + args, env

    # -- the registered download, by size and SHA-256 (the registered rule only asked that its folder exist)
    def check_drafter(self, lab=L):
        d = self.df2["drafter"]
        path = drafter_path(d, lab)
        size = os.path.getsize(path) if os.path.isfile(path) else None
        digest = file_sha256(path) if size == d["bytes"] else None
        ok = digest == d["sha256"]
        self.log("drafter", path=path, bytes=size, sha256=digest, ok=ok)
        if not ok:
            for c in self.plan["conditions"]:
                if c["config"] == "nif-xxs-df2":
                    self.dropped[c["id"]] = "the DFlash2 drafter is missing or differs from the registered download"
                    self.log("drop", condition=c["id"], reason=self.dropped[c["id"]])
        return ok

    def main(self):
        g = RC.gpu_state()
        self.baseline_mib = int(g.get("mem_used_mib", 400)) if g else 400
        self.log("session", plan="Amendment 2: x-p8-confirm-df2", frozen=F.frozen_ok(self.df2), gpu=g,
                 load=os.getloadavg(), baseline_mib=self.baseline_mib,
                 conditions=[c["id"] for c in self.plan["conditions"]])
        self.preflight()
        self.check_drafter()
        self.make_filler()
        self.prepare()
        kept = [c["id"] for c in self.plan["conditions"] if c["id"] not in self.dropped]
        for rnd in range(1, self.plan["rounds"] + 1):
            if rnd > 1:
                kept = self.cut_to_cap(rnd, [k for k in kept if k not in self.dropped], 0)
            ordered = [c for c in self.plan["conditions"] if c["id"] in kept]
            if not ordered:
                break
            n = len(ordered)
            ordered = [ordered[(i + rnd - 1) % n] for i in range(n)]  # ab_speed.sh's rotation, as registered
            self.log("round", round=rnd, conditions=[c["id"] for c in ordered])
            for c in ordered:
                if c["id"] not in self.dropped:
                    self.run_condition(c, rnd)
        self.log("done", hours=round((time.time() - self.t0) / 3600, 2), dropped=self.dropped)


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
    run = DF2Run(plan, df2, args.out, args.dry_run)
    if args.dry_run:
        run.baseline_mib = 0
        for c in run.plan["conditions"]:
            tag = f"confirm-{c['id']}-r1"
            if run.plan["engines"][c["engine"]]["kind"] == "census":
                cmd, _ = run.census_cmd(c, tag)
            else:
                cmd, _ = run.server_cmd(c)
            print(tag, " ".join(cmd))
        print("drafter", drafter_path(df2["drafter"]))
        return
    run.main()


if __name__ == "__main__":
    sys.exit(main())
