"""Scoped reuse of the unchanged primary P5 algorithms, with private output paths.

No server starts on import. This bridge is single-threaded and restores the
caller's module/path state. CLI work runs in its own process, never in the
process that owns the primary experiment.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

NAMES = ("p5common", "p5run", "build_prompts", "summarize", "judge", "run_grid", "lingua_probs")


@contextmanager
def algorithms(workspace):
    saved_path = list(sys.path)
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in NAMES}
    source = workspace.root / "testsuite/paper5"
    def load(name):
        spec = importlib.util.spec_from_file_location(name, source / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    try:
        for name in NAMES:
            sys.modules.pop(name, None)
        common = load("p5common")
        common.WORK = str(workspace.base)
        runner = load("p5run")
        prompts = load("build_prompts")
        summaries = load("summarize")
        judge = load("judge")
        grid = load("run_grid")
        prompts.MODEL = {"cross-model": "qwen3-8b", "locomo": "qwen38-exl3"}[workspace.leg]
        # Primary PAUSE remains a shared read-only operator stop signal. Every
        # writable path (prompts, caches, journals, logs) points to this leg.
        for value in (runner.LOGS, prompts.OUT, summaries.JOURNAL, judge.JOURNALS,
                      grid.PROMPTS, grid.JOURNALS, grid.CAP_FILE):
            if not Path(value).is_relative_to(workspace.base):
                raise ValueError("legacy bridge did not isolate an output path")
        yield SimpleNamespace(C=common, runner=runner, BP=prompts, SM=summaries, judge=judge, grid=grid)
    finally:
        for name, old in saved.items():
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        sys.path[:] = saved_path
