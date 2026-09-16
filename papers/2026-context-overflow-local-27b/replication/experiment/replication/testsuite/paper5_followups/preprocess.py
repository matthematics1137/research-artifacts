#!/usr/bin/env python3
"""CPU-only LoCoMo classifier-cache preparation, only if C was selected.

Check machine availability first. Histories shared by questions are classified
once; each item gets a separate bound cache record for the unchanged selector.
No generation server, GPU inference, model download, or primary-file write.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import time

from legacy import algorithms
from workspace import Workspace, canonical, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        if not 1 <= args.threads <= 32:
            raise ValueError("threads must be from 1 to 32")
        work = Workspace("locomo")
        _, items = work.load()
        if work.expected_arms() != ("T", "C"):
            raise ValueError("LoCoMo C was not selected from the completed primary results")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        started = time.monotonic()
        with algorithms(work) as a:
            import torch
            import lingua_probs as L
            torch.set_num_threads(args.threads)
            compressor = L.load_compressor("cpu")
            cache = {}
            inputs = {str(work.path("WORKSPACE.json")): sha(work.path("WORKSPACE.json")),
                      str(work.path("SELECTION.json")): sha(work.path("SELECTION.json"))}
            model = Path(L.MODEL_PATH)
            paths = [p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".model", ".bin", ".safetensors")]
            inputs.update({str(p): sha(p) for p in paths})
            for item in items:
                history = a.C.full_history(item)
                digest = a.C.sha256_text(history)
                if digest not in cache:
                    cache[digest] = L.chunk_probs(compressor, history)
                record = {"question_id": item["question_id"], "history_sha256": digest,
                          "model_revision": L.MODEL_REVISION, "device": "cpu", "chunks": cache[digest]}
                payload = gzip.compress(canonical(record).encode(), mtime=0)
                work.write_once("lingua/" + item["question_id"] + ".json.gz", payload)
            from workspace import verify_inputs
            verify_inputs(inputs)
            receipt = {"schema": "paper5-followup-classifier-v1", "inputs": inputs, "histories": len(cache),
                       "items": len(items), "host": platform.node(), "device": "cpu", "threads": args.threads,
                       "seconds": time.monotonic() - started, "includes_model_load": True}
            work.write_json_once("preparation/classifier.json", receipt)
            print(json.dumps({"items": len(items), "unique_histories": len(cache), "device": "cpu"}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"LoCoMo classifier preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
