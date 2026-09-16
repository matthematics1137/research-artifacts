"""Batch tokenizer verification in the existing serving environment, CPU only.

The analysis Python has SciPy/matplotlib but not tokenizers. This narrow
subprocess bridge uses the already-installed exl3-env without installing
packages or mixing its site-packages into the analysis interpreter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


class TokenCounts:
    def __init__(self, tokenizer_file, interpreter):
        self.tokenizer_file = Path(tokenizer_file).resolve()
        self.interpreter = Path(interpreter).absolute()
        self.cache = {}
        self.proof = None

    def prepare(self, texts):
        unique = {}
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("tokenizer input is not text")
            key = text_sha(text)
            if key in unique and unique[key] != text:
                raise ValueError("tokenizer text digest collision")
            unique[key] = text
        payload = "".join(json.dumps({"sha256": key, "text": value}, ensure_ascii=False) + "\n" for key, value in unique.items())
        source = Path(__file__).resolve()
        expected_files = {str(self.tokenizer_file): sha(self.tokenizer_file), str(source): sha(source)}
        env = os.environ.copy()
        env.update(TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            run = subprocess.run([str(self.interpreter), "-I", "-B", str(source), "--worker", "--tokenizer", str(self.tokenizer_file)],
                                 input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("CPU tokenizer verification exceeded its 600-second limit") from exc
        if run.returncode:
            raise ValueError("CPU tokenizer worker failed: " + run.stderr[-1500:])
        lines = [json.loads(line) for line in run.stdout.splitlines()]
        if not lines or lines[0].get("schema") != "p5-tokenizer-audit-worker-v1":
            raise ValueError("tokenizer worker identity is missing")
        header = lines[0]
        if header.get("add_special_tokens") is not False or header.get("cpu_threads") != 1:
            raise ValueError("tokenizer worker used different counting settings")
        if any(header["files"].get(path) != value or sha(path) != value for path, value in expected_files.items()):
            raise ValueError("tokenizer or verification source changed")
        if any(sha(path) != value for path, value in header["files"].items()):
            raise ValueError("tokenizer runtime changed during verification")
        counts = {}
        for row in lines[1:]:
            key = row.get("sha256")
            if set(row) != {"sha256", "tokens"} or key not in unique or key in counts or type(row["tokens"]) is not int or row["tokens"] < 0:
                raise ValueError("invalid, duplicate or foreign tokenizer count")
            counts[key] = row["tokens"]
        if counts.keys() != unique.keys():
            raise ValueError("tokenizer count batch is incomplete")
        self.cache = {key: (unique[key], value) for key, value in counts.items()}
        self.proof = {**header, "unique_texts": len(counts), "request_sha256": text_sha(payload),
                      "counts_sha256": text_sha(json.dumps(counts, sort_keys=True))}
        return self.proof

    def __call__(self, text):
        key = text_sha(text)
        if key not in self.cache or self.cache[key][0] != text:
            raise ValueError("text was not included in the verified tokenizer batch")
        return self.cache[key][1]


def worker(tokenizer_file):
    import tokenizers
    import tokenizers.tokenizers as native
    path = Path(tokenizer_file).resolve()
    files = [path, Path(__file__).resolve(), Path(sys.executable).resolve(), Path(tokenizers.__file__).resolve(),
             Path(native.__file__).resolve()]
    before = {str(file): sha(file) for file in files}
    tok = tokenizers.Tokenizer.from_file(str(path))
    print(json.dumps({"schema": "p5-tokenizer-audit-worker-v1", "tokenizers_version": tokenizers.__version__,
                      "python_version": sys.version.split()[0], "add_special_tokens": False, "cpu_threads": 1, "files": before}))
    for line in sys.stdin:
        row = json.loads(line)
        if set(row) != {"sha256", "text"} or not isinstance(row["text"], str) or text_sha(row["text"]) != row["sha256"]:
            raise ValueError("invalid tokenizer input record")
        print(json.dumps({"sha256": row["sha256"], "tokens": len(tok.encode(row["text"], add_special_tokens=False).ids)}))
    if any(sha(file) != value for file, value in before.items()):
        raise ValueError("tokenizer runtime/input changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    try:
        worker(args.tokenizer)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"Tokenizer verification stopped: {exc}\n")
