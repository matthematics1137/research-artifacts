#!/usr/bin/env python3
"""Verify P5's exact upstream EXL3 wheel without installing or executing it.

Reading the 298 MB archive and hashing its >1 GB uncompressed native member
is a substantial I/O job: check live candidate-machine availability first.
Only --download permits a network fetch; existing files are never overwritten.
No model, GPU, package installation, import from the wheel or native build runs.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile

NAME = "exllamav3-1.4.7+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl"
URL = "https://github.com/turboderp-org/exllamav3/releases/download/v1.4.7/" + urllib.parse.quote(NAME)
API = "https://api.github.com/repos/turboderp-org/exllamav3/releases/tags/v1.4.7"
INFO = "exllamav3-1.4.7+cu128.torch2.10.0.dist-info"
NATIVE = "exllamav3_ext.cpython-312-x86_64-linux-gnu.so"
SPEC = {"bytes": 298402779, "sha256": "57bd760ebaccb8247a1c89503489f4390f3f98282ca501d8e87e5bae5b749e42",
        "members": {
            "exllamav3/generator/generator.py": "d9679f40791c091382e00e5278d815a19fa7856b9788c376b2ae7093d1cf8231",
            "exllamav3/architecture/qwen3_5_mtp.py": "91a71f29dd7c80336527a7cac6aa566e7b7b97ce9e38f0afe3469eeeff51d829",
            NATIVE: "90bc134fc33df7575413cfa9b709b4f094cc27e268d7f3c40f01227f4f3d329c",
            INFO + "/METADATA": "f94afb148c56413fb10b260d3577bd2fef682a72e888ae2efa37b9c72cf50e26",
            INFO + "/WHEEL": "d7c654382c56994f026699578f852d5dc8f4ca46aa0e07995b64d38f7904d9d7"}}
MAX_EXPANDED = 2 * 1024**3
MAX_FILES = 5000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def safe_path(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe wheel/receipt path")
    return path


class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlsplit(newurl)
        require(parts.scheme == "https" and parts.hostname in ("github.com", "api.github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com")
                and parts.username is None and parts.password is None, "unexpected upstream download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirects())


def request(url):
    return urllib.request.Request(url, headers={"User-Agent": "p5-wheel-byte-reconstruction/1.0", "Accept": "application/vnd.github+json"})


def published_identity():
    with opener().open(request(API), timeout=30) as response:
        payload = response.read(1024 * 1024 + 1)
    require(len(payload) <= 1024 * 1024, "oversize upstream release metadata")
    release = json.loads(payload)
    require(release.get("id") == 383298508 and release.get("tag_name") == "v1.4.7", "upstream release identity changed")
    assets = [row for row in release["assets"] if row.get("name") == NAME]
    require(len(assets) == 1, "missing/ambiguous exact upstream wheel")
    asset = assets[0]
    require(asset.get("id") == 546068592 and asset.get("size") == SPEC["bytes"] and
            asset.get("digest") == "sha256:" + SPEC["sha256"] and asset.get("browser_download_url") == URL,
            "upstream wheel identity differs from the recorded P5 pin")
    return {"api": API, "release_id": release["id"], "asset_id": asset["id"], "name": NAME,
            "published_at": release["published_at"], "asset_updated_at": asset["updated_at"],
            "bytes": asset["size"], "sha256": SPEC["sha256"], "download_url": URL,
            "basis": "anonymous HTTPS readback of publisher metadata; not an independent signature"}


def download(path):
    path = safe_path(path)
    require(not path.exists() and path.parent.is_dir(), "download needs a new file in an existing directory")
    identity = published_identity()
    with tempfile.TemporaryDirectory(prefix="p5-wheel-download-", dir=path.parent) as raw:
        temporary = Path(raw) / NAME
        h, total = hashlib.sha256(), 0
        with opener().open(request(URL), timeout=60) as response, temporary.open("xb") as writer:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                total += len(block)
                require(total <= SPEC["bytes"], "upstream wheel exceeded its declared size")
                h.update(block)
                writer.write(block)
        require(total == SPEC["bytes"] and h.hexdigest() == SPEC["sha256"], "downloaded wheel size/hash differs")
        os.chmod(temporary, 0o600)
        os.link(temporary, path)  # create-only, including a raced destination
    return identity


def wheel_members(archive):
    infos = archive.infolist()
    require(0 < len(infos) <= MAX_FILES and sum(row.file_size for row in infos) <= MAX_EXPANDED, "wheel exceeds inventory/resource bounds")
    names, members = set(), []
    for info in infos:
        path = PurePosixPath(info.filename)
        require(not info.is_dir() and path.parts and not path.is_absolute() and ".." not in path.parts and
                str(path) == info.filename and "\\" not in info.filename and not any(ord(c) < 32 for c in info.filename),
                "unsafe wheel member path")
        require(path.parts[0] in ("exllamav3", NATIVE, INFO), "unrelated wheel import root")
        require(info.filename.casefold() not in names, "duplicate/aliased wheel member")
        names.add(info.filename.casefold())
        mode = info.external_attr >> 16
        require(stat.S_IFMT(mode) in (0, stat.S_IFREG) and not (info.flag_bits & 1), "wheel contains a special/encrypted entry")
        require(path.suffix not in (".pth", ".pyc", ".pyo") and path.name not in ("sitecustomize.py", "usercustomize.py"), "wheel contains startup hooks/bytecode")
        members.append(info)
    return members


def verify(path):
    path = safe_path(path)
    require(path.is_file() and path.stat().st_size == SPEC["bytes"], "wrong wheel size/type")
    before = path.stat()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before), "wheel changed before hashing")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    require(h.hexdigest() == SPEC["sha256"], "wrong wheel archive SHA-256")
    records = []
    with zipfile.ZipFile(path) as archive:
        members = wheel_members(archive)
        lookup = {info.filename: info for info in members}
        require(set(SPEC["members"]) | {INFO + "/RECORD"} <= lookup.keys(), "required wheel members missing")
        require(lookup[INFO + "/RECORD"].file_size <= 1024 * 1024, "oversize wheel RECORD")
        index = list(csv.reader(io.StringIO(archive.read(INFO + "/RECORD").decode())))
        require(all(len(row) == 3 for row in index) and len(index) == len({row[0] for row in index}) and
                {row[0] for row in index} == set(lookup), "wheel RECORD inventory differs")
        indexed = {name: (digest, size) for name, digest, size in index}
        for info in sorted(members, key=lambda row: row.filename):
            h, count = hashlib.sha256(), 0
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    count += len(block)
                    require(count <= info.file_size, "wheel member exceeded its declared size")
                    h.update(block)
            digest = h.hexdigest()
            require(count == info.file_size, "wheel member size differs")
            record_hash, record_size = indexed[info.filename]
            if info.filename == INFO + "/RECORD":
                require(record_hash == record_size == "", "unexpected self-hash for wheel RECORD")
            else:
                expected = "sha256=" + base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
                require(record_hash == expected and record_size == str(count), "wheel RECORD member hash/size differs")
            if info.filename in SPEC["members"]:
                require(digest == SPEC["members"][info.filename], "wheel member differs from P5's observed identity: " + info.filename)
            records.append({"path": info.filename, "bytes": count, "sha256": digest})
    require(identity(path.stat()) == identity(before), "wheel changed during verification")
    core = {"schema": "paper5-upstream-wheel-byte-verification-v1", "status": "verified-bytes-not-runtime-execution",
            "author_approval": False, "checked_at": datetime.now(timezone.utc).isoformat(),
            "wheel": {"name": NAME, "bytes": SPEC["bytes"], "sha256": SPEC["sha256"], "download_url": URL},
            "verifier_sha256": sha(Path(__file__).read_bytes()), "members": records,
            "member_inventory_sha256": sha(canonical(records).encode()),
            "observation_comparisons": SPEC["members"], "package_installed": False, "native_loaded": False}
    return core


def write_receipt(path, record):
    path = safe_path(path)
    core = dict(record)
    core["receipt_sha256"] = sha(canonical(core).encode())
    with path.open("x") as stream:
        stream.write(canonical(core))
    return core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--download", action="store_true", help="explicitly fetch the fixed upstream asset; check live placement first")
    parser.add_argument("--receipt", required=True, type=Path, help="new local verification-record file")
    args = parser.parse_args()
    try:
        require(not safe_path(args.receipt).exists() and args.wheel.absolute() != args.receipt.absolute(), "receipt must be a new separate file")
        upstream = download(args.wheel) if args.download else None
        result = verify(args.wheel)
        if upstream:
            result["publisher_readback"] = upstream
        receipt = write_receipt(args.receipt, result)
        print(canonical({"status": receipt["status"], "wheel": receipt["wheel"], "members": len(receipt["members"]),
                         "expanded_bytes": sum(row["bytes"] for row in receipt["members"]),
                         "receipt_sha256": receipt["receipt_sha256"], "author_approval": False}), end="")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        parser.exit(2, "Wheel verification refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
