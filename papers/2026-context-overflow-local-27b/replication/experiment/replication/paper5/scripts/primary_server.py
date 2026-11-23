"""Owned-server observations for future P5 primary stages; imports launch nothing."""
import os
from pathlib import Path
import re
import socket
import subprocess
import uuid

from evidence import ROOT
from diagnostics import require
from draft_support import read_json, sha
import execution_snapshot as E
from migrate_followup_preparation import once, safe_path
from primary_requests import canonical


AMENDMENTS_REL = E.AMENDMENTS_REL


def expected_machine(baseline, root=ROOT):
    """The baseline machine with the recorded, field-limited amendments applied."""
    return E.apply_amendments(baseline, root)["machine"]


def min_free_vram_mib():
    """Operational VRAM floor (adjustable, per the lab rule); default 11,000 MiB."""
    value = int(os.environ.get("P5_MIN_FREE_VRAM_MIB", "11000"))
    require(9_900 <= value <= 12_282, "P5_MIN_FREE_VRAM_MIB must cover the model's measured ~9.8 GiB peak")
    return value


def preflight(baseline, root=ROOT):
    for name in ("p5-phase1.service", "p5-phase2.service"):
        state = E.command(["systemctl", "--user", "show", name, "--property=ActiveState", "--value"])
        require(state not in ("active", "activating", "deactivating", "reloading"), "primary preparation unit is still active: " + name)
    for port in (18125, 18126, 18127, 18130):
        with socket.socket() as sock:
            sock.settimeout(1)
            require(sock.connect_ex(("127.0.0.1", port)) != 0, f"study endpoint is already occupied: {port}")
    require(not E.gpu_processes(), "a GPU compute process is already active")
    require(E.hardware() == expected_machine(baseline, root), "study machine/GPU/driver changed from baseline")
    lines = E.command(["nvidia-smi", "--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits"]).splitlines()
    require(len(lines) == 1, "ambiguous study GPU")
    free, temperature = [int(x.strip()) for x in lines[0].split(",")]
    floor = min_free_vram_mib()
    require(free >= floor and temperature < 75, f"GPU needs {floor} MiB free and temperature below 75 C")
    mem = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    require(mem >= 24*1024*1024, "less than 24 GiB available host RAM")
    require(not any((root / f"testsuite/{paper}/PAUSE").exists() for paper in ("paper5", "paper6")), "study pause marker is present")


def llama_prefix(path):
    buf = bytearray()
    with Path(path).open("rb") as stream:
        for line in stream:
            buf.extend(line)
            require(len(buf) <= 1024*1024, "judge startup prefix exceeds 1 MiB")
            if b"listening on http://127.0.0.1:18126" in line:
                return bytes(buf)
    raise ValueError("judge startup readiness is missing from log")


def llama_placement(text):
    builds = re.findall(r"build (\d+) \(([0-9a-f]+)\)", text)
    require(not builds or E.LLAMA.startswith(builds[0][1]), "judge reports a different engine build")
    matches = list(re.finditer(r"offloaded (\d+)/(\d+) layers to GPU", text))
    require(matches, "judge did not report layer placement")
    final = text[matches[-1].start():]
    layers = tuple(int(v) for v in matches[-1].groups())
    require(0 < layers[0] <= layers[1], "invalid judge layer placement")
    contexts = re.findall(r"llama_context: n_ctx\s*=\s*(\d+)", final)
    require(contexts and set(contexts) == {"16384"}, "judge context readback differs")
    buffers = {}
    for kind in ("model", "KV"):
        selected = {}
        for name, amount in re.findall(r"(\S+) " + kind + r" buffer size\s*=\s*([\d.]+) MiB", final):
            selected[name] = round(selected.get(name, 0.) + float(amount), 2)
        require(selected.get("CUDA0", 0) > 0, "judge GPU buffer readback is missing: " + kind)
        buffers[kind] = selected
    # Preserve the actual buffers, not an inference that layer counters imply
    # full tensor residency. Six CPU MoE layers are a configured request only.
    return {"build_from_log": builds[0][1] if builds else None, "offloaded_layers": layers[0], "total_layers": layers[1],
            "n_ctx": 16384, "buffers_mib": buffers, "cpu_moe_layers_requested": 6}


def validate_version(text):
    # This pinned revision logs build details at trace verbosity; its explicit
    # --version output, not an invented normal-verbosity log line, is required.
    found = re.findall(r"commit ([0-9a-f]{7,40})\)", text)
    require(len(found) == 1 and E.LLAMA.startswith(found[0]), "judge binary version does not match the pinned commit")
    return text


JUDGE_LOG_VERBOSITY = "4"


def judge_server(a):
    class ScopedLlama(a.runner.LlamaServer):
        def cmd_env_cwd(self):
            cmd, env, cwd = super().cmd_env_cwd()
            env["LD_LIBRARY_PATH"] = str(Path.home() / "cuda-12.6/lib64")
            # This pinned revision routes the library's info messages (layer offload and
            # buffer sizes, which the placement receipt requires) to log level 4; the default
            # is 3. Logging only; the argv stays the frozen judge command
            # (research/P5_AMENDMENT_2026-10-29.md).
            env["LLAMA_ARG_LOG_VERBOSITY"] = JUDGE_LOG_VERBOSITY
            return cmd, env, cwd
    original = a.judge.gptoss_server()
    return ScopedLlama(original.name, original.port, original.model, original.extra)


def version_readback(server):
    cmd, env, cwd = server.cmd_env_cwd()
    run = subprocess.run([cmd[3], "--version"], env=env, cwd=cwd, capture_output=True, text=True, timeout=30, check=True)
    return validate_version((run.stdout + run.stderr).strip())


def llama_identity(server, root=ROOT):
    pid = server.proc.pid
    proc = Path("/proc") / str(pid)
    before = E.process_stat((proc / "stat").read_text())
    cmd, env, cwd = server.cmd_env_cwd()
    require(cmd[:3] == ["nice", "-n", "5"], "unexpected judge launch wrapper")
    expected = cmd[3:]
    argv = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    require(argv == expected and (proc / "exe").resolve() == Path(expected[0]).resolve() and
            (proc / "cwd").resolve() == Path(cwd).resolve(), "judge PID is not the exact owned command")
    require(server.name == "judge-gptoss20b" and server.port == 18126, "wrong judge server")
    listeners = E.listening_inodes((proc / "net/tcp").read_text(), server.port)
    fds = set()
    for fd in (proc / "fd").iterdir():
        try:
            fds.add(os.readlink(fd))
        except FileNotFoundError:
            continue
    require(len(listeners) == 1 and {"socket:[" + v + "]" for v in listeners} <= fds, "judge PID does not own the listener")
    log = (proc / "fd/1").resolve()
    require(log == Path(server.log_path).resolve() and log.is_relative_to(root / "testsuite/paper5/work/logs"), "judge stdout/log mismatch")
    mapped = E.mapped_libraries((proc / "maps").read_text())
    require(str(Path(expected[0]).resolve()) in mapped, "judge binary is not mapped")
    require(E.process_stat((proc / "stat").read_text()) == before, "judge process identity changed")
    recorded_env = dict(entry.split(b"=", 1) for entry in (proc / "environ").read_bytes().split(b"\0") if b"=" in entry)
    require(recorded_env.get(b"LD_LIBRARY_PATH", b"").decode() == env["LD_LIBRARY_PATH"], "judge library-path override differs")
    require(recorded_env.get(b"LLAMA_ARG_LOG_VERBOSITY", b"").decode() == env["LLAMA_ARG_LOG_VERBOSITY"], "judge log verbosity differs")
    return {**before, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "argv": argv,
            "cwd": str(Path(cwd).resolve()), "executable": str((proc / "exe").resolve()),
            "log": str(log), "mapped_files": sorted(mapped), "listener_port": server.port,
            "environment_allowlist": {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
                                      "LLAMA_ARG_LOG_VERBOSITY": env["LLAMA_ARG_LOG_VERBOSITY"]}}


class ObservedServer:
    """Wrap only this invocation's server object; the frozen source stays intact."""
    def __init__(self, raw, session, session_sha, baseline, *, root=ROOT, inherited_placement=None):
        self.raw, self.session, self.session_sha, self.root = raw, Path(session), session_sha, root
        self.baseline = E.apply_amendments(baseline, root)
        # Stat identities of the content-verified baseline files, taken at session start: the
        # launcher runs the full content check (E.check) just before constructing this object,
        # so these are the reference for "unchanged during this stage". Content stays bound to
        # the 2026-09-20 baseline by E.check before and after every stage; comparing stat
        # identities to 09-20 instead would block forever on harmless metadata-only changes
        # (research/P5_AMENDMENT_2026-09-25.md).
        # A regenerable Triton cache library that is absent at session start has no stat
        # identity yet; if the pinned stack compiles it later, its hash is checked after the stage.
        self.identity_ref = {p: E.identity(p) for p, v in self.baseline["files"].items() if E.present(p, v)}
        self.instance = None
        self.placement = inherited_placement
        self.runtime_files = {}

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def start(self, *args, **kwargs):
        binary_version = version_readback(self.raw) if self.raw.name == "judge-gptoss20b" else None
        self.raw.start(*args, **kwargs)
        if self.instance and self.instance["server"]["pid"] == self.raw.proc.pid:
            self.verify_live()
            return
        server = E.owned_server(self.raw.proc.pid, self.root) if self.raw.name == "tabby-p5" else llama_identity(self.raw, self.root)
        if self.raw.name == "tabby-p5":
            log = E.startup_prefix(server["log"], server["pid"])
            placement = {"server": "tabby-p5", "mtp": True, "max_seq_len": 32768,
                         "config_sha256": sha(self.root / "testsuite/paper5/tabby-p5.yml")}
        else:
            log = llama_prefix(server["log"])
            placement = {"server": "judge-gptoss20b", **llama_placement(log.decode())}
        require(self.placement is None or self.placement == placement, "server placement changed across invocations/restarts")
        machine = E.hardware()
        require(machine == expected_machine(self.baseline, self.root), "running server is on changed hardware/driver")
        processes = E.gpu_processes()
        require(len(processes) == 1 and processes[0]["pid"] == server["pid"] and
                processes[0]["gpu_uuid"] == machine["gpu"]["uuid"], "GPU compute owner differs from server PID")
        files = {p: E.file_record(p) for p in server["mapped_files"]}
        for p, record in files.items():
            if p in self.baseline["files"]:
                expected = self.baseline["files"][p]
                accepted = expected["accepted_sha256"] if expected.get("regenerable") else [expected["sha256"]]
                require(record["sha256"] in accepted, "mapped baseline runtime changed")
            if p in self.runtime_files:
                require(record["sha256"] == self.runtime_files[p]["sha256"], "mapped supplemental runtime changed across restarts")
        self.runtime_files.update(files)
        record = {"schema": "p5-primary-server-instance-v1", "session_sha256": self.session_sha,
                  "observed_at": E.utc(), "server": server, "machine": machine, "gpu_processes": processes,
                  "placement": placement, "runtime_files": files,
                  "binary_version_readback": binary_version,
                  "startup_prefix": log.decode(), "startup_prefix_sha256": E.hashlib.sha256(log).hexdigest()}
        path = safe_path(self.session, "servers/" + uuid.uuid4().hex + ".json")
        once(path, canonical(record) + "\n")
        self.instance = {**record, "path": str(path), "sha256": sha(path)}
        self.placement = placement
        self.verify_live()

    def verify_live(self):
        require(self.raw.proc is not None and self.raw.proc.poll() is None and self.instance is not None, "owned server is not live/observed")
        current = E.owned_server(self.raw.proc.pid, self.root) if self.raw.name == "tabby-p5" else llama_identity(self.raw, self.root)
        previous = self.instance["server"]
        for key in ("pid", "ppid", "start_ticks", "boot_id", "argv", "cwd", "executable", "listener_port", "log", "environment_allowlist"):
            require(current[key] == previous[key], "serving identity changed: " + key)
        # Known native files are checked by stat at every gate; full hashes are
        # verified at instance creation and after the stage. A newly mapped
        # library needs a new observation before it can serve another request.
        require(set(current["mapped_files"]) <= self.runtime_files.keys() | self.baseline["files"].keys(),
                "server mapped an unobserved runtime library; preserve results and inspect")
        reference = {**self.identity_ref,
                     **{p: {k: v for k, v in record.items() if k != "sha256"} for p, record in self.runtime_files.items()}}
        for p, ident in reference.items():
            require(E.identity(p) == ident, "runtime/input file identity changed: " + p)
        processes = E.gpu_processes()
        require(len(processes) == 1 and processes[0]["pid"] == current["pid"] and
                processes[0]["gpu_uuid"] == self.baseline["machine"]["gpu"]["uuid"], "GPU compute ownership changed")

    def stop(self):
        instance = self.instance
        self.raw.stop()  # only the raw object's owned process group
        if instance is not None:
            path = Path(instance["path"])
            once(path.with_name(path.stem + "-stopped.json"), canonical({
                "instance_sha256": instance["sha256"], "stopped_at": E.utc(),
                "log_sha256": sha(instance["server"]["log"]), "log_bytes": Path(instance["server"]["log"]).stat().st_size}) + "\n")
            self.instance = None

    def reference(self):
        return None if self.instance is None else {"path": str(Path(self.instance["path"]).relative_to(self.root)),
                                                  "sha256": self.instance["sha256"]}
