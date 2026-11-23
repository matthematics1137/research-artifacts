#!/usr/bin/env python3
"""Future guarded qualification only: two modes, four synthetic generations.

Inert unless --run. The queue supplies 890 seconds plus a ten-second kill tail.
This driver reserves cleanup time by stopping at 880 seconds internally. No
benchmark dataset is imported/read, and no generation request is ever retried.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

from adapter import load_legacy, normalize

PIN = 'f4ab369a06c728325b706c989d71be7dc0c05a1b'
REV = '841f7812793972b5de6dc48910bdd1e2722ae48b'
MODEL_SHA = 'cdc4810b0ff17c40d0f62cf214b6e0bcd08346e9eb05ca53371507037793c14a'
TEMPLATE_SHA = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
PROMPT = 'Compute 3 + 4. End with Answer: 7.'
MODEL_ID = 'bonsai2-27b'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def server_argv(lab, mode):
    if mode not in ('none', 'mtp3'):
        raise ValueError('unknown qualification mode')
    src = lab / 'src/ninfer-f4ab369a'
    argv = [str(src / 'build-sm75/apps/ninfer-serve'),
            str(lab / 'models/bonsai2-ninfer-841f7812/Ternary-Bonsai-2-27B-ninfer-v3.ninfer'),
            '--host', 'localhost', '--port', '8092', '--model-id', MODEL_ID,
            '--max-context', '16384', '--kv-capacity', '16384', '--max-concurrency', '1',
            '--kv-dtype', 'int8', '--gdn-state-fp16', '--prefix-reuse',
            '--default-reasoning-effort', 'medium', '--reasoning-loop', 'off',
            '--no-post-thinking', '--ngram-draft-tokens', '0', '--lookup-ngram', '0']
    argv += (['--spec', 'none'] if mode == 'none' else
             ['--spec', 'mtp', '--draft-tokens', '3', '--draft-min-p', '0',
              '--no-adaptive-mtp', '--full-draft-head'])
    return argv


def request_body(effort='medium', cap=4096):
    return {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': PROMPT}],
            'temperature': 0, 'top_p': .95, 'top_k': 20, 'max_tokens': cap,
            'reasoning_effort': effort, 'enable_thinking': True, 'stream': False, 'n': 1}


def render_original(template, effort):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(message)
    env.globals['raise_exception'] = fail
    env.filters['tojson'] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
    return env.from_string(template).render(messages=[{'role': 'user', 'content': PROMPT}],
                                          add_generation_prompt=True, enable_thinking=True,
                                          reasoning_effort=effort)


def validate_templates(original, actual):
    expected = {effort: render_original(original, effort) for effort in ('medium', 'xhigh')}
    if expected != actual:
        raise ValueError('NInfer rendered template differs from original Qwen template')
    if actual['medium'] == actual['xhigh']:
        raise ValueError('medium/xhigh effort has no rendered effect')
    if any(not text.endswith('<|im_start|>assistant\n<think>\n') for text in actual.values()):
        raise ValueError('thinking generation prefix is not open')
    return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in actual.items()}


def validate_response(native, cap):
    if native.get('model') != MODEL_ID:
        raise ValueError('served model mismatch')
    legacy = load_legacy()
    logical, how = normalize(native, legacy)
    usage = native['usage']
    if usage.get('total_tokens') != usage['prompt_tokens'] + usage['completion_tokens']:
        raise ValueError('native token accounting mismatch')
    choice = native['choices'][0]
    if cap == 1:
        if choice['finish_reason'] != 'length' or usage['completion_tokens'] != 1:
            raise ValueError('one-token synthetic cap not honored')
    else:
        answer = legacy.visible_content(logical['choices'][0]['message'])
        if choice['finish_reason'] != 'stop' or not legacy.SCORERS['gsm8k'](answer, {'answer_number': 7}):
            raise ValueError('normal synthetic response did not finish with the known answer')
    return {'finish_reason': choice['finish_reason'], 'usage': usage,
            'normalization': how, 'native_shape': 'split' if choice['message'].get('reasoning_content') else 'inline',
            'logical_response': logical, 'byte_fidelity': 'not claimed; registered logical scoring'}


def sync_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.used,memory.total,power.limit,temperature.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(io.StringIO(raw)))
    if len(rows) != 1:
        raise ValueError('qualification requires exactly one visible card')
    name, cc, used, total, power, temp = [x.strip() for x in rows[0]]
    result = {'name': name, 'compute_cap': cc, 'memory_used_mib': float(used),
              'memory_total_mib': float(total), 'power_limit_w': float(power), 'temperature_c': float(temp)}
    if name != 'NVIDIA GeForce RTX 2080 Ti' or cc != '7.5' or float(power) != 260:
        raise ValueError('card/architecture/power limit mismatch')
    if not 0 <= float(used) <= float(total):
        raise ValueError('invalid device residency')
    return result


def run(lab, out):
    started = time.monotonic(); deadline = started + 880
    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('qualification 880-second internal deadline')
        return min(left, 120)
    # An atomic fresh claim makes queue retries inert before any model is started.
    out.mkdir(parents=True, exist_ok=False)
    sync_json(out / 'qualification.intent.json', {'pin': PIN, 'revision': REV, 'prompt': PROMPT,
                                                'internal_cap_s': 880, 'queue_cap_with_tail_s': 900})
    src = lab / 'src/ninfer-f4ab369a'
    task = lab / 'src/p8-bonsai-lane'
    binary = src / 'build-sm75/apps/ninfer-serve'
    model = lab / 'models/bonsai2-ninfer-841f7812/Ternary-Bonsai-2-27B-ninfer-v3.ninfer'
    server = None
    quality_lock = None
    state = {'status': 'failed', 'modes_complete': [], 'gpu_processes_started': 0}
    def interrupted(signum, frame):
        raise InterruptedError('qualification interrupted by queue/guard')
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            raise ValueError('local qualification redirect refused')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    def post(label, endpoint, body):
        # Save/fsync intent before the single POST; never retry generation or rendering.
        data = json.dumps(body).encode()
        (out / (label + '.request.json')).write_bytes(data)
        sync_json(out / (label + '.intent.json'), {'endpoint': endpoint, 'request_sha256': hashlib.sha256(data).hexdigest()})
        req = urllib.request.Request('http://localhost:8092' + endpoint, data=data, headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=remaining()) as response:
            raw = response.read()
        (out / (label + '.response.json')).write_bytes(raw)
        return json.loads(raw)
    def get(endpoint):
        with opener.open('http://localhost:8092' + endpoint, timeout=min(2, remaining())) as response:
            return json.loads(response.read())
    try:
        if subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
            raise ValueError('source pin mismatch')
        if subprocess.check_output(['git', '-C', str(src), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('NInfer source is dirty')
        cache = (src / 'build-sm75/CMakeCache.txt').read_text()
        if not any(line.startswith('CMAKE_CUDA_ARCHITECTURES:') and line.split('=', 1)[1] == '75' for line in cache.splitlines()):
            raise ValueError('build cache is not sm_75')
        binary_sha = sha_file(binary)
        if model.stat().st_size != 9520051456 or sha_file(model) != MODEL_SHA:
            raise ValueError('model file size/hash mismatch')
        original_path = task / 'references/original-qwen-chat-template.jinja'
        if sha_file(original_path) != TEMPLATE_SHA:
            raise ValueError('original serving template hash mismatch')
        original = original_path.read_text()
        subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'turing-guard'], check=True)
        units = re.findall(r'turing-run-[A-Za-z0-9_.-]+\.service', Path('/proc/self/cgroup').read_text())
        if len(set(units)) != 1:
            raise ValueError('qualification must run in its guarded queue unit')
        active = subprocess.check_output(['systemctl', '--user', 'list-units', '--state=active,activating,deactivating',
                                          '--plain', '--no-legend', 'turing-run-*'], text=True)
        if any(line.split()[0] != units[0] for line in active.splitlines() if line.split()):
            raise ValueError('another lab unit is active')
        quality_lock = (lab / 'runs/quality.lock').open('a')
        fcntl.flock(quality_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(); sync_json(out / 'gpu-before.json', before)
        if before['temperature_c'] >= 65:
            raise ValueError('card is not cool enough to qualify')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=process_name', '--format=csv,noheader'], text=True)
        if any(line.strip() and 'sunshine' not in line.lower() for line in apps.splitlines()):
            raise ValueError('another GPU compute workload is present')
        with socket.socket() as sock:
            sock.settimeout(2)
            if sock.connect_ex(('localhost', 8092)) == 0:
                raise ValueError('qualification port is occupied')
        prior_elapsed = 0
        for mode in ('none', 'mtp3'):
            argv = server_argv(lab, mode)
            sync_json(out / (mode + '.server-intent.json'), {'argv': argv, 'binary_sha256': binary_sha, 'model_sha256': MODEL_SHA})
            with (out / (mode + '.server.log')).open('x') as log:
                server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                          env=dict(os.environ, TMPDIR=str(out), XDG_CACHE_HOME=str(out / 'cache'),
                                                   CUDA_CACHE_PATH=str(out / 'cache/cuda')))
                state['gpu_processes_started'] += 1
                while True:
                    remaining()
                    if server.poll() is not None:
                        raise ValueError('server exited before healthy')
                    try:
                        if get('/health').get('status') == 'ok':
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.5)
                # Inspect the actual running process argv, not only the planned command.
                actual_argv = Path('/proc/' + str(server.pid) + '/cmdline').read_bytes().decode().rstrip('\0').split('\0')
                if actual_argv != argv:
                    raise ValueError('actual server argv differs from planned mode')
                props = get('/props'); sync_json(out / (mode + '.props.json'), props)
                settings = props['default_generation_settings']
                if (settings['n_ctx'] != 16384 or props['total_slots'] != 1
                        or settings['speculative'] != (mode == 'mtp3')):
                    raise ValueError('effective context/concurrency/speculation mismatch')
                rendered = {effort: post(mode + '-template-' + effort, '/apply-template', request_body(effort))['prompt']
                            for effort in ('medium', 'xhigh')}
                template_hashes = validate_templates(original, rendered)
                shapes = {}
                for name, cap in (('normal', 4096), ('one-token', 1)):
                    native = post(mode + '-' + name, '/v1/chat/completions', request_body(cap=cap))
                    shapes[name] = validate_response(native, cap)
                loaded = snapshot(); sync_json(out / (mode + '.gpu-loaded.json'), loaded)
                server.terminate()
                try:
                    server.wait(timeout=min(5, remaining()))
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait(timeout=2)
                server = None
            elapsed = math.ceil(time.monotonic() - started)
            contract = json.loads((task / 'research/raw/2026-11-12-p8-bonsai-lane/contract.json').read_text())
            receipt = dict(contract['runtime_required'], mode=mode, binary_sha256=binary_sha,
                           qualification_elapsed_s=elapsed-prior_elapsed,
                           evidence_paths=[str(out / (mode + '.server.log')), str(out / (mode + '.props.json')),
                                           str(out / (mode + '-normal.response.json')), str(out / (mode + '-one-token.response.json'))],
                           template_hashes=template_hashes, response_checks=shapes,
                           ngram_draft_tokens=0, lookup_ngram=0, actual_server_argv=actual_argv,
                           original_template_sha256=TEMPLATE_SHA, gpu_loaded=loaded)
            receipt['registration_commit'] = '0b2959c73fa5392bfd1e3ed33fb594c862ef7f9a'
            receipt['registration_rife'] = 'research-ledger-reference'
            sync_json(out / (mode + '.json'), receipt)
            prior_elapsed = elapsed; state['modes_complete'].append(mode)
        state['status'] = 'complete'
        return 0
    except BaseException as error:
        state['error'] = type(error).__name__ + ': ' + str(error)
        return 3
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=2)
        state['elapsed_s'] = time.monotonic() - started
        sync_json(out / 'qualification-summary.json', state)
        if quality_lock is not None:
            quality_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, default=Path('/opt/p8'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='start GPU engines only in the ambassador-queued qualification')
    args = parser.parse_args()
    if args.run:
        raise SystemExit(run(args.lab, args.output_dir))
    print(json.dumps({'inert': True, 'modes': {m: server_argv(args.lab, m) for m in ('none', 'mtp3')},
                      'prompt': PROMPT, 'internal_cap_s': 880, 'required_queue_timeout_s': 890, 'termination_tail_s': 10}, indent=2))
