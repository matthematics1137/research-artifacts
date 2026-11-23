#!/usr/bin/env python3
"""Single-attempt NInfer transport for P8's existing CPU eval client.

No retry, thinking fallback, new prompt, scorer or token estimate. Native response
is retained; P8's existing laptop_inline_text supplies the scoring representation.
The normalization does not establish byte fidelity to a parser-disabled engine.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[3]


def load_legacy():
    path = REPO / 'tools/turing/quality/run_eval_turing.py'
    spec = importlib.util.spec_from_file_location('p8_legacy_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(data):
    return hashlib.sha256(data).hexdigest()


def translate(body, model):
    if (body.get('temperature') != 0 or body.get('max_tokens') != 4096
            or body.get('top_p') != 0.95 or body.get('top_k') != 20
            or body.get('chat_template_kwargs') != {'reasoning_effort': 'medium'}):
        raise ValueError('P8 sampling/thinking contract mismatch')
    allowed = {'model', 'messages', 'temperature', 'top_p', 'top_k', 'max_tokens',
               'chat_template_kwargs'}
    messages = body.get('messages')
    if (set(body) != allowed or not isinstance(messages, list) or len(messages) != 1
            or set(messages[0]) != {'role', 'content'}
            or messages[0]['role'] != 'user' or not isinstance(messages[0]['content'], str)):
        raise ValueError('P8 single-user-message contract mismatch')
    wire = copy.deepcopy(body)
    wire.pop('chat_template_kwargs')
    wire.update(model=model, reasoning_effort='medium', enable_thinking=True,
                stream=False, n=1)
    return wire


def normalize(native, legacy):
    if not isinstance(native, dict) or len(native.get('choices', [])) != 1:
        raise ValueError('expected exactly one choice')
    choice = native['choices'][0]
    if choice.get('finish_reason') not in ('stop', 'length'):
        raise ValueError('unexpected finish reason')
    msg = choice.get('message', {})
    if msg.get('tool_calls') or msg.get('refusal'):
        raise ValueError('unexpected tool call/refusal')
    for field in ('content', 'reasoning_content'):
        if msg.get(field) is not None and not isinstance(msg[field], str):
            raise ValueError('non-text response')
    usage = native.get('usage') or {}
    for field in ('prompt_tokens', 'completion_tokens'):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise ValueError('native integer token usage required')
    inline, how = legacy.laptop_inline_text(msg, choice['finish_reason'])
    out = copy.deepcopy(native)
    out['choices'][0]['message'] = {'role': 'assistant', 'content': inline}
    return out, how


class Adapter:
    def __init__(self, base_url, model='bonsai2-27b', timeout=600, opener=None):
        u = urllib.parse.urlsplit(base_url)
        if u.scheme != 'http' or u.hostname != 'localhost' or u.username or u.password or u.query or u.fragment or u.path not in ('', '/'):
            raise ValueError('use an owner-established localhost HTTP endpoint with no credentials')
        self.base_url = base_url.rstrip('/')
        self.model, self.timeout = model, timeout
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                raise ValueError('generation redirect refused')
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.legacy = load_legacy()
        self.last_exchange = None

    def chat_completion(self, base_url, prompt, args, use_ctk=True):
        self.last_exchange = None
        if base_url.rstrip('/') != self.base_url or not use_ctk:
            raise ValueError('endpoint change or thinking fallback refused')
        original = self.legacy.request_body_bytes(prompt, args, use_ctk=True)
        wire = json.dumps(translate(json.loads(original), self.model)).encode()
        self.last_exchange = {'original_request_sha256': digest(original),
                              'wire_request_sha256': digest(wire),
                              'wire_request': json.loads(wire), 'attempts': 1}
        req = urllib.request.Request(self.base_url + '/v1/chat/completions', data=wire,
                                     headers={'Content-Type': 'application/json'})
        # One generation POST. Even HTTP errors and lost replies are never repeated.
        with self.opener.open(req, timeout=self.timeout) as response:
            raw = response.read()
        native = json.loads(raw)
        self.last_exchange.update(native_response=native, native_response_sha256=digest(raw))
        if native.get('model') != self.model:
            raise ValueError('served model id mismatch')
        normalized, how = normalize(native, self.legacy)
        self.last_exchange.update(normalization=how, byte_fidelity='unverified',
                                  normalized_response_sha256=digest(json.dumps(normalized).encode()))
        return normalized, False
