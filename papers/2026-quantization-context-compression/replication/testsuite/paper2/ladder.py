#!/usr/bin/env python3
"""Faithful public reimplementation of the tsi compression ladder (paper 2).

Ported from the private corpus pipeline so the SAME transformation runs on the
public replication corpus (WildChat). Rungs:
  A = advanced stopword filter only
  E = code/error removal (confidence >= 0.85) THEN advanced stopword filter
Destructive by design: lowercases, keeps [a-zA-Z]+ words of length >= 3 only —
digits do NOT survive (an analysis stratum, not a bug).
"""
import re

CODE_PATTERNS = [  # (regex, confidence), priority order
    (r"```[\s\S]*?```", 0.99),
    (r"def\s+\w+\s*\([^)]*\):\s*\n(?:\s{4}.*\n)*", 0.95),
    (r"class\s+\w+.*?:\s*\n(?:\s{4}.*\n)*", 0.95),
    (r"^(import|from)\s+[\w.]+.*$", 0.90),
    (r"(ERROR:|Error:|error:|404|500|stdout:|stderr:|Traceback|Exception|"
     r"SyntaxError|NameError|TypeError|ValueError|AttributeError|IndexError|"
     r"KeyError|FileNotFoundError|PermissionError|ConnectionError|TimeoutError|"
     r"RuntimeError|AssertionError|ImportError|ModuleNotFoundError|"
     r"IndentationError|TabError|UnboundLocalError|RecursionError|MemoryError|"
     r"SystemError|OSError|IOError|EOFError|KeyboardInterrupt|SystemExit|"
     r"GeneratorExit|StopIteration|StopAsyncIteration|ArithmeticError|"
     r"OverflowError|ZeroDivisionError|FloatingPointError|BufferError|"
     r"LookupError|UnicodeError|UnicodeDecodeError|UnicodeEncodeError|"
     r"UnicodeTranslateError|ReferenceError|Warning|UserWarning|"
     r"DeprecationWarning|PendingDeprecationWarning|SyntaxWarning|"
     r"RuntimeWarning|FutureWarning|ImportWarning|UnicodeWarning|BytesWarning|"
     r"ResourceWarning)", 0.92),
    (r"`[^`\n]{3,}`", 0.85),
    (r"<[^>]{3,}>.*?</[^>]+>", 0.80),
]

BASIC_STOPWORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
    'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old',
    'see', 'two', 'way', 'who', 'boy', 'did', 'she', 'use', 'man', 'oil', 'sit', 'set', 'run',
    'eat', 'far', 'sea', 'eye',
}
ADVANCED_STOPWORDS = BASIC_STOPWORDS | {
    'i', 'me', 'my', 'myself', 'we', 'ours', 'ourselves', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'himself', 'hers', 'herself', 'it', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'were', 'be', 'been', 'being',
    'have', 'having', 'do', 'does', 'doing', 'a', 'an', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'up', 'down', 'in', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'will', 'just', 'don',
    'should', 'uses', 'used', 'using', 'gets', 'got', 'make', 'makes', 'made',
    'user', 'users', 'system', 'systems', 'data', 'process', 'processes',
    'from', 'help', 'helps', 'helping', 'try', 'tries', 'trying', 'know',
    'knows', 'think', 'thinks', 'based',
}

def remove_code(text, min_confidence=0.85):
    spans = []
    for rx, conf in CODE_PATTERNS:
        if conf < min_confidence:
            continue
        for m in re.finditer(rx, text, re.MULTILINE):
            s, e = m.span()
            if not any(bs <= s < be or bs < e <= be for bs, be in spans):
                spans.append((s, e))
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + text[e:]
    return text

def stopword_filter(text, level="advanced"):
    sw = ADVANCED_STOPWORDS if level == "advanced" else BASIC_STOPWORDS
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return " ".join(w for w in words if w not in sw and len(w) >= 3)

def rung_A(text):
    return stopword_filter(text)

def rung_E(text):
    return stopword_filter(remove_code(text))
