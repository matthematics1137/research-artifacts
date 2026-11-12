#!/usr/bin/env python3
"""Make a moved lab self-contained: rewrite an old absolute path prefix to the new one, in place.

Usage: relocate_paths.py OLD NEW [ROOT]   (ROOT defaults to NEW)
- Text files (UTF-8, under 8 MB) containing OLD are rewritten with NEW (script shebangs, venv
  activators, pyvenv.cfg, .pth files, direct_url.json, ...), keeping their permission bits.
- Symlinks whose target starts with OLD are repointed to NEW via an atomic rename of a fresh link.
- Binary files that mention OLD are only reported. Nothing is deleted.
Prints a summary and every path it could not fix; exit code 1 if anything still points at OLD.
"""
import os
import sys


def main():
    old, new = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    root = sys.argv[3] if len(sys.argv) > 3 else new
    texts = links = 0
    left = []
    for base, dirs, files in os.walk(root):
        for name in dirs + files:
            path = os.path.join(base, name)
            if os.path.islink(path):
                target = os.readlink(path)
                if target.startswith(old):
                    fresh = path + ".relink"
                    os.symlink(new + target[len(old):], fresh)
                    os.replace(fresh, path)  # atomic swap of the link itself
                    links += 1
                continue
            if name not in files or not os.path.isfile(path) or os.path.getsize(path) > 8 << 20:
                continue
            with open(path, "rb") as f:
                data = f.read()
            if old.encode() not in data:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                left.append(path + "  (binary)")
                continue
            mode = os.stat(path).st_mode
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text.replace(old, new))
            os.chmod(path, mode)
            texts += 1
    print(f"rewrote {texts} text files, repointed {links} symlinks")
    for path in left:
        print("NOT FIXED:", path)
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
