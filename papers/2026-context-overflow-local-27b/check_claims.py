"""Verify public scored-data calculations; never run model inference."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, '-B', str(root / 'verify_inputs.py'), '--staged'], cwd=root, check=True)
subprocess.run([sys.executable, '-B', str(root / 'replication/analysis/check_results.py')], cwd=root, check=True)
print('Scored-data claim verification passed; no regrading, inference or LoCoMo bootstrap was performed.')
