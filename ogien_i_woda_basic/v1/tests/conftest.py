"""Puts v1/ on sys.path so tests can `import main` and `from generator... import ...`."""
import sys
from pathlib import Path

V1_DIR = Path(__file__).resolve().parents[1]
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))
