"""Put the repository root on `sys.path` so `pytest` works without `python -m`.

The package is not installed in the test environment, and `build_tibkat_csv.py` is a
root-level script rather than part of the package, so neither imports otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
