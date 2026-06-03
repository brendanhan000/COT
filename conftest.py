"""Root conftest so the in-repo ``cot`` package is importable during tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
