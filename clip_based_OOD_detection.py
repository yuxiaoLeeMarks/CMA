"""
Convenience entry point for running Concept Matching with Agents (CMA) inside the
original CLIP-based OOD evaluation pipeline. Any CLI arguments accepted by
`eval_ood_detection.py` can be forwarded to this script; it simply enforces
`--score CMA` when the flag is not explicitly provided.
"""

import sys
from eval_ood_detection import main as eval_main


def _ensure_cma_score(argv):
    for arg in argv[1:]:
        if arg.startswith("--score"):
            return
    argv[1:1] = ["--score", "CMA"]


def main():
    _ensure_cma_score(sys.argv)
    eval_main()


if __name__ == "__main__":
    main()
