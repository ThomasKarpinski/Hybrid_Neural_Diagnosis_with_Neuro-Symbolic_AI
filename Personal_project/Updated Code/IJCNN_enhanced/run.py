"""Convenience CLI for the IJCNN project.

Examples:
  python run.py baseline
  python run.py hpo
  python run.py paper

This file does not change any model logic; it simply calls the existing modules.
"""

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python run.py {baseline|hpo|paper}")
        return 2

    mode = sys.argv[1].lower()

    if mode == "baseline":
        from src.pipeline import run_pipeline

        run_pipeline()
        return 0

    if mode == "hpo":
        from src.pipeline import run_all_hpo

        run_all_hpo()
        return 0

    if mode == "paper":
        # generate_paper_outputs is written as a script; execute it as a module.
        import subprocess
        import sys as _sys

        subprocess.check_call([_sys.executable, "-m", "src.analysis.generate_paper_outputs"])
        return 0

    print(f"Unknown mode: {mode}. Expected one of: baseline, hpo, paper")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
