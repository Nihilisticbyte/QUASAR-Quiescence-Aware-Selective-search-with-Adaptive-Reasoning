"""
Run every test script and print one summary line each.

    python run_tests.py            # all tests
    python run_tests.py --fast     # skip the slow Phase 0 self-play test

Stockfish-dependent tests (test_oracle, test_harness live run) report SKIPPED
instead of failing when Stockfish cannot be found; set STOCKFISH_PATH to run them.
Exit code 0 = nothing failed.
"""

import subprocess
import sys
import time

TESTS = ["test_phase0", "test_phase1", "test_phase2_budget", "test_positions",
         "test_oracle", "test_harness", "test_phase3", "test_extensions", "test_phase5"]


def main():
    fast = "--fast" in sys.argv
    failed = []
    for name in TESTS:
        if fast and name == "test_phase0":
            print(f"{name:<22} skipped (--fast)")
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, f"{name}.py"], capture_output=True, text=True)
        out = p.stdout + p.stderr
        dt = time.time() - t0
        if p.returncode != 0:
            status = "FAIL"
            failed.append(name)
        elif "SKIPPED" in out:
            status = "ok (some checks SKIPPED: no Stockfish)"
        else:
            status = "ok"
        print(f"{name:<22} {status:<40} {dt:5.1f}s")
        if p.returncode != 0:
            print("\n".join("    " + l for l in out.strip().splitlines()[-12:]))
    print("\n" + ("ALL TESTS PASSED" if not failed else "FAILED: " + ", ".join(failed)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
