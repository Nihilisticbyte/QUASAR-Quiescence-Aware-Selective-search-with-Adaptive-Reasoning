"""
Phase 3: analyse the instability log and build the trigger model.

Reads data/instability_log.csv (from log_instability.py) and prints
  1. how well each feature predicts "the value changes with one more ply"
  2. precision / recall / share-flagged for candidate trigger rules
  3. P(changed | best immediate capture gain), saved to data/trigger_model.json,
     which the cost-weighted controllers (costw_*) load.

LOOK at these numbers before choosing a threshold (project-plan.md, Phase 3).

Usage:  python analyze_instability.py [--log data/instability_log.csv]
"""

import argparse
import json
import os

from engine.trigger_analysis import load_log, bucket_model, format_report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=os.path.join("data", "instability_log.csv"))
    ap.add_argument("--model-out", default=os.path.join("data", "trigger_model.json"))
    args = ap.parse_args()

    rows = load_log(args.log)
    if len(rows) < 50:
        raise SystemExit(f"Only {len(rows)} rows in {args.log}: run log_instability.py first "
                         "(a model from so few nodes would not mean anything).")
    model = bucket_model(rows)
    print(format_report(rows, model))

    model_json = {k: v for k, v in model.items() if k not in ("ci_lo", "ci_hi")}
    model_json["source_log"] = args.log
    model_json["n_nodes"] = len(rows)
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    with open(args.model_out, "w") as f:
        json.dump(model_json, f, indent=1)
    print(f"\ntrigger model written to {args.model_out}")


if __name__ == "__main__":
    main()
