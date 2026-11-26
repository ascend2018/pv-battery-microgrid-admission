"""Run the proposed method on a supplied case JSON file."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The input supplies a seed and an ordered schedule of 24- or 48-hour units. "
            "Each unit supplies pv_kw, load_kw, price_buy_y_per_kwh, carbon_kg_per_kwh "
            "and forecast_windows (hours × 3 × 4, in the same channel order). "
            "See examples/synthetic_case.json."
        ),
    )
    parser.add_argument("case", type=Path, help="input case JSON")
    parser.add_argument("--output", required=True, type=Path, help="destination for final policy sets")
    parser.add_argument("--device", default="cuda", help="CUDA device for SAC, e.g., cuda:0")
    args = parser.parse_args()

    from src.hybrid.main_case import run_case, scenario_from_mapping

    try:
        case = json.loads(args.case.read_text(encoding="utf-8"))
        schedule = [scenario_from_mapping(unit) for unit in case["schedule"]]
        if args.output.exists():
            raise FileExistsError(f"choose a new output file: {args.output}")
        result = run_case(schedule, seed=case["seed"], device=args.device)
        payload = json.dumps(result, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Case error: {exc}\n")


if __name__ == "__main__":
    main()
