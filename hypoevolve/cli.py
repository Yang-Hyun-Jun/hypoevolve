from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from elg import hypothesis_from_dict, render_pretty, render_tree
from hypoevolve.config import ConfigError, HypoEvolveConfig, load_config
from hypoevolve.controller import HypoEvolveController
from hypoevolve.llm import LLMClient
from hypoevolve.parser import ParseError, parse_hypothesis_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hypoevolve", description="HypoEvolve MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a HypoEvolve iteration loop")
    run_parser.add_argument("hypothesis", help="Natural-language hypothesis input")
    run_parser.add_argument("--config", default="hypoevolve.yaml", help="Path to hypoevolve.yaml")
    run_parser.add_argument("--workers", type=int, default=None, help="Override local worker count")

    render_parser = subparsers.add_parser("render", help="Render a natural-language hypothesis via LLM parser")
    render_parser.add_argument("hypothesis", help="Natural-language hypothesis input")
    render_parser.add_argument("--config", default="hypoevolve.yaml", help="Path to hypoevolve.yaml")
    render_parser.add_argument("--tree", action="store_true", help="Render as ASCII tree instead of pretty form")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved best/checkpoint JSON file")
    inspect_parser.add_argument("path", help="Path to best.json or checkpoint.json")

    doctor_parser = subparsers.add_parser("doctor", help="Show environment and config diagnostics")
    doctor_parser.add_argument("--config", default="hypoevolve.yaml", help="Path to hypoevolve.yaml")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        return _run_command(args)
    if args.command == "render":
        return _render_command(args)
    if args.command == "inspect":
        return _inspect_command(args)
    if args.command == "doctor":
        return _doctor_command(args)
    return 1


def _run_command(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if Path(args.config).exists() else HypoEvolveConfig()
        if args.workers is not None:
            config.workers.count = args.workers
            config.workers.enabled = args.workers > 1
        controller = HypoEvolveController(config)
        result = controller.run(args.hypothesis)
        print(f"Run directory: {result.run_dir}")
        print("Best hypothesis:")
        print(render_pretty(result.best_hypothesis))
        print("Best metrics:")
        print(json.dumps(result.best_metrics, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ConfigError, ParseError) as exc:
        print(f"Error: {exc}")
        if getattr(exc, "errors", None):
            for item in exc.errors:
                print(f"  - {item}")
        return 1


def _render_command(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config) if Path(args.config).exists() else HypoEvolveConfig()
        hypothesis = parse_hypothesis_text(
            args.hypothesis,
            llm=LLMClient(config.llm),
            retries=config.parser.retries,
        )
        if args.tree:
            print(render_tree(hypothesis))
        else:
            print(render_pretty(hypothesis))
        return 0
    except (ConfigError, ParseError) as exc:
        print(f"Error: {exc}")
        if getattr(exc, "errors", None):
            for item in exc.errors:
                print(f"  - {item}")
        return 1


def _inspect_command(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    if "hypothesis" in payload:
        hypothesis = hypothesis_from_dict(payload["hypothesis"])
        print(render_pretty(hypothesis))
        if "metrics" in payload:
            print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    elif "best_hypothesis" in payload and payload["best_hypothesis"] is not None:
        hypothesis = hypothesis_from_dict(payload["best_hypothesis"])
        print(render_pretty(hypothesis))
        if "best_metrics" in payload:
            print(json.dumps(payload["best_metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _doctor_command(args: argparse.Namespace) -> int:
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    config_path = Path(args.config)
    print(f"config_exists: {config_path.exists()}")
    if config_path.exists():
        try:
            config = load_config(config_path)
            print(f"config_ok: true")
            print(f"archive_top_k: {config.archive.top_k}")
            print(f"iterations: {config.search.iterations}")
            print(f"output_base_dir: {config.output.base_dir}")
            print(f"workers_enabled: {config.workers.enabled}")
            print(f"worker_count: {config.workers.count}")
        except Exception as exc:  # noqa: BLE001
            print(f"config_ok: false")
            print(f"config_error: {exc}")
            return 1
    else:
        print("config_ok: using defaults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
