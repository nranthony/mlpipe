"""mlpipe CLI — the single write path. Every execution lands in manifests/.

    mlpipe run --pipeline demo --from make_data --to summarize --set k.v=1 --tag exp
"""

from __future__ import annotations

import importlib
from pathlib import Path

import typer
import yaml

from mlpipe.core.pipeline import Pipeline
from mlpipe.core.store import LocalCasStore

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    pipeline: str = typer.Option("demo", help="pipeline module in mlpipe.steps"),
    from_step: str = typer.Option(None, "--from", help="first step to (re)run"),
    to_step: str = typer.Option(None, "--to", help="last step to run"),
    set_: list[str] = typer.Option(None, "--set", help="config override key.path=value"),
    tag: str = typer.Option(None, help="experiment tag recorded in manifests"),
    config: Path = typer.Option(None, help="YAML config file (else module default)"),
    store_dir: Path = typer.Option(Path("store"), help="artifact store root"),
    manifests_dir: Path = typer.Option(Path("manifests"), help="manifest log root"),
) -> None:
    module = importlib.import_module(f"mlpipe.steps.{pipeline}")
    cfg = yaml.safe_load(config.read_text()) if config else getattr(module, "DEFAULT_CONFIG", {})
    pipe = Pipeline(module.build_steps(), LocalCasStore(store_dir), manifests_dir, cfg)
    result = pipe.run(from_step, to_step, overrides=set_, tag=tag)
    for m in result["manifests"]:
        typer.echo(f"{m['step']:<20} {m['status']:<8} sig={m['signature'][:8]} "
                   f"outputs={ {k: h[:8] for k, h in m['outputs'].items()} }")


@app.command()
def lineage(
    content_hash: str,
    manifests_dir: Path = typer.Option(Path("manifests")),
) -> None:
    from mlpipe.core.manifest import ManifestLog

    for m in ManifestLog(manifests_dir).lineage(content_hash):
        typer.echo(f"{m['step']:<20} sig={m['signature'][:8]} "
                   f"commit={(m['git_commit'] or '?')[:8]} inputs={list(m['inputs'])}")


if __name__ == "__main__":
    app()
