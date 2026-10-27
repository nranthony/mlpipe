"""work/0006 step 15 (ADR-0008): the Copier template renders without running code,
pins mlpipe to its own tag, and the rendered pipeline passes its own smoke test."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from mlpipe import plugins

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = {"project_name": "churn-model", "description": "Churn scoring"}
TAG = "v9.9.9"


def test_template_runs_no_code():
    spec = yaml.safe_load((ROOT / "copier.yml").read_text())
    assert not {"_tasks", "_migrations", "_jinja_extensions"} & set(spec)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Rendered from a throwaway repo holding only the template, tagged like a
    release, so the result never depends on this checkout's state."""
    copier = pytest.importorskip("copier")
    src = tmp_path_factory.mktemp("tpl")
    shutil.copy(ROOT / "copier.yml", src)
    shutil.copytree(ROOT / "template", src / "template")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "t"], ["tag", TAG]):
        subprocess.run(["git", *args], cwd=src, env=env, check=True)
    dst = tmp_path_factory.mktemp("host")
    copier.run_copy(str(src), dst, data=ANSWERS, defaults=True, unsafe=False, quiet=True)
    return dst


def test_rendered_layout_and_pin(rendered):
    for path in ("AGENTS.md", "CLAUDE.md", ".copier-answers.yml", ".gitignore", "DESIGN.md",
                 "goals/STATUS.md", "work/README.md", "docs/adr/0000-template.md",
                 ".claude/skills/template-sync/SKILL.md", "src/churn_model/pipeline.py",
                 "tests/test_smoke.py"):
        assert (rendered / path).is_file(), path
    assert (rendered / "CLAUDE.md").read_text() == "# CLAUDE.md\n\n@AGENTS.md\n"
    py = tomllib.loads((rendered / "pyproject.toml").read_text())
    assert py["project"]["entry-points"]["mlpipe.pipelines"] == {"churn-model": "churn_model.pipeline"}
    assert yaml.safe_load((rendered / ".copier-answers.yml").read_text())["_commit"] == TAG
    assert py["tool"]["uv"]["sources"]["mlpipe"]["tag"] == TAG  # one tag: library + template
    assert "{{" not in (rendered / "AGENTS.md").read_text()


def test_rendered_smoke_test_passes(rendered, tmp_path, monkeypatch):
    """The rendered repo's own smoke test, against this checkout's mlpipe. Its entry
    point needs an install, so the pipeline is registered as a builtin here."""
    monkeypatch.syspath_prepend(str(rendered / "src"))
    monkeypatch.setitem(plugins.BUILTIN_PIPELINES, "churn-model", "churn_model.pipeline")
    spec = importlib.util.spec_from_file_location("rendered_smoke", rendered / "tests/test_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.test_smoke_run_then_cache_hit(tmp_path)
    finally:
        for name in [m for m in sys.modules if m.startswith("churn_model")]:
            del sys.modules[name]
