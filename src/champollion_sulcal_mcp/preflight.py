from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel

STAGE_SCRIPTS = {
    "morphologist": "generate_morphologist_graphs.py",
    "cortical_tiles": "run_cortical_tiles.py",
    "config": "generate_champollion_config.py",
    "embeddings": "generate_embeddings.py",
    "combine": "put_together_embeddings.py",
    "snapshots": "generate_snapshots.py",
}


class PipelineLocation(BaseModel):
    pipeline_dir: Path
    scripts_dir: Path
    has_champollion_submodule: bool
    has_cortical_tiles_submodule: bool
    python_exe: Path

    model_config = {"arbitrary_types_allowed": True}


def resolve_pipeline_dir() -> Path:
    env_val = os.environ.get("CHAMPOLLION_PIPELINE_DIR")
    if env_val:
        p = Path(env_val)
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"CHAMPOLLION_PIPELINE_DIR={env_val!r} does not exist or is not a directory."
        )
    # Sibling directory fallback
    this_file = Path(__file__).resolve()
    sibling = this_file.parents[3] / "champollion_pipeline"
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError(
        "Cannot locate champollion_pipeline directory. "
        "Set CHAMPOLLION_PIPELINE_DIR env var to its absolute path."
    )


def detect() -> PipelineLocation:
    pipeline_dir = resolve_pipeline_dir()
    scripts_dir = pipeline_dir / "src"

    # Prefer pixi-managed python; fall back to current interpreter
    pixi_python = pipeline_dir / ".pixi" / "envs" / "default" / "bin" / "python"
    python_exe = pixi_python if pixi_python.exists() else Path(sys.executable)

    return PipelineLocation(
        pipeline_dir=pipeline_dir,
        scripts_dir=scripts_dir,
        has_champollion_submodule=(pipeline_dir / "external" / "champollion_V1").is_dir(),
        has_cortical_tiles_submodule=(pipeline_dir / "external" / "cortical_tiles").is_dir(),
        python_exe=python_exe,
    )


def check_hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def summarize() -> dict:
    try:
        loc = detect()
        scripts = {name: (loc.scripts_dir / fname).exists() for name, fname in STAGE_SCRIPTS.items()}
        return {
            "ok": all(scripts.values()),
            "pipeline_dir": str(loc.pipeline_dir),
            "python_exe": str(loc.python_exe),
            "scripts": scripts,
            "submodules": {
                "champollion_V1": loc.has_champollion_submodule,
                "cortical_tiles": loc.has_cortical_tiles_submodule,
            },
            "hf_token_present": check_hf_token_present(),
        }
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e)}
