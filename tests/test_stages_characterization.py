"""Characterization tests for tools/stages.py (TASK-042, REQ-MCP-COV-02..24).

Each test pins the command, environment or refusal a stage tool produces today.
runner.launch is replaced by a recorder, so no pipeline script is executed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.tools import stages

pytestmark = pytest.mark.unit

_GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1"
_SK_HULL = "t1mri/default_acquisition/default_analysis/segmentation"

PUBLIC_TOOLS = [
    "start_morphologist",
    "start_cortical_tiles",
    "start_config",
    "start_embeddings",
    "start_combine",
    "start_streaming",
    "start_training",
    "purge_subject",
    "prune_failed_subjects",
    "start_snapshots",
]

HF_STRIPPING_TOOLS = [
    "start_morphologist",
    "start_cortical_tiles",
    "start_config",
    "start_combine",
    "start_training",
    "start_snapshots",
    "purge_subject",
    "prune_failed_subjects",
]


def _tool_kwargs(tool: str, base: Path) -> dict:
    """Minimal valid arguments for `tool`, every path absolute and under `base`."""
    return {
        "start_morphologist": {"input_dir": str(base / "in"), "output_dir": str(base / "out")},
        "start_cortical_tiles": {
            "input_dir": str(base / "in"),
            "output_dir": str(base / "out"),
            "path_to_graph": _GRAPH,
            "path_sk_with_hull": _SK_HULL,
        },
        "start_config": {"crop_path": str(base / "crops"), "dataset": "ds"},
        "start_embeddings": {"models_path": str(base / "models"), "datasets_root": str(base / "ds")},
        "start_combine": {"embeddings_source": str(base / "emb"), "output_path": str(base / "combined")},
        "start_streaming": {
            "input_dir": str(base / "in"),
            "output_dir": str(base / "out"),
            "dataset": "ds",
            "path_to_graph": _GRAPH,
            "path_sk_with_hull": _SK_HULL,
        },
        "start_training": {"dataset": "ds", "region": "S.C.-sylv.", "output_dir": str(base / "models" / "r")},
        "purge_subject": {"derivatives": str(base / "deriv"), "subject": "sub-01"},
        "prune_failed_subjects": {"output": str(base / "deriv"), "qc": str(base / "qc.tsv")},
        "start_snapshots": {"output_dir": str(base / "snap")},
    }[tool]


def _contains_run(argv: list[str], tokens: list[str]) -> bool:
    """True when `tokens` appear in `argv` as one contiguous run."""
    n = len(tokens)
    return any(argv[i : i + n] == tokens for i in range(len(argv) - n + 1))


def _script_args(argv: list[str]) -> list[str]:
    """argv after `<python> <script>`."""
    return argv[2:]


# --- REQ-MCP-COV-02 / -03: start_embeddings command ---


class TestEmbeddingsArgv:
    async def test_models_path_and_datasets_root_are_first_two_script_args(self, pipeline_dir, tmp_path, launches):
        """REQ-MCP-COV-02: models_path then datasets_root are the first two script arguments."""
        kwargs = _tool_kwargs("start_embeddings", tmp_path)
        await stages.start_embeddings(**kwargs)
        argv = launches[0]["argv"]
        assert Path(argv[1]).name == "generate_embeddings.py"
        assert _script_args(argv)[:2] == [kwargs["models_path"], kwargs["datasets_root"]]

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("cpu", True, ["--cpu"]),
            ("overwrite", True, ["--overwrite"]),
            ("run_cka", True, ["--run-cka"]),
            ("masks", "sk_masks", ["--masks", "sk_masks"]),
            ("masks_version", "v2", ["--masks-version", "v2"]),
            ("output", "/abs/emb_out", ["--output", "/abs/emb_out"]),
            ("subjects", "/abs/subjects.csv", ["--subjects", "/abs/subjects.csv"]),
            ("regions", ["S.C.-sylv.", "F.I.P."], ["--regions", "S.C.-sylv.", "F.I.P."]),
            ("cortical_version", "2025", ["--cortical_version", "2025"]),
        ],
    )
    async def test_forwards_optional_argument(self, pipeline_dir, tmp_path, launches, name, value, expected):
        """REQ-MCP-COV-03: a supplied optional argument becomes its documented flag (plus value)."""
        await stages.start_embeddings(**_tool_kwargs("start_embeddings", tmp_path), **{name: value})
        assert _contains_run(launches[0]["argv"], expected), launches[0]["argv"]

    async def test_omits_optional_flags_by_default(self, pipeline_dir, tmp_path, launches):
        """REQ-MCP-COV-03: with no optional arguments, only the two positionals follow the script."""
        await stages.start_embeddings(**_tool_kwargs("start_embeddings", tmp_path))
        assert len(_script_args(launches[0]["argv"])) == 2


# --- REQ-MCP-COV-04 / -05 / -06: HF_TOKEN handling ---


class TestHfToken:
    async def test_embeddings_passes_hf_token(self, pipeline_dir, tmp_path, launches, monkeypatch):
        """REQ-MCP-COV-04: start_embeddings keeps HF_TOKEN in the subprocess environment."""
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        await stages.start_embeddings(**_tool_kwargs("start_embeddings", tmp_path))
        assert launches[0]["env"].get("HF_TOKEN") == "hf_secret"

    async def test_streaming_passes_hf_token(self, pipeline_dir, tmp_path, launches, monkeypatch):
        """REQ-MCP-COV-05: start_streaming keeps HF_TOKEN in the subprocess environment."""
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        await stages.start_streaming(**_tool_kwargs("start_streaming", tmp_path))
        assert launches[0]["env"].get("HF_TOKEN") == "hf_secret"

    @pytest.mark.parametrize("tool", HF_STRIPPING_TOOLS)
    async def test_tool_strips_hf_token(self, pipeline_dir, tmp_path, launches, monkeypatch, tool):
        """REQ-MCP-COV-06: the listed tools launch without HF_TOKEN in the subprocess environment."""
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path))
        assert "HF_TOKEN" not in launches[0]["env"]


# --- REQ-MCP-COV-07 / -08: start_embeddings models_path validation ---


class TestEmbeddingsModelsPath:
    @pytest.mark.parametrize("models_path", ["./models", "../models"])
    async def test_relative_local_models_path_rejected(self, pipeline_dir, tmp_path, launches, models_path):
        """REQ-MCP-COV-07: a `./`/`../` models_path raises a ToolError naming models_path; no job."""
        with pytest.raises(ToolError, match="models_path"):
            await stages.start_embeddings(models_path=models_path, datasets_root=str(tmp_path / "ds"))
        assert launches == []

    async def test_hub_models_path_skips_root_check(self, pipeline_dir, mock_roots, launches):
        """REQ-MCP-COV-08: a Hub-style models_path is launched even though it lies outside declared roots."""
        ctx, root = mock_roots
        await stages.start_embeddings(models_path="neurospin/Champollion_V1", datasets_root=str(root / "ds"), ctx=ctx)
        assert _script_args(launches[0]["argv"])[0] == "neurospin/Champollion_V1"


# --- REQ-MCP-COV-09: relative path refusal ---


class TestRelativePathRefusal:
    @pytest.mark.parametrize(
        ("tool", "param"),
        [
            ("purge_subject", "derivatives"),
            ("prune_failed_subjects", "output"),
            ("prune_failed_subjects", "qc"),
            ("start_streaming", "input_dir"),
            ("start_streaming", "output_dir"),
            ("start_combine", "embeddings_source"),
            ("start_combine", "output_path"),
            ("start_embeddings", "datasets_root"),
        ],
    )
    async def test_relative_path_rejected(self, pipeline_dir, tmp_path, launches, tool, param):
        """REQ-MCP-COV-09: a relative value raises a ToolError naming the parameter; no job."""
        kwargs = _tool_kwargs(tool, tmp_path)
        kwargs[param] = "relative/path"
        with pytest.raises(ToolError, match=param):
            await getattr(stages, tool)(**kwargs)
        assert launches == []


# --- REQ-MCP-COV-10: start_combine command ---


class TestCombineArgv:
    async def test_argv_shape(self, pipeline_dir, tmp_path, launches):
        """REQ-MCP-COV-10: `<embeddings_source> --output_path <output_path>`."""
        kwargs = _tool_kwargs("start_combine", tmp_path)
        await stages.start_combine(**kwargs)
        argv = launches[0]["argv"]
        assert Path(argv[1]).name == "put_together_embeddings.py"
        assert _script_args(argv) == [kwargs["embeddings_source"], "--output_path", kwargs["output_path"]]


# --- REQ-MCP-COV-11 / -12: start_streaming command ---


class TestStreamingArgv:
    @pytest.mark.parametrize(
        "expected",
        [
            ["--n-workers", "0"],
            ["--worker-timeout", "7200"],
            ["--poll-interval", "10"],
            ["--dataset-localization", "local"],
            ["--short-name", "eval"],
            ["--embeddings-path", "champollion_V1"],
        ],
    )
    async def test_defaults(self, pipeline_dir, tmp_path, launches, expected):
        """REQ-MCP-COV-11: without optional arguments, the documented defaults are passed."""
        await stages.start_streaming(**_tool_kwargs("start_streaming", tmp_path))
        assert _contains_run(launches[0]["argv"], expected), launches[0]["argv"]

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("bids", True, ["--bids"]),
            ("sk_qc_path", "/abs/qc.tsv", ["--sk-qc-path", "/abs/qc.tsv"]),
            ("models_path", "/abs/models", ["--models-path", "/abs/models"]),
            ("datasets_root", "/abs/datasets", ["--datasets-root", "/abs/datasets"]),
            ("dry_run", True, ["--dry-run"]),
        ],
    )
    async def test_forwards_optional_argument(self, pipeline_dir, tmp_path, launches, name, value, expected):
        """REQ-MCP-COV-12: a supplied optional argument becomes its documented flag (plus value)."""
        await stages.start_streaming(**_tool_kwargs("start_streaming", tmp_path), **{name: value})
        assert _contains_run(launches[0]["argv"], expected), launches[0]["argv"]


# --- REQ-MCP-COV-13 / -14: purge_subject and prune_failed_subjects commands ---


class TestPurgeSubjectArgv:
    @pytest.mark.parametrize("dry_run", [False, True])
    async def test_argv(self, pipeline_dir, tmp_path, launches, dry_run):
        """REQ-MCP-COV-13: `<derivatives> --subject <subject>` plus `--dry-run` only when dry_run."""
        kwargs = _tool_kwargs("purge_subject", tmp_path)
        await stages.purge_subject(**kwargs, dry_run=dry_run)
        expected = [kwargs["derivatives"], "--subject", "sub-01"] + (["--dry-run"] if dry_run else [])
        assert _script_args(launches[0]["argv"]) == expected


class TestPruneFailedSubjectsArgv:
    @pytest.mark.parametrize("dry_run", [False, True])
    async def test_argv(self, pipeline_dir, tmp_path, launches, dry_run):
        """REQ-MCP-COV-14: `<output> --qc <qc>` plus `--dry-run` only when dry_run."""
        kwargs = _tool_kwargs("prune_failed_subjects", tmp_path)
        await stages.prune_failed_subjects(**kwargs, dry_run=dry_run)
        expected = [kwargs["output"], "--qc", kwargs["qc"]] + (["--dry-run"] if dry_run else [])
        assert _script_args(launches[0]["argv"]) == expected


# --- REQ-MCP-COV-15: start_snapshots optional flags ---


class TestSnapshotsArgv:
    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("morphologist_dir", "/abs/morpho", ["--morphologist_dir", "/abs/morpho"]),
            ("embeddings_dir", "/abs/emb", ["--embeddings_dir", "/abs/emb"]),
            ("cortical_tiles_dir", "/abs/tiles", ["--cortical_tiles_dir", "/abs/tiles"]),
            ("subject", "sub-01", ["--subject", "sub-01"]),
            ("acquisition", "default_acquisition", ["--acquisition", "default_acquisition"]),
            ("umap_region", "S.C.-sylv.", ["--umap_region", "S.C.-sylv."]),
            ("champollion_data_root", "/abs/champ", ["--champollion_data_root", "/abs/champ"]),
            ("sulcal_only", True, ["--sulcal-only"]),
            ("tiles_only", True, ["--tiles-only"]),
            ("umap_only", True, ["--umap-only"]),
        ],
    )
    async def test_forwards_optional_argument(self, pipeline_dir, tmp_path, launches, name, value, expected):
        """REQ-MCP-COV-15: a supplied optional argument becomes its documented flag (plus value)."""
        await stages.start_snapshots(**_tool_kwargs("start_snapshots", tmp_path), **{name: value})
        assert _contains_run(launches[0]["argv"], expected), launches[0]["argv"]


# --- REQ-MCP-COV-16..19: remaining optional flags ---


class TestTrainingArgv:
    async def test_forwards_njobs_cpu_overwrite(self, pipeline_dir, tmp_path, launches):
        """REQ-MCP-COV-16: njobs, cpu and overwrite reach train_champollion.py."""
        await stages.start_training(**_tool_kwargs("start_training", tmp_path), njobs=4, cpu=True, overwrite=True)
        argv = launches[0]["argv"]
        assert _contains_run(argv, ["--njobs", "4"])
        assert "--cpu" in argv
        assert "--overwrite" in argv


class TestCorticalTilesArgv:
    async def test_forwards_sk_qc_path_masks_regions(self, pipeline_dir, tmp_path, launches):
        """REQ-MCP-COV-17: sk_qc_path, masks and regions reach run_cortical_tiles.py."""
        await stages.start_cortical_tiles(
            **_tool_kwargs("start_cortical_tiles", tmp_path),
            sk_qc_path="/abs/qc.tsv",
            masks="sk_masks",
            regions=["S.C.-sylv.", "F.I.P."],
        )
        argv = launches[0]["argv"]
        assert _contains_run(argv, ["--sk_qc_path", "/abs/qc.tsv"])
        assert _contains_run(argv, ["--masks", "sk_masks"])
        assert _contains_run(argv, ["--regions", "S.C.-sylv.", "F.I.P."])


class TestMorphologistArgv:
    @pytest.mark.parametrize("enabled", [False, True])
    async def test_enable_sulcal_recognition_flag(self, pipeline_dir, tmp_path, launches, enabled):
        """REQ-MCP-COV-18: `--enable-sulcal-recognition` appears exactly when requested."""
        await stages.start_morphologist(
            **_tool_kwargs("start_morphologist", tmp_path), enable_sulcal_recognition=enabled
        )
        assert ("--enable-sulcal-recognition" in launches[0]["argv"]) is enabled


class TestConfigArgv:
    async def test_forwards_champollion_loc_and_external_crops(self, pipeline_dir, tmp_path, launches):
        """REQ-MCP-COV-19: champollion_loc and external_crops reach generate_champollion_config.py."""
        await stages.start_config(
            **_tool_kwargs("start_config", tmp_path), champollion_loc="/abs/champollion_V1", external_crops=True
        )
        argv = launches[0]["argv"]
        assert _contains_run(argv, ["--champollion_loc", "/abs/champollion_V1"])
        assert "--external_crops" in argv


# --- REQ-MCP-COV-20 / -21: pipeline location failures ---


class TestMissingScript:
    @pytest.mark.parametrize("tool", PUBLIC_TOOLS)
    async def test_tool_raises_script_not_found(self, tmp_path, monkeypatch, launches, tool):
        """REQ-MCP-COV-20: an absent stage script raises a ToolError "Script not found"; no job."""
        pipeline = tmp_path / "champollion_pipeline"
        (pipeline / "src" / "champollion_pipeline").mkdir(parents=True)
        monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(pipeline))
        with pytest.raises(ToolError, match="Script not found"):
            await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path))
        assert launches == []


class TestMissingPipelineDir:
    @pytest.mark.parametrize("tool", PUBLIC_TOOLS)
    async def test_tool_raises_tool_error(self, tmp_path, monkeypatch, launches, tool):
        """REQ-MCP-COV-21: a nonexistent CHAMPOLLION_PIPELINE_DIR raises a ToolError; no job."""
        monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(tmp_path / "does_not_exist"))
        with pytest.raises(ToolError):
            await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path))
        assert launches == []


# --- REQ-MCP-COV-22..24: subprocess environment built for BrainVISA tools ---


class TestBuildEnv:
    @pytest.mark.parametrize("tool", ["start_morphologist", "start_embeddings"])
    async def test_brainvisa_share_injected(self, pipeline_dir, tmp_path, launches, monkeypatch, tool):
        """REQ-MCP-COV-22: BRAINVISA_SHARE defaults to <pipeline>/.pixi/envs/default/share."""
        share = pipeline_dir / ".pixi" / "envs" / "default" / "share"
        share.mkdir(parents=True)
        monkeypatch.delenv("BRAINVISA_SHARE", raising=False)
        await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path))
        assert launches[0]["env"]["BRAINVISA_SHARE"] == str(share)

    @pytest.mark.parametrize("tool", ["start_morphologist", "start_embeddings"])
    async def test_pixi_bin_prepended_to_path(self, pipeline_dir, tmp_path, launches, monkeypatch, tool):
        """REQ-MCP-COV-23: <pipeline>/.pixi/envs/default/bin is put in front of PATH."""
        pixi_bin = pipeline_dir / ".pixi" / "envs" / "default" / "bin"
        pixi_bin.mkdir(parents=True)
        monkeypatch.delenv("BRAINVISA", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin")
        await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path))
        assert launches[0]["env"]["PATH"].split(os.pathsep) == [str(pixi_bin), "/usr/bin"]

    async def test_brainvisa_bin_precedes_pixi_bin(self, pipeline_dir, tmp_path, launches, monkeypatch):
        """REQ-MCP-COV-24: $BRAINVISA/bin comes first in PATH, ahead of the pixi env bin."""
        pixi_bin = pipeline_dir / ".pixi" / "envs" / "default" / "bin"
        pixi_bin.mkdir(parents=True)
        brainvisa = tmp_path / "brainvisa"
        (brainvisa / "bin").mkdir(parents=True)
        monkeypatch.setenv("BRAINVISA", str(brainvisa))
        monkeypatch.setenv("PATH", "/usr/bin")
        await stages.start_morphologist(**_tool_kwargs("start_morphologist", tmp_path))
        assert launches[0]["env"]["PATH"].split(os.pathsep) == [str(brainvisa / "bin"), str(pixi_bin), "/usr/bin"]
