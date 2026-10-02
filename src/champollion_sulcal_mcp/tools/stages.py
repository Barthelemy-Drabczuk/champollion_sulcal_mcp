from __future__ import annotations

import os
from pathlib import Path

from fastmcp import Context
from fastmcp.exceptions import ToolError

from .. import preflight, runner
from ..roots import get_roots, validate_within_roots


def _require_absolute(path: str, name: str) -> None:
    if not Path(path).is_absolute():
        raise ToolError(f"{name} must be an absolute path, got: {path!r}")


def _job_result(state, stage: str) -> dict:
    return {"job_id": state.job_id, "status": state.status, "stage": stage,
            "log_path": state.log_path, "output_dir": state.output_dir}


def _build_env(*, pass_hf_token: bool = False) -> dict:
    env = os.environ.copy()
    if not pass_hf_token:
        env.pop("HF_TOKEN", None)
    # Use system CA bundle when the pixi env's cert store is incomplete (e.g. CEA proxy)
    if not env.get("SSL_CERT_FILE") and not env.get("REQUESTS_CA_BUNDLE"):
        system_ca = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.exists(system_ca):
            env["SSL_CERT_FILE"] = system_ca

    try:
        pipeline_dir = preflight.resolve_pipeline_dir()
        pixi_bin = pipeline_dir / ".pixi" / "envs" / "default" / "bin"

        # Inject BRAINVISA_SHARE when not set — required by AIMS/Anatomist to locate
        # nomenclature files and read .arg sulcal graphs correctly.
        if not env.get("BRAINVISA_SHARE"):
            brainvisa_share = pipeline_dir / ".pixi" / "envs" / "default" / "share"
            if brainvisa_share.is_dir():
                env["BRAINVISA_SHARE"] = str(brainvisa_share)

        # Prepend the pixi env bin to PATH so BrainVISA tools (e.g. VipSkeleton)
        # are found by subprocesses. If BRAINVISA is set, also prepend $BRAINVISA/bin.
        path_parts = []
        brainvisa_var = env.get("BRAINVISA")
        if brainvisa_var:
            brainvisa_bin = Path(brainvisa_var) / "bin"
            if brainvisa_bin.is_dir():
                path_parts.append(str(brainvisa_bin))
        if pixi_bin.is_dir():
            path_parts.append(str(pixi_bin))
        if path_parts:
            current_path = env.get("PATH", "")
            env["PATH"] = os.pathsep.join(path_parts + [current_path])

    except FileNotFoundError:
        pass

    return env


def _compute_training_derivatives_dir(base: Path, dataset: str) -> Path:
    """Return the champollion_V1 derivatives dir start_training defaults hang off.

    Args:
        base: roots[0] when the MCP client declares roots, else the
            champollion_pipeline root (loc.pipeline_dir).
        dataset: start_training's dataset argument.

    Returns:
        Path(base) / "data" / dataset / "derivatives" / "champollion_V1".

    Complexity: O(1).
    """
    return Path(base) / "data" / dataset / "derivatives" / "champollion_V1"


async def start_morphologist(
    input_dir: str,
    output_dir: str,
    parallel: bool = False,
    enable_sulcal_recognition: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Launch Stage 1: generate sulcal graphs with Morphologist from raw T1 MRI data."""
    _require_absolute(input_dir, "input_dir")
    _require_absolute(output_dir, "output_dir")
    roots = await get_roots(ctx)
    validate_within_roots(input_dir, roots, "input_dir")
    validate_within_roots(output_dir, roots, "output_dir")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "generate_morphologist_graphs.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [str(loc.python_exe), str(script), input_dir, output_dir]
    if parallel:
        argv.append("--parallel")
    if enable_sulcal_recognition:
        argv.append("--enable-sulcal-recognition")

    if ctx:
        await ctx.info(f"Launching morphologist stage: {input_dir} → {output_dir}")

    state = await runner.launch(
        stage="morphologist",
        argv=argv,
        output_dir=output_dir,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={"input_dir": input_dir, "output_dir": output_dir, "parallel": parallel},
    )
    return _job_result(state, "morphologist")


async def start_cortical_tiles(
    input_dir: str,
    output_dir: str,
    path_to_graph: str,
    path_sk_with_hull: str,
    sk_qc_path: str | None = None,
    njobs: int | None = None,
    masks: str | None = None,
    regions: list[str] | None = None,
    labelling_session: str | None = None,
    overwrite: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Launch Stage 2: extract 28 sulcal region crops with cortical_tiles."""
    _require_absolute(input_dir, "input_dir")
    _require_absolute(output_dir, "output_dir")
    roots = await get_roots(ctx)
    validate_within_roots(input_dir, roots, "input_dir")
    validate_within_roots(output_dir, roots, "output_dir")
    validate_within_roots(path_to_graph, roots, "path_to_graph")
    validate_within_roots(path_sk_with_hull, roots, "path_sk_with_hull")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "run_cortical_tiles.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [
        str(loc.python_exe), str(script),
        input_dir, output_dir,
        "--path_to_graph", path_to_graph,
        "--path_sk_with_hull", path_sk_with_hull,
    ]
    if sk_qc_path:
        argv += ["--sk_qc_path", sk_qc_path]
    if njobs is not None:
        argv += ["--njobs", str(njobs)]
    if masks:
        argv += ["--masks", masks]
    if regions:
        argv += ["--regions"] + regions
    if labelling_session:
        argv += ["--labelling_session", labelling_session]
    if overwrite:
        argv.append("--overwrite")

    if ctx:
        await ctx.info(f"Launching cortical_tiles stage: {input_dir} → {output_dir}")

    state = await runner.launch(
        stage="cortical_tiles",
        argv=argv,
        output_dir=output_dir,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={
            "input_dir": input_dir,
            "output_dir": output_dir,
            "path_to_graph": path_to_graph,
            "path_sk_with_hull": path_sk_with_hull,
        },
    )
    return _job_result(state, "cortical_tiles")


async def start_config(
    crop_path: str,
    dataset: str,
    champollion_loc: str | None = None,
    output: str | None = None,
    external_config: str | None = None,
    external_crops: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Launch Stage 3: generate Champollion dataset YAML configuration files.

    The configs root defaults to <D>/<dataset>/derivatives/champollion_V1/configs,
    where <D> is the parent of the <dataset> directory in crop_path; region YAMLs
    land at {configs root}/dataset/<dataset>/. `output` overrides the configs root;
    `external_config` overrides where the dataset_localization YAML is written
    (default {configs root}/dataset_localization/). Each is forwarded only when
    supplied, so the pipeline defaults apply otherwise.
    """
    _require_absolute(crop_path, "crop_path")
    roots = await get_roots(ctx)
    validate_within_roots(crop_path, roots, "crop_path")
    if output:
        validate_within_roots(output, roots, "output")

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "generate_champollion_config.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [str(loc.python_exe), str(script), crop_path, "--dataset", dataset]
    if champollion_loc:
        argv += ["--champollion_loc", champollion_loc]
    if output:
        argv += ["--output", output]
    if external_config:
        argv += ["--external-config", external_config]
    if external_crops:
        argv.append("--external_crops")

    output_dir = output or crop_path

    if ctx:
        await ctx.info(f"Launching config stage for dataset: {dataset}")

    state = await runner.launch(
        stage="config",
        argv=argv,
        output_dir=output_dir,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={"crop_path": crop_path, "dataset": dataset},
    )
    return _job_result(state, "config")


def _is_local_path(p: str) -> bool:
    return p.startswith("/") or p.startswith("./") or p.startswith("../")


async def start_embeddings(
    models_path: str,
    datasets_root: str,
    cpu: bool = False,
    overwrite: bool = False,
    masks: str | None = None,
    masks_version: str | None = None,
    output: str | None = None,
    subjects: str | None = None,
    regions: list[str] | None = None,
    run_cka: bool = False,
    cortical_version: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Launch Stage 4: compute sulcal embeddings across all 56 model folds (28 regions × 2 hemispheres)."""
    if _is_local_path(models_path):
        _require_absolute(models_path, "models_path")
        roots = await get_roots(ctx)
        validate_within_roots(models_path, roots, "models_path")
    else:
        roots = await get_roots(ctx)
    _require_absolute(datasets_root, "datasets_root")
    validate_within_roots(datasets_root, roots, "datasets_root")

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "generate_embeddings.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [str(loc.python_exe), str(script), models_path, datasets_root]

    if cpu:
        argv.append("--cpu")
    if overwrite:
        argv.append("--overwrite")
    if run_cka:
        argv.append("--run-cka")
    if masks:
        argv += ["--masks", masks]
    if masks_version:
        argv += ["--masks-version", masks_version]
    if output:
        argv += ["--output", output]
    if subjects:
        argv += ["--subjects", subjects]
    if regions:
        argv += ["--regions"] + regions
    if cortical_version:
        argv += ["--cortical_version", cortical_version]

    env = _build_env(pass_hf_token=True)

    if ctx:
        await ctx.info(f"Launching embeddings stage: models={models_path}, datasets_root={datasets_root}")

    state = await runner.launch(
        stage="embeddings",
        argv=argv,
        output_dir=datasets_root,
        cwd=str(loc.pipeline_dir),
        env=env,
        args_snapshot={"models_path": models_path, "datasets_root": datasets_root},
    )
    return _job_result(state, "embeddings")


async def start_combine(
    embeddings_source: str,
    output_path: str,
    ctx: Context | None = None,
) -> dict:
    """Launch Stage 5: collect all per-region embedding CSVs into a single output directory."""
    _require_absolute(embeddings_source, "embeddings_source")
    _require_absolute(output_path, "output_path")
    roots = await get_roots(ctx)
    validate_within_roots(embeddings_source, roots, "embeddings_source")
    validate_within_roots(output_path, roots, "output_path")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "put_together_embeddings.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [str(loc.python_exe), str(script), embeddings_source, "--output_path", output_path]

    if ctx:
        await ctx.info(f"Launching combine stage: {embeddings_source} → {output_path}")

    state = await runner.launch(
        stage="combine",
        argv=argv,
        output_dir=output_path,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={"embeddings_source": embeddings_source, "output_path": output_path},
    )
    return _job_result(state, "combine")


async def start_streaming(
    input_dir: str,
    output_dir: str,
    dataset: str,
    path_to_graph: str,
    path_sk_with_hull: str,
    bids: bool = False,
    n_workers: int = 0,
    worker_timeout: int = 7200,
    poll_interval: int = 10,
    sk_qc_path: str | None = None,
    models_path: str | None = None,
    dataset_localization: str = "local",
    datasets_root: str | None = None,
    short_name: str = "eval",
    embeddings_path: str = "champollion_V1",
    dry_run: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Launch scan-centric streaming pipeline: one worker per scan runs stages 2-4 in parallel.

    Each worker owns one ScanId and processes cortical_tiles → config → embeddings
    sequentially for its scan, using file-presence barriers between stages.
    Stage 5 (combine) runs once after all workers drain.

    Requires embeddings_only mode (training aggregates all subjects and cannot be parallelised per-scan).
    """
    _require_absolute(input_dir, "input_dir")
    _require_absolute(output_dir, "output_dir")
    roots = await get_roots(ctx)
    validate_within_roots(input_dir, roots, "input_dir")
    validate_within_roots(output_dir, roots, "output_dir")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "run_streaming.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [
        str(loc.python_exe), str(script),
        input_dir, output_dir,
        "--dataset", dataset,
        "--path-to-graph", path_to_graph,
        "--path-sk-with-hull", path_sk_with_hull,
        "--n-workers", str(n_workers),
        "--worker-timeout", str(worker_timeout),
        "--poll-interval", str(poll_interval),
        "--dataset-localization", dataset_localization,
        "--short-name", short_name,
        "--embeddings-path", embeddings_path,
    ]
    if bids:
        argv.append("--bids")
    if sk_qc_path:
        argv += ["--sk-qc-path", sk_qc_path]
    if models_path:
        argv += ["--models-path", models_path]
    if datasets_root:
        argv += ["--datasets-root", datasets_root]
    if dry_run:
        argv.append("--dry-run")

    if ctx:
        await ctx.info(
            f"Launching streaming pipeline: {input_dir} → {output_dir} "
            f"(n_workers={n_workers or 'auto'}, bids={bids})"
        )

    state = await runner.launch(
        stage="streaming",
        argv=argv,
        output_dir=output_dir,
        cwd=str(loc.pipeline_dir),
        env=_build_env(pass_hf_token=True),
        args_snapshot={
            "input_dir": input_dir,
            "output_dir": output_dir,
            "dataset": dataset,
            "n_workers": n_workers,
            "bids": bids,
            "dry_run": dry_run,
        },
    )
    return _job_result(state, "streaming")


async def start_training(
    dataset: str,
    region: str,
    mode: str = "encoder",
    output_dir: str | None = None,
    config_dir: str | None = None,
    njobs: int | None = None,
    cpu: bool = False,
    overwrite: bool = False,
    swf: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Launch encoder training: train a champollion_V1 self-supervised encoder for one sulcal region.

    Dataset configs must exist before calling this tool: run start_config first.
    When the MCP client declares roots, config_dir defaults to
    <roots[0]>/data/<dataset>/derivatives/champollion_V1/configs (roots[0] = the
    first declared root; same base as the output_dir default). Otherwise
    config_dir defaults to <pipeline>/data/<dataset>/derivatives/champollion_V1/configs
    (<pipeline> = the champollion_pipeline root). Pass config_dir when the configs
    live elsewhere (use <D>/<dataset>/derivatives/champollion_V1/configs) or when
    start_config was given an explicit `output` (use that configs root).
    """
    if swf:
        raise ToolError("swf=True is not supported: train_champollion has no soma-workflow mode (REQ-SWF-01).")
    if output_dir is not None:
        _require_absolute(output_dir, "output_dir")
    if config_dir is not None:
        _require_absolute(config_dir, "config_dir")
    roots = await get_roots(ctx)

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "train_champollion.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    default_base = roots[0] if roots else loc.pipeline_dir
    derivatives_dir = _compute_training_derivatives_dir(default_base, dataset)
    resolved_output_dir = output_dir or str(derivatives_dir / "models" / region)
    if config_dir is None and roots:
        config_dir = str(derivatives_dir / "configs")
    validate_within_roots(resolved_output_dir, roots, "output_dir")
    Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)

    argv = [
        str(loc.python_exe), str(script),
        "--dataset", dataset,
        "--region", region,
        "--mode", mode,
        "--output_dir", resolved_output_dir,
    ]
    if config_dir:
        argv += ["--config-dir", config_dir]
    if njobs is not None:
        argv += ["--njobs", str(njobs)]
    if cpu:
        argv.append("--cpu")
    if overwrite:
        argv.append("--overwrite")

    if ctx:
        await ctx.info(
            f"Launching training: dataset={dataset}, region={region}, mode={mode}, "
            f"output={resolved_output_dir}"
        )

    state = await runner.launch(
        stage="training",
        argv=argv,
        output_dir=resolved_output_dir,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={"dataset": dataset, "region": region, "mode": mode},
    )
    return _job_result(state, "training")


async def purge_subject(
    derivatives: str,
    subject: str,
    dry_run: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Remove all cortical_tiles derivatives for a single subject.

    Deletes per-subject NIfTI files (crops, labels, extremities, distbottom),
    per-subject subdirectories (skeletons/, foldlabels/, transforms/, distmaps/),
    and filters the subject's row from aggregated .npy arrays and their subject CSVs.

    Use dry_run=True to preview what would be deleted without modifying anything.
    """
    _require_absolute(derivatives, "derivatives")
    roots = await get_roots(ctx)
    validate_within_roots(derivatives, roots, "derivatives")

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "purge_subject.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [str(loc.python_exe), str(script), derivatives, "--subject", subject]
    if dry_run:
        argv.append("--dry-run")

    if ctx:
        await ctx.info(f"Purging subject '{subject}' from: {derivatives} (dry_run={dry_run})")

    state = await runner.launch(
        stage="purge_subject",
        argv=argv,
        output_dir=derivatives,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={"derivatives": derivatives, "subject": subject, "dry_run": dry_run},
    )
    return _job_result(state, "purge_subject")


async def prune_failed_subjects(
    output: str,
    qc: str,
    dry_run: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Remove cortical_tiles outputs for all subjects that failed QC.

    Reads a QC TSV/CSV file with 'participant_id' and 'qc' columns and deletes
    all files belonging to subjects with qc==0 or absent from the QC file.
    Equivalent to having run cortical_tiles with --sk_qc_path from the start.

    Use dry_run=True to preview what would be deleted without modifying anything.
    """
    _require_absolute(output, "output")
    _require_absolute(qc, "qc")
    roots = await get_roots(ctx)
    validate_within_roots(output, roots, "output")
    validate_within_roots(qc, roots, "qc")

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "prune_failed_subjects.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [str(loc.python_exe), str(script), output, "--qc", qc]
    if dry_run:
        argv.append("--dry-run")

    if ctx:
        await ctx.info(f"Pruning QC-failing subjects from: {output} (qc={qc}, dry_run={dry_run})")

    state = await runner.launch(
        stage="prune_failed_subjects",
        argv=argv,
        output_dir=output,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={"output": output, "qc": qc, "dry_run": dry_run},
    )
    return _job_result(state, "prune_failed_subjects")


async def start_snapshots(
    output_dir: str,
    morphologist_dir: str | None = None,
    embeddings_dir: str | None = None,
    cortical_tiles_dir: str | None = None,
    subject: str | None = None,
    acquisition: str | None = None,
    sulcal_only: bool = False,
    tiles_only: bool = False,
    umap_only: bool = False,
    umap_region: str | None = None,
    champollion_data_root: str | None = None,
    reference_data_dir: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Launch Stage 6: render sulcal graph meshes, cortical tile masks, and UMAP scatter plots.

    UMAP plots need both ``embeddings_dir`` and ``reference_data_dir``; there is no default for
    ``reference_data_dir`` (not shipped with champollion_pipeline).
    """
    _require_absolute(output_dir, "output_dir")
    roots = await get_roots(ctx)
    validate_within_roots(output_dir, roots, "output_dir")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        loc = preflight.detect()
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e

    script = loc.scripts_dir / "generate_snapshots.py"
    if not script.exists():
        raise ToolError(f"Script not found: {script}")

    argv = [str(loc.python_exe), str(script), "--output_dir", output_dir]
    if morphologist_dir:
        argv += ["--morphologist_dir", morphologist_dir]
    if embeddings_dir:
        argv += ["--embeddings_dir", embeddings_dir]
    if cortical_tiles_dir:
        argv += ["--cortical_tiles_dir", cortical_tiles_dir]
    if subject:
        argv += ["--subject", subject]
    if acquisition:
        argv += ["--acquisition", acquisition]
    if sulcal_only:
        argv.append("--sulcal-only")
    if tiles_only:
        argv.append("--tiles-only")
    if umap_only:
        argv.append("--umap-only")
    if umap_region:
        argv += ["--umap_region", umap_region]
    if champollion_data_root:
        argv += ["--champollion_data_root", champollion_data_root]
    if reference_data_dir:
        argv += ["--reference_data_dir", reference_data_dir]

    if ctx:
        await ctx.info(f"Launching snapshots stage → {output_dir}")

    state = await runner.launch(
        stage="snapshots",
        argv=argv,
        output_dir=output_dir,
        cwd=str(loc.pipeline_dir),
        env=_build_env(),
        args_snapshot={
            "output_dir": output_dir,
            "morphologist_dir": morphologist_dir,
            "cortical_tiles_dir": cortical_tiles_dir,
            "embeddings_dir": embeddings_dir,
            "subject": subject,
            "champollion_data_root": champollion_data_root,
            "reference_data_dir": reference_data_dir,
        },
    )
    return _job_result(state, "snapshots")
