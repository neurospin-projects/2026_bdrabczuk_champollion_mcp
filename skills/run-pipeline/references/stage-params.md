# Stage Parameters Reference

Detailed parameter guide for each `start_<stage>` tool. Use this when running stages individually rather than via `start_pipeline`.

---

## Stage 1 — `start_morphologist`

Calls `morphologist-cli` to generate sulcal graphs (`.arg` files) from T1 NIfTI images.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `input_dir` | str | yes | Directory of T1 NIfTI files (`.nii.gz`) |
| `output_dir` | str | yes | The dataset root (e.g. `./data/{dataset}/`); `morphologist-cli` writes to `derivatives/morphologist-{version}/subjects/` under it |
| `parallel` | bool | no | Enable soma-workflow parallel scheduling (Neurospin cluster) |
| `enable_sulcal_recognition` | bool | no | Run sulcal labeling (slower, not required for embeddings) |
| `log` | bool | no | Pass a bare `--log` so the stage writes its run log to its default `.log` folder (default: true) |

Output: `{output_dir}/derivatives/morphologist-{version}/subjects/{subject}/`

---

## Stage 2 — `start_cortical_tiles`

Extracts 28 standardized sulcal region crops from Morphologist's graphs.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `input_dir` | str | yes | Morphologist `subjects/` dir (read-only safe) |
| `output_dir` | str | yes | The derivatives dir (`<dataset root>/derivatives/`); writes `cortical_tiles-{YEAR}/crops/{masks}/2mm/` (`{masks}` defaults to `canonical_25`) |
| `path_to_graph` | str | yes | Relative path to `.arg` graph from subject dir. Supports `*` wildcards. E.g. `t1mri/default_acquisition/default_analysis/folds/3.1` |
| `path_sk_with_hull` | str | yes | Relative path to skeleton dir. E.g. `t1mri/default_acquisition/default_analysis/segmentation` |
| `sk_qc_path` | str | no | TSV file with `participant_id` and `qc` columns for filtering |
| `njobs` | int | no | CPU cores (default: auto) |
| `masks` | str | no | Mask version tag override (e.g. `canonical_25`) |
| `regions` | list[str] | no | Restrict to a subset of the 28 sulcal regions (space-separated in CLI; list here) |
| `labelling_session` | str | no | Morphologist labelling session whose labelled graphs locate the ventricle for whole-brain removal (default: `deepcnn_session_auto`) |
| `overwrite` | bool | no | Re-generate crops even if they already exist for this mask version |
| `log` | bool | no | Pass a bare `--log` so the stage writes its run log to its default `.log` folder (default: true) |

Output: `{output_dir}/cortical_tiles-{YEAR}/crops/{masks}/2mm/` — 28 region folders (`{masks}` = the `masks` parameter, default `canonical_25`).

> **CLI-only flags (not exposed via MCP):** `--input-types` (default `skeleton foldlabel`), `--with-distbottom` (opt-in distbottom generation), `--skip-distbottom` (deprecated no-op; distbottom is off by default). Do not attempt to pass these through the MCP tool.

**Default `path_to_graph` by mode:**
- Non-BIDS: `t1mri/default_acquisition/default_analysis/folds/3.1`
- BIDS (`bids=True`): `t1mri/default_acquisition/0/folds/3.1`
- Wildcard fallback: `t1mri/default_acquisition/*/folds/3.1`

---

## Stage 3 — `start_config`

Generates Champollion dataset YAML configuration files (`reference.yaml`, `local.yaml`).

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `crop_path` | str | yes | Path to the `crops/{masks}/2mm/` directory from stage 2 (default mask version `canonical_25`) |
| `dataset` | str | yes | Short dataset name (e.g. `TESTXX`) |
| `output` | str | no | Absolute path to the configs root; must be inside a declared MCP root when roots are declared. Region YAMLs land at `{output}/dataset/{dataset}/`. Default: `<D>/<dataset>/derivatives/champollion_V1/configs` (`<D>` = parent of the `<dataset>` directory in `crop_path`) |
| `champollion_loc` | str | no | Override path to champollion_V1 (default: `external/champollion_V1`) |
| `external_config` | str | no | Absolute path (directory or file) where the `dataset_localization` YAML is written; must be inside a declared MCP root when roots are declared. Default: `{configs root}/dataset_localization/`. Use for read-only containers |
| `external_crops` | bool | no | Set when crops are outside the pipeline dir |
| `masks` | str | no | Mask version tag forwarded as `--masks` (e.g. `canonical_corrected_26_1`); must match the stage-2 masks. Default: the script's `canonical_25` |
| `log` | bool | no | Pass a bare `--log` so the stage writes its run log to its default `.log` folder (default: true) |

Output: `{configs root}/dataset/{dataset}/` (`reference.yaml` + one YAML per region) and `{configs root}/dataset_localization/` (unless `external_config` is set). `start_embeddings` (stage 4) does not consume these files (each model folder carries its own `.hydra/config.yaml`); they feed `start_training` via `config_dir`.

---

## Training — `start_training`

Trains a champollion_V1 self-supervised encoder for a single sulcal region. Use this
instead of (or before) `start_embeddings` when custom model weights are needed.

**Prerequisite:** `start_config` must have run first, or configs must exist in `config_dir`.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `dataset` | str | yes | Dataset name — must match a directory in the config search path |
| `region` | str | yes | Region config name, e.g. `cingulate_left`. Must match a YAML file under `configs/dataset/{dataset}/` |
| `mode` | str | no | `encoder` (default), `classifier`, or `regresser` |
| `output_dir` | str | no | Absolute path for Hydra logs and model checkpoints. Defaults to `{pipeline_dir}/data/{dataset}/derivatives/champollion_V1/models/{region}/` |
| `config_dir` | str | no | Absolute path to a Hydra configs root containing `dataset/{dataset}/{region}.yaml`. Default: `{roots[0]}/data/{dataset}/derivatives/champollion_V1/configs` when the MCP client declares roots (`{roots[0]}` = first declared root, same base as the `output_dir` default), otherwise `{pipeline_dir}/data/{dataset}/derivatives/champollion_V1/configs`. Pass it when the configs live elsewhere (use `<D>/<dataset>/derivatives/champollion_V1/configs`) or when `start_config` was given an explicit `output` (use that configs root) |
| `njobs` | int | no | CPU DataLoader workers |
| `cpu` | bool | no | Force CPU (disable CUDA) |
| `overwrite` | bool | no | Re-train even if `output_dir` already exists |
| `swf` | bool | no | Not supported — `True` raises a ToolError before anything is created or launched (training has no soma-workflow mode) |
| `log` | bool | no | Pass a bare `--log` so the stage writes its run log to its default `.log` folder (default: true) |

Output: `{output_dir}/` — Hydra run directory containing the trained model checkpoint and config snapshot.

---

## Stage 4 — `start_embeddings`

Runs inference for every region model in `models_path` (56 for the full Champollion_V1 set: 28 regions × 2 hemispheres), one region after another. This is the longest stage.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `models_path` | str | yes | Local directory of per-region model folders, `.tar.gz` archive, URL, or HF repo ID (`neurospin/Champollion_V1`) |
| `datasets_root` | str | yes | Absolute path to the dataset root — the directory that contains `derivatives/` (e.g. `/data/TESTXX`). Crops are read from `{datasets_root}/derivatives/{cortical_version}/crops/{masks}/2mm/` |
| `cpu` | bool | no | Force CPU (disable CUDA) |
| `overwrite` | bool | no | Recompute regions whose `full_embeddings.csv` already exists (default: they are skipped) |
| `masks` | str | no | Mask version used as the crops subdirectory (default: `canonical_25`) |
| `masks_version` | str | no | Mask version subfolder to download from HuggingFace (e.g. `canonical_25`); ignored when `models_path` is a local directory |
| `output` | str | no | Output base directory. Default: `{datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/` (`{masks}` = the `masks` above; e.g. `/data/TESTXX` → `/data/TESTXX/derivatives/champollion_V1/canonical_25/region_embeddings/`) |
| `subjects` | str | no | Deprecated and ignored: each region reads its subject list from the `{side}skeleton_subject.csv` next to its skeleton |
| `regions` | list[str] | no | Restrict to these region model names (e.g. `SCsylv_left`). Default: every region found in `models_path` |
| `run_cka` | bool | no | Run the CKA coherence test after embeddings (results in `{output}/cka_results/`) |
| `cortical_version` | str | no | Derivatives folder that holds the crops (default: `cortical_tiles-2026`) |
| `profiling` | bool | no | Run the embeddings script under cProfile (`--profiling`) |
| `no_cache` | bool | no | Force archive re-extraction / HuggingFace re-download (`--no-cache`) |
| `legacy` | bool | no | Read crops from `derivatives/deep_folding-2025/crops/2mm/` (`--legacy`; overrides `cortical_version`) |
| `checkpoint_epoch` | int | no | Training epoch N (0-based, must be >= 0) of the Lightning checkpoint `epoch=N-step=*.ckpt` evaluated for each region (`--checkpoint-epoch`; pipeline default 80) |
| `log` | bool | no | Pass a bare `--log` so the stage writes its run log to its default `.log` folder (default: true) |

**HF_TOKEN**: Set in env before starting the MCP server if downloading from HuggingFace.

Output: `{output}/{region}/full_embeddings.csv` — one folder per region model (56 for the full set). Default `{output}`: `{datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/`. Downloaded or extracted models are cached under `<pipeline>/data/{datasets_root without its leading /}/derivatives/champollion_V1/models_cache/` (`<pipeline>` = champollion_pipeline root).

---

## Stage 5 — `start_combine`

Copies every per-region `full_embeddings.csv` produced by stage 4 into a single flat directory.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `embeddings_source` | str | yes | Absolute path to the stage-4 output directory (one `{region}/full_embeddings.csv` per region): the `output` given to `start_embeddings`, default `{datasets_root}/derivatives/champollion_V1/{masks}/region_embeddings/` |
| `output_path` | str | yes | Directory for the collected CSVs (created if missing) |
| `log` | bool | no | Pass a bare `--log` so the stage writes its run log to its default `.log` folder (default: true) |

Output: `{output_path}/{region}_embeddings.csv` — one file per region folder of `embeddings_source` that holds a `full_embeddings.csv`; regions without one are skipped with a `[skip]` log line. Zero files copied is logged as a warning, not a failure.

---

## Stage 6 — `start_snapshots`

Renders sulcal graph meshes, cortical tile masks, and UMAP scatter plots.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `output_dir` | str | yes | Where to write snapshot images |
| `morphologist_dir` | str | no | Morphologist output dir (for sulcal graph snapshots) |
| `cortical_tiles_dir` | str | no | `crops/{masks}/2mm/` dir from stage 2, default mask version `canonical_25` (for tile mask snapshots) |
| `embeddings_dir` | str | no | Combined embeddings dir from stage 5 (for UMAP plots) |
| `subject` | str | no | Subject folder name to visualize (default: first found) |
| `acquisition` | str | no | Acquisition tag (required if subject has multiple segmentations) |
| `sulcal_only` | bool | no | Render only sulcal graph images (requires `morphologist_dir`) |
| `tiles_only` | bool | no | Render only cortical tile mask images (requires `cortical_tiles_dir`) |
| `umap_only` | bool | no | Render only UMAP scatter plots (requires `embeddings_dir`) |
| `umap_region` | str | no | Restrict UMAP plot to a specific sulcal region (e.g. `SPoC_left`) |
| `champollion_data_root` | str | no | Override path to the Champollion data directory (used to resolve tile masks) |
| `reference_data_dir` | str | no | Dir of pre-trained UMAP artefacts (`umap_{region}_{hemi}.pkl` / `_coords.npy`). Not shipped, no default; without it no UMAP plots are produced |
| `log` | bool | no | Pass a bare `--log` so the stage writes its run log to its default `.log` folder (default: true) |

At least one of `morphologist_dir`, `cortical_tiles_dir`, or `embeddings_dir` must be provided.

UMAP plots need both `embeddings_dir` and `reference_data_dir`. Reference artefacts are not shipped with `champollion_pipeline` (gitignored); obtain them separately or build them with the pipeline's `pixi run generate-umap-reference` task (see the pipeline README "UMAP Visualization" section).
