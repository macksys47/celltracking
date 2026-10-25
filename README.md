# Robust Cell Tracking

Official implementation of **“Optical Flow Temporal Subsampling and Query
Denoising with Coopetition Label Assignment for Robust Cell Tracking.”** The
framework combines temporal sampling, denoised tracking queries, and
coopetition label assignment for robust tracking in time-lapse microscopy.

The repository contains the main TrackFormer-based model, configurations for
five cell lines, leave-one-dataset-out (LODO) experiments, ablations, and
standalone CellDINO- and Trackastra-style tracking baselines.

## Highlights

- Joint cell detection and association over consecutive microscopy frames.
- Experiments on `3t3`, `hek`, `hela`, `pc`, and `raw` datasets.
- Ready-to-use configurations for the main model, LODO evaluation, and model
  ablations.
- CTC-compatible inference output and scripts for evaluation, failure-case
  analysis, and resource profiling.
- Reproducible comparison pipelines for CellDINO and Trackastra-style models.

## Installation

The code is intended for Linux with an NVIDIA GPU. Python 3.10, PyTorch 1.13.1,
and CUDA 11.6 were used in our environment.

```bash
git clone <your-repository-url>
cd ct

conda env create -f env.yml
conda activate deform_env
pip install -r requirements.txt
pip install -e .

# Install the COCO API used by TrackFormer.
pip install -U "git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI"

# Compile multi-scale deformable attention after installing PyTorch/CUDA.
cd src/trackformer/models/ops
python setup.py build install
cd ../../../..
```

Install the CUDA-enabled PyTorch build separately if it is not already present.
The custom attention operator must be compiled with a CUDA toolkit compatible
with the installed PyTorch version.

## Data preparation

Datasets follow the [Cell Tracking Challenge (CTC)](https://celltrackingchallenge.net/)
layout. Each dataset also needs the COCO annotations and images used during
training:

```text
data/<dataset>/
├── train/
│   ├── 01/                 # TIFF image sequence
│   └── 01_GT/{SEG,TRA}/    # segmentation and tracking labels
├── val/
│   ├── 01/
│   └── 01_GT/{SEG,TRA}/
├── test/
│   └── 01/
└── COCO/
    ├── annotations/{train,val}/
    ├── man_track/{train,val}/
    ├── train/{img,gt}/
    └── val/{img,gt}/
```

Before running an experiment, edit its file in `cfgs/` and set `data_dir` and
`output_dir` to local paths. The base configurations are `train_3t3.yaml`,
`train_hek.yaml`, `train_hela.yaml`, `train_pc.yaml`, and `train_raw.yaml`.

## Training and inference

The main entry points use [Sacred](https://sacred.readthedocs.io/) to load a
configuration selected by dataset/experiment name:

```bash
# Train; checkpoints and metrics are written to results/3t3 by default.
python src/train.py with dataset=3t3

# Run inference with results/3t3/checkpoint.pth.
python src/pipeline.py with dataset=3t3
```

Training resumes automatically when `checkpoint.pth` is present in the result
directory. Inference reads TIFF sequences from `data/<dataset>/test`; if that
folder is absent, it falls back to `val`. Predictions are written below the
experiment result directory in CTC format, together with `FPS.txt`.

Convenience launchers such as `run_train_3t3.sh` and `run_inf_3t3.sh` are also
provided. These scripts contain machine-specific `ROOT` and GPU settings, so
adjust them before use.

### Ablations and LODO

Ablation configurations use names such as:

```bash
python src/train.py with dataset=3t3-ab-dq0-cola0
python src/pipeline.py with dataset=3t3-ab-dq0-cola0
```

LODO splits can be generated with `scripts/create_lodo_cross_dataset_splits.py`.
Use the `train_lodo_test_*.yaml` configurations with `src/train.py` and
`src/pipeline.py` afterward.

### Tracking baselines

Standalone CellDINO- and Trackastra-style comparison implementations live in
`src/celldino_tracking/` and `src/trackastra_tracking/`. For example:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

bash run_train_celldino_3t3.sh
bash run_inf_celldino_3t3.sh

bash run_train_trackastra_3t3.sh
bash run_inf_trackastra_3t3.sh
```

As with the main launchers, update their `ROOT` variable when the repository is
not located at `/group/dingyi/ct`.

## Repository structure

```text
cfgs/                       experiment configurations
data/                       local datasets (ignored by Git)
scripts/                    evaluation and analysis utilities
src/train.py                main training entry point
src/pipeline.py             main inference entry point
src/trackformer/            detection and tracking model
src/celldino_tracking/      CellDINO-style baseline
src/trackastra_tracking/    Trackastra-style baseline
results/                    checkpoints and predictions (ignored by Git)
```

This implementation builds on
[DETR](https://github.com/facebookresearch/detr),
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR),
and [TrackFormer](https://github.com/timmeinhardt/trackformer).

## Citation

If this repository is useful in your research, please cite the accompanying
paper. The final BibTeX entry will be added after publication.

## License

Released under the [Apache License 2.0](LICENSE).
