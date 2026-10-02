# MENO

This repository contains the minimal code needed to reproduce the MENO experiments. Generated raw data, moments, normalization statistics, checkpoints, logs, and results are intentionally excluded.

## Project structure

```text
MENO/
|-- GlobalLocalMFE.py          # Global/local MENO model
|-- benchmark_common.py       # Reproducibility, scaling, and checkpoint helpers
|-- dataset_common.py         # Shared coordinate and normalization utilities
|-- metrics_common.py         # Shared physical-space metrics
|-- query_sampling.py         # Deterministic per-epoch query sampling
|-- training_protocol.py      # Shared optimizer, scheduler, and CLI protocol
|-- prepare_data.py           # raw_zipped -> raw -> moments/statistics entry point
|-- train.py                  # One training entry point for every benchmark
|-- requirements.txt          # Runtime and public-data conversion dependencies
|-- poisson/                  # Cross-geometry Poisson experiment
|   |-- dataset.py            # Dataset loader
|   |-- native.py             # Raw artifact reader and validation
|   |-- metric.py             # Scalar evaluation metric
|   |-- train.py              # Paper training configuration
|   `-- generate/             # Raw splitting and moment/statistics generation
|-- darcy/                    # Paper single-geometry Poisson/Darcy experiment
|   |-- dataset.py
|   |-- native.py
|   |-- metric.py
|   |-- train.py
|   `-- generate/             # Raw validation and moment/statistics generation
|-- nasa/                     # NASA-CRM surface experiment
|   |-- dataset.py
|   |-- native.py
|   |-- metric.py
|   |-- train.py
|   `-- generate/             # Official HDF5 splitting and derived artifacts
`-- ahmed/                    # AhmedML surface experiment
    |-- dataset.py
    |-- native.py
    |-- metric.py
    |-- train.py              # Explicit AhmedML model configuration
    `-- generate/             # Official VTP conversion and derived artifacts
```

Every `<dataset>/data/` and `<dataset>/result/` directory is generated locally and ignored by Git.

## Reproduction

### 1. Create the environment

Run every command from the repository root. Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

### 2. Download one dataset into `data/raw_zipped`

Do not rename the files listed below. Archives may remain compressed; `prepare_data.py` extracts ZIP and TAR archives safely.

| Experiment | Download | Files to place under `<dataset>/data/raw_zipped/` |
|---|---|---|
| NASA-CRM | [Official benchmark page](https://www.aiaa-appliedsurrogate.org/real-hover-problem), [official AASM Drive](https://drive.google.com/drive/folders/1KhoZiEHlZhGI8omMwHrp2mZRKGiSAydO?usp=drive_link) | `trainingData_NASA-CRM.h5` and `testData_NASA-CRM.h5` |
| AhmedML | [Official Hugging Face dataset](https://huggingface.co/datasets/neashton/ahmedml) | `run_1` through `run_500`, each containing `boundary_<id>.vtp`, `boundary_cell_area_<id>.npy`, and `geo_parameters_<id>.csv` |
| Poisson cross-geometry | [PKU download](https://disk.pku.edu.cn/link/AA268BBEB00A7941F19A449A0F040B5817) | `poisson_data.zip`; contains `data_manifest.json` with format `mfd-poisson-raw`, `star_raw_data.npz`, and `annular_raw_data.npz` |
| Poisson single-geometry / Darcy | [PKU download](https://disk.pku.edu.cn/link/AAB53388BC78DF445ABAB035728A525A0A) | `deformed_domain_darcy.zip`; only `smooth_small_scale` and `smooth_large_scale` are used |

The AhmedML download is large. The following command downloads only the files used by MENO:

```bash
python -m pip install huggingface_hub
hf download neashton/ahmedml --repo-type dataset --include "run_*/boundary_*.vtp" --include "run_*/boundary_cell_area_*.npy" --include "run_*/geo_parameters_*.csv" --local-dir ahmed/data/raw_zipped
```

Expected placement examples:

```text
nasa/data/raw_zipped/trainingData_NASA-CRM.h5
nasa/data/raw_zipped/testData_NASA-CRM.h5
ahmed/data/raw_zipped/run_1/boundary_1.vtp
poisson/data/raw_zipped/poisson_data.zip
darcy/data/raw_zipped/deformed_domain_darcy.zip
```

For Darcy, keep the downloaded ZIP directly in `darcy/data/raw_zipped/`;
manual extraction is not required. The preparer ignores macOS `__MACOSX`,
`.DS_Store`, and `._*` metadata in the official archive. It locates the real
`smooth_small_scale` and `smooth_large_scale` directories beneath any wrapper
folder. Keep only one source copy: do not place both the ZIP and its extracted
data in `raw_zipped/`. If `data/raw/` is already populated, use
`python prepare_data.py darcy --stage moment --mode 32` to build derived data.

### 3. Build trainable raw files, moments, and statistics

One command performs the complete `raw_zipped -> raw -> moment -> normalization` chain. It refuses to overwrite an existing build.

```bash
python prepare_data.py nasa --mode 8
python prepare_data.py poisson --mode 12
python prepare_data.py darcy --mode 32
python prepare_data.py ahmed --mode 16
```

Each dataset stores one active moment order at a time. When changing `--mode`, rebuild only the derived artifacts; raw files are preserved. For example:

```bash
python prepare_data.py nasa --stage moment --mode 10 --replace-derived
```

The generated layout is:

```text
<dataset>/data/
|-- raw_zipped/               # Downloaded files; never committed
|-- raw/                      # Trainable per-case files made by prepare_data.py
|-- moment/                   # Moments made only from raw
|   `-- normalization/        # Training-split statistics made only from raw
`-- manifest.json             # Shapes, splits, hashes, and artifact lineage
```

### 4. Run the paper configurations

All runs explicitly use the project-wide reproducibility seed, `42`. A CUDA GPU is recommended.

```bash
python train.py poisson --mode 12 --batch-size 10 --train-query-limit 10000 --layers 4 --global-width 128 --local-width 128 --heads 8 --feedforward 256 --global-position-frequencies 4 --local-fourier-frequencies 6 --normalize-inputs --normalize-outputs --global-dropout 0 --no-amp --log-every 1 --checkpoint-every 1 --epochs 500 --seed 42 --device cuda
python train.py darcy --mode 32 --batch-size 4 --train-query-limit 2500 --layers 4 --global-width 160 --local-width 160 --heads 10 --feedforward 280 --global-position-frequencies 6 --local-fourier-frequencies 3 --normalize-inputs --normalize-outputs --no-amp --validation-every 25 --log-every 1 --epochs 500 --seed 42 --device cuda
python train.py nasa --mode 8 --batch-size 1 --train-query-limit 16384 --layers 6 --global-width 512 --local-width 512 --heads 8 --feedforward 512 --global-position-frequencies 4 --local-fourier-frequencies 6 --normalize-inputs --normalize-outputs --global-dropout 0 --local-dropout 0 --no-amp --eval-every 1 --log-every 1 --checkpoint-every 0 --epochs 500 --seed 42 --device cuda
python train.py ahmed --mode 16 --batch-size 1 --train-query-limit 16384 --layers 6 --global-width 512 --local-width 512 --heads 8 --feedforward 256 --global-position-frequencies 4 --local-fourier-frequencies 6 --normalize-inputs --normalize-outputs --global-dropout 0 --local-dropout 0 --no-amp --eval-every 1 --log-every 1 --checkpoint-every 0 --epochs 500 --seed 42 --device cuda
```

The commands above pass the paper settings explicitly; they are not fixed profiles. The existing model and training options remain unchanged, while `--mode` accepts any integer greater than or equal to 2 for every dataset. The training `--mode` must match the mode used by `prepare_data.py`.

To run the paper's smaller AhmedML configuration, first replace only the derived mode-16 artifacts, then pass the smaller model parameters explicitly:

```bash
python prepare_data.py ahmed --stage moment --mode 8 --replace-derived
python train.py ahmed --mode 8 --batch-size 1 --train-query-limit 16384 --layers 4 --global-width 256 --local-width 256 --heads 8 --feedforward 256 --global-position-frequencies 4 --local-fourier-frequencies 6 --normalize-inputs --normalize-outputs --global-dropout 0 --local-dropout 0 --no-amp --eval-every 1 --log-every 1 --checkpoint-every 0 --epochs 500 --seed 42 --device cuda
```

Each command writes a new run under `<dataset>/result/`; that directory is ignored by Git. Use `--help` after the dataset name to inspect all parameters, for example `python train.py nasa --help` or `python prepare_data.py nasa --help`.
