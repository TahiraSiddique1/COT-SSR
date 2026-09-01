# Integrated Satellite + ERA5 + CERES Training Pipeline

This repository builds training data for Bangladesh cloud/solar modeling and trains a multi-output deep learning model.

## Project Files

- `pipeline.py`: Builds integrated `.npz` training samples by combining:
  - Himawari-9 satellite bands (`B03`, `B13`) from NOAA S3
  - ERA5 auxiliary features (cloud cover, solar radiation, 2m temperature)
  - CERES labels from NASA Earthdata
  - also optimiazed for different region data 
- `train.py`: Trains a dual-head U-Net-like Keras model on generated `.npz` files.

## Data Flow

1. Download ERA5 monthly files with `fetch_aux.py`.
2. Run `pipeline.py` to generate sample files in `training_data_master/`.
3. Run `train.py` to train and evaluate the model.

## Prerequisites

- Python 3.10+ (recommended)
- Access credentials:
  - Copernicus CDS API (for ERA5 via `cdsapi`)
  - NASA Earthdata login (for `earthaccess` / CERES)

## Python Dependencies

Install required packages:

```bash
pip install numpy pandas tensorflow matplotlib s3fs satpy opencv-python xarray h5netcdf earthaccess cdsapi dask
```

Notes:
- `satpy` may pull additional reader dependencies depending on your environment.
- On some systems, `opencv-python-headless` can be used instead of `opencv-python`.

## Configuration You Should Review

Before running, update hardcoded paths and ranges in scripts:

### `fetch_aux.py`

- `YEARS` (currently `['2023']`)
- `DATA_ROOT` (currently `aux_data1` in current working directory)

### `pipeline.py`

- `START_DATE`, `END_DATE`, `HOURS`
- `AUX_DATA_ROOT` (currently a hardcoded absolute Windows path)
- `MAX_WORKERS`, `DOWNLOAD_THREADS`

### `train.py`

- `TRAIN_DIR` (currently a hardcoded absolute Windows path)
- `MODEL_PATH`, `BATCH_SIZE`, `EPOCHS`

## Running

From repository root:

```bash
python fetch_aux.py
python pipeline.py
python train.py
```

## Output Artifacts

### From `fetch_aux.py`

- `aux_data1/era5_bd_YYYY_MM/`
  - ERA5 NetCDF files (zipped or unzipped depending on CDS response)

### From `pipeline.py`

- `training_data_master/master_YYYYMMDD_HHMM.npz`
  - `X`: 5-channel input tensor
    - Channel 0: VIS (`B03`) normalized
    - Channel 1: IR (`B13`) normalized
    - Channel 2: ERA5 total cloud cover
    - Channel 3: ERA5 solar radiation scaled
    - Channel 4: ERA5 2m temperature normalized
  - `y_cot`: CERES cloud optical depth label
  - `y_solar`: solar target array (from pipeline output)

### From `train.py`

- `grand_unified_model.keras` (best checkpoint)
- `test_set_files.txt` (test split file list)
- `training_history_split.png` (loss/metric curves)

## Model Summary

`train.py` defines:

- A custom `UnifiedDataGenerator` with corruption tolerance and optional augmentation.
- A U-Net-like encoder/decoder with batch normalization at input.
- Two regression heads:
  - `cot_head` with masked MSE/MAE (ignores NaNs)
  - `solar_head` with MSE/MAE

Loss configuration:

- Total loss = `0.2 * cot_head_loss + 1.0 * solar_head_loss`

## Operational Notes

- `pipeline.py` uses `ProcessPoolExecutor`; each worker initializes Earthdata login.
- Temporary folder `temp_integrated/` is used and cleaned per time slot.
- Night scenes are skipped when mean VIS is too low.
- Some target naming in comments/variables may differ from source provenance; rely on saved array keys in `.npz`.

## Troubleshooting

- If ERA5 retrieval fails: verify CDS credentials and API setup.
- If CERES download/search fails: run Earthdata login and verify account access.
- If no training files found in `train.py`: confirm `TRAIN_DIR` points to `training_data_master`.
- If memory/CPU spikes: lower `MAX_WORKERS`, `DOWNLOAD_THREADS`, or date range.

## Suggested Improvement (Optional)

To make this project portable, move hardcoded paths/settings into a single config file or CLI arguments (for example, `argparse`), then update all scripts to read from that config.
