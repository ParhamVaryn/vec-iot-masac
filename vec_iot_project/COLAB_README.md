# VEC-IoT Phase 3 MASAC — Google Colab

## Recommended way to run

1. Open `VEC_IoT_MASAC_Colab.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. When prompted, upload `vec_iot_project_phase3_masac_upgrade.zip`.
4. The notebook installs dependencies, checks the XML traces, runs a smoke test, trains MASAC, evaluates it, and downloads the model/results.

## GPU

The project runs on CPU. For faster PyTorch training, in Colab choose:
`Runtime -> Change runtime type -> T4 GPU` (when available).

## Files expected in the project

- `datasets/vehicles.xml`
- `datasets/tasks.xml`
- `train_masac.py`
- `evaluate_masac.py`
- `src/`

The notebook is designed so that the ZIP can have an extra top-level folder; it searches for the folder containing `train_masac.py` and `src/` automatically.
