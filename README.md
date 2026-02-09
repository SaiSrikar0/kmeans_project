# Mall Customer Clustering (Streamlit)

This Streamlit app predicts customer clusters and visualizes cluster distributions using a trained model and precomputed clustered CSV files.

[![Live App Screenshot](screenshot.png)](https://kmeans-project.streamlit.app/)

Live demo: https://kmeans-project.streamlit.app/

## Prerequisites
- Python 3.8+ (Windows)
- Git (optional)

## Recommended setup (Windows - Powershell)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n mall-cluster python=3.10 -y
conda activate mall-cluster
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Required files
Place the following files in the project root (same folder as `app.py`):

- `Mall_Customers.csv` — original dataset used for scaling/stats
- `clustered_customers.csv` — dataset with a `Cluster` column used for visualizations/stats
- `customer_cluster_model.pkl` — trained model used for predictions

If any of these files are missing the app will display an error. Update the filenames or move files to the project root as needed.

## Notes
- This repository contains only UI and visualization code; training scripts are not included here. If you need training code/steps, ask and I can add them.
- Colors and UI theme were updated to a dark neon palette; logic and layout remain unchanged.

## Contact
If you want alternate color palettes (colorblind-safe, pastel, or corporate brand palette), tell me which and I will add variants.
