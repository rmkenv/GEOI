### GEOI — Geospatial Investment Analysis

#### Overview
GEOI is a research repo focused on analyzing publicly traded “geospatial” companies (mapping, satellite, sensors, location-based software, etc.). It ships with:
- Routine “snapshot” parquet files containing precomputed fundamentals and performance metrics.
- An interactive analysis app (Jupyter/Colab) to filter tickers, view recent performance, compare to benchmarks, and run a conservative DCF panel.

Use it as a reproducible workflow to explore the geospatial equity universe, monitor cohorts, and quickly surface candidates for deeper research.

#### Key Features
- Automatic “latest snapshot” discovery from GitHub (no hardcoded file paths).
- Fast loading of a curated universe with precomputed metrics (e.g., 1M/3M/6M/YTD/1Y/5Y).
- Interactive dashboard with:
  - Filter & Export panel
  - Performance analysis vs. benchmarks (e.g., SPY, IXN)
  - Value investing view with conservative DCF + margin-of-safety verdicts
- Works in Google Colab, local Jupyter Notebook/Lab, and macOS virtual environments.

---

### Repository Structure
- [GEOI (root)](https://github.com/rmkenv/GEOI)
  - [analysis.py](https://github.com/rmkenv/GEOI/blob/main/analysis.py) — main app entrypoint (Jupyter/Colab-friendly).
  - [snapshots/](https://github.com/rmkenv/GEOI/tree/main/snapshots) — monthly parquet snapshots (e.g., snapshots/2025/…).
  - [geospatial_companies_cleaned.parquet](https://github.com/rmkenv/GEOI/raw/main/geospatial_companies_cleaned.parquet) — cleaned universe (raw file link).

Note: The app will auto-discover the most recent snapshot under snapshots/<current_year> using the GitHub API.

---

### Quickstart

#### Option A: Google Colab (recommended for zero-setup)
- Open Colab and run these cells in order:

```python
# 1) Install deps and enable widgets
!pip -q install ipywidgets yfinance fastparquet --upgrade
from google.colab import output
output.enable_custom_widget_manager()
```

```python
# 2) Pull and load the app
!curl -L -o analysis.py https://raw.githubusercontent.com/rmkenv/GEOI/main/analysis.py
import importlib, analysis
importlib.reload(analysis)
analysis.run_portfolio_app(year_folder=None)  # or "2025"
```

If you hit GitHub API rate limits when auto-discovering the latest snapshot:
```python
import os
os.environ["GITHUB_TOKEN"] = "ghp_your_personal_access_token"
```

#### Option B: Local Jupyter (Notebook or Lab)
1) Create a virtual environment (macOS/Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:
```bash
pip install --upgrade pip
pip install jupyter ipywidgets yfinance fastparquet
# Optional alternative parquet engine:
# pip install pyarrow
```

3) Launch Jupyter:
```bash
jupyter notebook
# or
jupyter lab
```

4) In a new notebook:
```python
!git clone https://github.com/rmkenv/GEOI.git || true
%cd GEOI
import importlib, analysis
importlib.reload(analysis)
# Optional: set GitHub token to avoid API rate limits
# import os; os.environ["GITHUB_TOKEN"] = "ghp_your_personal_access_token"
analysis.run_portfolio_app(year_folder=None)
```

macOS note (classic Notebook only): if widgets don’t render, run once in Terminal:
```bash
jupyter nbextension enable --py widgetsnbextension
```

---

### Usage

- Default behavior: analysis.run_portfolio_app() will:
  - Query the GitHub API for snapshots/<current_year>
  - Pick the lexicographically latest parquet file (snapshot_YYYY-MM-DD.parquet)
  - Load and display 3 interactive panels:
    - Filter & Load Stocks (with precomputed metrics)
    - Performance Analysis panel (5Y/YTD/6M/3M/30D vs. benchmarks)
    - Value Investing panel (conservative DCF + margin-of-safety verdicts)

- Force a specific snapshot (optional):
```python
from analysis import parse_tickers_from_parquet_github
df = parse_tickers_from_parquet_github(
    parquet_url="https://raw.githubusercontent.com/rmkenv/GEOI/main/snapshots/2025/snapshot_2025-10-09.parquet"
)
```

- Switch year folder (e.g., when a new year begins):
```python
analysis.run_portfolio_app(year_folder="2025")
```

---

### Requirements
- Python 3.9+
- Packages:
  - ipywidgets
  - yfinance
  - fastparquet (or pyarrow)
  - pandas, numpy, requests (installed transitively)

You can also capture these in a requirements.txt for automated installs:
```
ipywidgets>=8.0.0
yfinance>=0.2.40
fastparquet>=2024.2.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
# pyarrow>=16.0.0  # optional parquet engine
```

---

### Troubleshooting
- “Panels don’t appear in Colab”:
  - Ensure you ran: output.enable_custom_widget_manager() before importing/launching the app.
  - Avoid double execution; import the module and call run_portfolio_app() once.
- “GitHub API rate limit exceeded”:
  - Set GITHUB_TOKEN in your environment (Personal Access Token).
- “Parquet engine errors”:
  - Install pyarrow: pip install pyarrow, then restart kernel.
- “Printed headers but no UI” (local Jupyter):
  - Confirm widgets are installed in the same env that runs Jupyter.
  - For classic Notebook: jupyter nbextension enable --py widgetsnbextension (then restart).

---

### Contributing
- Open issues or pull requests on [GEOI](https://github.com/rmkenv/GEOI).
- Suggested improvements:
  - Additional benchmarks or sector/industry filters
  - New snapshot metrics or factor screens
  - Export/Reporting enhancements

---

### License
- See the repository for licensing details or include a LICENSE file if needed.

---

### Acknowledgments
- Price and fundamentals via yfinance (Yahoo Finance data; subject to availability/changes).
- GitHub-hosted parquet snapshots for reproducible states over time.
