
# 🚢 FOREX PAIR DOWNLOADER

> Powered by the **Drillship Model**

---

## 📖 About

A structured data engineering pipeline that acquires, cleans, and serves historical Forex candle data (OHLCV) for backtesting and ML analysis. Built to run locally and deploy seamlessly to Google Colab for cloud execution.

Currently supports **4 major pairs** across **1-Hour** and **4-Hour** timeframes:

| Pair | Role |
|------|------|
| EUR/USD | Control pair (most studied, sanity-check baseline) |
| GBP/USD | Volatile major |
| USD/JPY | Carry-trade sensitive |
| AUD/USD | Commodity-linked major |

---

## 🛠 Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.x |
| Environment | [uv](https://github.com/astral-sh/uv) |
| Data Ingestion | [dukascopy-python](https://pypi.org/project/dukascopy-python/) |
| Data Processing | [pandas](https://pandas.pydata.org/) |
| Storage Format | [Parquet](https://parquet.apache.org/) (via PyArrow) |
| Cloud Execution | Google Colab |
| Version Control | Git / GitHub |

---

## 🧠 The Drillship Model — Evaluation Phase

The Drillship Model is a unified decision engine that maps data characteristics to a pipeline specification. The model equation:

```
Pipeline = F(DataType, Source, Access, Velocity, Volume, Sensitivity, UseCase)
```

For this project, **F** was evaluated as follows:

| Parameter | Value | Decision |
|-----------|-------|----------|
| **DataType** | Time-series (OHLCV) | Structured candle data |
| **Source** | Dukascopy | Free, high-quality forex tick/bar data |
| **Access** | REST / HTTP Download | `dukascopy-python` library |
| **Velocity** | Batch | Historical replay (streaming planned) |
| **Volume** | ~300K rows total | Single-node (Pandas/Polars sufficient) |
| **Sensitivity** | Public | No PII / No GDPR constraints |
| **UseCase** | Backtesting / ML Features | Serve to feature store / model |

### Output: Pipeline Spec

| Spec Field | Value |
|------------|-------|
| Ingestion | Batch download from Dukascopy (Bid prices) |
| Storage | Parquet, partitioned by pair & timeframe |
| Transformation | Dedup → Sort → UTC align → Weekend filter → Validate |
| Serving | DataLoader class returning clean DataFrame |
| Observability | Logging + row count validation |

---

## ⚙️ Methodology — Execution Stage

The Drillship Model executes in a closed loop:

```
Discover → Spec → Ingest → Transform → Serve → Observe → Loop
```

This project implements the following phases:

---

### 1. 🔍 Discovery

Identify data sources, validate coverage, and confirm data quality before ingestion.

- **Source selected:** Dukascopy (native H1/H4 candle data, tick volume included)
- **Coverage:** 8–10 years of history
- **Regime coverage:** Trending (2020–2021), Ranging (2015–2017), Volatile shock (March 2020)

---

### 2. 📥 Ingestion

Download raw candle data directly from Dukascopy's historical data feed and save it as CSV in the raw layer. No modifications are made at this stage.

**What it does:**
- Maps pair/timeframe to Dukascopy's internal constants
- Fetches Bid-side OHLCV data for a given date range
- Saves raw CSV to `data/raw/`

📁 **File:** `src/fxpipe/Phases/ingest/duckascopy_downloader.py`

---

### 3. 🔄 Transformation

Clean, validate, and normalize the raw data. This is where the data becomes trustworthy.

**What it does:**
- **Deduplication:** Drops duplicate timestamps
- **Sorting:** Orders by timestamp ascending
- **Timezone Alignment:** Converts to UTC, strips timezone info for ML compatibility
- **Weekend/Holiday Logic:** Filters bars outside Forex market hours (Sun 22:00 UTC → Fri 22:00 UTC)
- **Data Validation:** Drops rows where `High < Low`, `Close` outside High/Low range, negative prices, or negative volume
- **Output:** Saves cleaned data as Parquet to `data/processed/`

📁 **File:** `src/fxpipe/Phases/transform/cleaner.py`

---

### 4. 📤 Serve

The single entry point for ML models. Abstracts storage details behind a clean API.

**What it does:**
- Loads cleaned Parquet files from `data/processed/`
- Filters by date range on request
- Returns a clean `pandas.DataFrame` ready for feature engineering or model training

📁 **File:** `src/fxpipe/Phases/serve/loader.py`

---

## 🚀 Usage in Google Colab

### Step 1: Clone & Install

```python
!git clone https://github.com/celpha2svx/fx_Pipe_Download.git
%cd fxpipe
!pip install .
```

### Step 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Ingest Data (Download to Drive)

```python
from src.fxpipe.phases.ingest.duckascopy_downloader import IngestionPhase
from datetime import datetime

DATA_DIR = '/content/drive/MyDrive/fxpipe_data'
ingestion = IngestionPhase(base_dir=DATA_DIR)

ingestion.fetch_data(
    pair="EURUSD",
    timeframe="H1",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1)
)
```

### Step 4: Transform Data (Clean & Save as Parquet)

```python
from src.fxpipe.phases.transform.cleaner import TransformationPhase

transformation = TransformationPhase(base_dir=DATA_DIR)

filename = "EURUSD_H1_20230101_20240101_raw.csv"
transformation.clean_data(pair="EURUSD", timeframe="H1", filename=filename)
```

### Step 5: Serve Data (Load for ML)

```python
from src.fxpipe.phases.serve.loader import DataLoader

loader = DataLoader(base_dir=f'{DATA_DIR}/processed')
df = loader.get_data(pair="EURUSD", timeframe="H1", start_date="2023-01-01", end_date="2023-06-30")

print(df.head())
print(df.shape)
```

> **Note:** Once data is in Google Drive, you never re-download it. Future sessions only require Steps 1, 2, and 5.

---

## 📂 Project Structure

```
fxpipe/
├── src/
│   └── fxpipe/
│       ├── __init__.py
│       └── phases/
│           ├── __init__.py
│           ├── ingest/
│           │   ├── __init__.py
│           │   └── duckascopy_downloader.py
│           ├── transform/
│           │   ├── __init__.py
│           │   └── cleaner.py
│           └── serve/
│               ├── __init__.py
│               └── loader.py
├── data/
│   ├── raw/
│   └── processed/
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## 📚 Source

- **Model:** The Drillship Model — by Afeez Ademuyiwa
- **Data Provider:** [Dukascopy Bank SA](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
- **Ingestion Library:** [dukascopy-python](https://pypi.org/project/dukascopy-python/)

---

## 👤 Author

**Afeez Ademuyiwa**

---
