# SocialPulse Analytics — Big Data Analytics (BDA)

An end-to-end Big Data social media text analytics and sentiment exploration platform. SocialPulse Analytics combines distributed data processing capabilities (**PySpark**) with an interactive presentation interface (**Streamlit**) while preserving the original exploratory research workflow (**Jupyter Notebook**).

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │ Raw Social Media Dataset                     │
                    │ (data/raw/tweets-engagement-metrics.csv)     │
                    │ 102,062 rows × 20 columns                    │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │ Processing Layer (src/spark_pipeline.py)     │
                    ├──────────────────────────────┬───────────────┤
                    │ PySpark Distributed Engine   │ Local Engine  │
                    │ - StructType Schema Ingestion│ (Pandas /     │
                    │ - Parallel Null & Dup Checks │ NumPy Fallback│
                    │ - log1p Scaled Metrics       │ when JDK > 21)│
                    │ - Grouped Aggregations       │               │
                    └──────────────┬───────────────┴───────┬───────┘
                                   │                       │
                                   ▼                       ▼
                    ┌──────────────────────────────────────────────┐
                    │ Core Logic & Scoring (src/utils.py)          │
                    │ - Exact Regex Preprocessing & Stopword Filter│
                    │ - VADER Lexicon Compound Scoring             │
                    │ - Threshold Classification (Pos/Neu/Neg)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │ Presentation Layer (Streamlit & Artifacts)   │
                    │ - Interactive Web Interface (app.py)         │
                    │ - Exploratory Notebook (notebooks/01_*.ipynb)│
                    │ - Saved Plots & Wordclouds (outputs/)        │
                    └──────────────────────────────────────────────┘
```

---

## 🚀 Key Features & Enhancements

1. **PySpark Integration (`src/spark_pipeline.py`)**:
   - **Explicit Schema Definition**: 20-field `StructType` mapping all raw dataset attributes accurately without guessing or schema drift.
   - **Distributed Data Profiling**: Spark SQL aggregations computing total rows, column datatypes, duplicate detection, and parallel missing value checks.
   - **Scalable Transformations**: PySpark `log1p` feature engineering on engagement metrics (`Likes`, `RetweetCount`) and distributed aggregations by posting hour and weekday.
   - **Driver Size Protection**: Safe conversion boundaries preventing driver memory overflow when integrating Python-based VADER scoring.
   - **Graceful Compatibility Adapter**: Transparent diagnostics detecting Java/JVM compatibility; seamlessly runs PySpark on supported JDKs (8, 11, 17) and verified fallback on newer JDKs.

2. **Streamlit Interactive UI (`app.py`)**:
   - Real-time engine status badge and environment diagnostics.
   - Dynamic dataset selector (Raw, Processed, or custom CSV upload up to 100MB).
   - Profiling dashboard displaying records, attributes, null distributions, and preview.
   - On-demand sentiment analysis execution with live progress reporting.
   - Interactive Plotly visualizations for sentiment distribution, engagement comparisons, and temporal patterns.
   - Preserved artifact gallery (word clouds, correlation heatmap, LDA topics).
   - Filterable results explorer with one-click CSV export.

---

## 📋 Prerequisites & Compatibility

- **Python**: 3.10 – 3.14 (Verified on Python 3.14.3)
- **Java / JDK Requirement for PySpark**:
  - Apache Spark 4.x officially requires **Java 17 (LTS)** or Java 21.
  - **Project-Local OpenJDK 17**: A standalone, self-contained OpenJDK 17 runtime is configured locally inside `C:\Project\SocialPulse-Analytics\jdk-17`. The pipeline automatically prioritizes this project-local JDK for Spark without modifying system-wide PATH or affecting system-level JDK installations (such as JDK 27).
  - **Windows Hadoop Utilities**: Preconfigured inside `C:\Project\SocialPulse-Analytics\hadoop\bin` (`winutils.exe`, `hadoop.dll`) to ensure clean execution with zero Windows shell warnings.
  - **Verified Fallback Mechanism**: If neither a compatible system JDK nor project-local JDK is detected, the application automatically engages the **Verified Local Pandas Fallback**, displaying the active engine status accurately.

### Verifying Environment Diagnostics

The project includes built-in auto-configuration:
```powershell
.\venv\Scripts\python.exe -c "from src.spark_pipeline import check_spark_environment; print(check_spark_environment())"
```

---

## 🛠️ Setup & Installation

From the project root (`C:\Project\SocialPulse-Analytics`):

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install / verify required packages
pip install -r requirements.txt
```

---

## 🏃 Run Instructions

### 1. Launching the Streamlit Web Application

```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Or with active venv:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Running the Original Notebook Workflow

The original analysis pipeline remains fully intact and usable:

```powershell
.\venv\Scripts\jupyter.exe lab
```
Open and execute `notebooks/01_dataset_inspection.ipynb`.

### 3. Running Validation Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```

---

## 📊 Dataset & Preserved Semantics

- **Raw Data**: `data/raw/tweets-engagement-metrics.csv` (102,062 rows, 20 columns).
- **Processed Data**: `data/processed/tweets_processed.csv` (102,062 rows, 24 columns).
- **Sentiment Thresholds (VADER Compound)**:
  - $\ge 0.05$ $\rightarrow$ **Positive** (48,759 records, 47.77%)
  - $\le -0.05$ $\rightarrow$ **Negative** (7,346 records, 7.20%)
  - Between $-0.05$ and $0.05$ $\rightarrow$ **Neutral** (45,957 records, 45.03%)
- **Zero Mock Data**: All metrics, charts, and tables display actual dataset calculations.
