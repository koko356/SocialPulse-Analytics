"""
SocialPulse-Analytics Streamlit Application
A Big Data Analytics (BDA) interactive demonstration interface.
Integrates PySpark processing status, data profiling, sentiment analysis,
engagement analytics, and artifact exploration.
"""

import os
import sys
import time
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.spark_pipeline import (
    DataProcessingPipeline,
    check_spark_environment,
    SparkSessionManager
)
from src.utils import clean_text, classify_sentiment, get_vader_analyzer

st.set_page_config(
    page_title="SocialPulse Analytics | BDA Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .metric-container {
        text-align: center;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


def load_data_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def main():
    # Header
    st.markdown('<div class="main-header">📊 SocialPulse Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Big Data Analytics (BDA) Platform &bull; PySpark Ingestion & Sentiment Analysis</div>', unsafe_allow_html=True)

    # Sidebar: Engine & Environment Diagnostics
    st.sidebar.title("⚙️ Engine & Controls")

    env_status = check_spark_environment()
    engine_choice = st.sidebar.radio(
        "Select Processing Engine:",
        ["Auto-Detect / PySpark", "Local Pandas Engine"],
        help="PySpark utilizes distributed Spark DataFrames when compatible Java (JDK 8/11/17) is configured."
    )

    prefer_spark = (engine_choice == "Auto-Detect / PySpark")
    pipeline = DataProcessingPipeline(prefer_spark=prefer_spark)

    # Engine Status Display
    st.sidebar.markdown("---")
    st.sidebar.subheader("Engine Status")
    if "PySpark" in pipeline.active_engine and "fallback" not in pipeline.active_engine.lower():
        st.sidebar.success(f"🟢 **{pipeline.active_engine}**")
    else:
        st.sidebar.info(f"🔵 **{pipeline.active_engine}**")

    with st.sidebar.expander("🔍 Environment Diagnostics", expanded=False):
        st.write(f"**Python Version**: {sys.version.split()[0]}")
        st.write(f"**PySpark Installed**: {'Yes (' + str(env_status['pyspark_version']) + ')' if env_status['pyspark_installed'] else 'No'}")
        st.write(f"**Java Installed**: {env_status['java_version'] or 'Not Found'}")
        st.write(f"**JAVA_HOME**: {env_status['java_home'] or 'Not Set'}")
        st.write(f"**Compatibility**: {env_status['status_label']}")
        if env_status["diagnostic_message"]:
            st.caption(env_status["diagnostic_message"])

    # Dataset Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Dataset Selection")
    data_source = st.sidebar.selectbox(
        "Choose Data Source:",
        [
            "Raw Dataset (tweets-engagement-metrics.csv)",
            "Processed Dataset (tweets_processed.csv)",
            "Upload Custom CSV"
        ]
    )

    df = None
    data_path = ""
    raw_default = os.path.join(BASE_DIR, "data", "raw", "tweets-engagement-metrics.csv")
    proc_default = os.path.join(BASE_DIR, "data", "processed", "tweets_processed.csv")

    if data_source == "Raw Dataset (tweets-engagement-metrics.csv)":
        data_path = raw_default
        if os.path.exists(data_path):
            df = load_data_file(data_path)
        else:
            st.error(f"Raw dataset not found at `{data_path}`")
    elif data_source == "Processed Dataset (tweets_processed.csv)":
        data_path = proc_default
        if os.path.exists(data_path):
            df = load_data_file(data_path)
        else:
            st.error(f"Processed dataset not found at `{data_path}`")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV File (Max 100MB)", type=["csv"])
        if uploaded_file is not None:
            try:
                # File size validation (limit 100MB)
                if uploaded_file.size > 100 * 1024 * 1024:
                    st.error("Uploaded file exceeds 100MB limit.")
                else:
                    df = pd.read_csv(uploaded_file)
                    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(df):,} rows)")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")

    if df is None:
        st.warning("Please select or upload a valid dataset to proceed.")
        return

    # Tabs for Organization
    tab_overview, tab_processing, tab_analytics, tab_artifacts, tab_streaming = st.tabs([
        "📋 Data Overview & BDA Profiling",
        "⚡ Data Processing & Sentiment",
        "📈 Analytics & Visualizations",
        "🖼️ Existing Project Artifacts",
        "🌊 Streaming Algorithms Lab",
    ])

    # TAB 1: DATA OVERVIEW & PROFILING
    with tab_overview:
        st.subheader("Data Profiling Summary")

        # Top Metrics
        profile = pipeline.profile_data(df)
        total_rows = profile.get("total_rows", len(df))
        total_cols = profile.get("total_columns", len(df.columns))
        dup_count = profile.get("duplicate_count", 0)
        null_counts = profile.get("null_counts", {})
        total_nulls = sum(null_counts.values()) if null_counts else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{total_rows:,}")
        c2.metric("Total Attributes", f"{total_cols}")
        c3.metric("Missing Values", f"{total_nulls:,}")
        c4.metric("Duplicate Rows", f"{dup_count}")

        # Engine Badge
        st.markdown(f"""
        <div class="status-card">
            <strong>Active Processing Engine:</strong> {pipeline.active_engine}<br>
            <span style="font-size:0.9rem; color:#4B5563;">{pipeline.engine_details}</span>
        </div>
        """, unsafe_allow_html=True)

        # Missing Values & Columns Details
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Attributes & Data Types")
            dtypes_df = pd.DataFrame([
                {"Column": col, "Data Type": str(dtype), "Null Count": null_counts.get(col, 0)}
                for col, dtype in df.dtypes.items()
            ])
            st.dataframe(dtypes_df, use_container_width=True, height=280)

        with col_right:
            st.markdown("#### Missing Values Distribution")
            missing_cols = {k: v for k, v in null_counts.items() if v > 0}
            if missing_cols:
                missing_df = pd.DataFrame(list(missing_cols.items()), columns=["Column", "Missing Count"])
                fig_null = px.bar(
                    missing_df,
                    x="Column",
                    y="Missing Count",
                    color="Missing Count",
                    color_continuous_scale="Reds",
                    text_auto=True
                )
                fig_null.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_null, use_container_width=True)
            else:
                st.success("No missing values found in any columns!")

        # Data Preview
        st.markdown("#### Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

    # TAB 2: PROCESSING & SENTIMENT PIPELINE
    with tab_processing:
        st.subheader("Data Cleaning & Sentiment Analysis Execution")
        st.markdown(r"""
        Executes the exact text-cleaning regex rules and VADER sentiment analyzer from `01_dataset_inspection.ipynb`.
        - **Preprocessing**: URL, mention, hashtag, special-character removal + lowercase + NLTK stopwords.
        - **Sentiment Scoring**: Lexicon-based VADER scoring (`compound_score`).
        - **Threshold Classification**: `Positive` ($\ge 0.05$), `Negative` ($\le -0.05$), `Neutral` (between $-0.05$ and $0.05$).
        """)

        has_processed_cols = all(c in df.columns for c in ["cleaned_text", "compound_score", "sentiment_label"])

        run_mode = st.radio(
            "Select Processing Execution Scope:",
            [
                "Use Existing Columns (if already processed)" if has_processed_cols else "Run on Sample (5,000 rows)",
                "Run on Full Dataset (102,062 rows)",
                "Run on Custom Sample Size"
            ]
        )

        sample_size = None
        if "Custom Sample Size" in run_mode:
            sample_size = st.slider("Select Sample Size:", min_value=500, max_value=len(df), value=5000, step=500)
        elif "5,000 rows" in run_mode:
            sample_size = 5000

        # Processing Button
        col_btn, col_info = st.columns([1, 2])
        process_clicked = col_btn.button("🚀 Execute Sentiment Pipeline", type="primary")

        if "processed_data" not in st.session_state:
            if has_processed_cols:
                st.session_state["processed_data"] = df
            else:
                st.session_state["processed_data"] = None

        if process_clicked:
            start_time = time.time()
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(val, msg):
                progress_bar.progress(val)
                status_text.text(msg)

            with st.spinner("Processing social media data..."):
                processed_df = pipeline.run_sentiment_pipeline(
                    df,
                    sample_size=sample_size,
                    progress_callback=update_progress
                )
                elapsed = time.time() - start_time
                st.session_state["processed_data"] = processed_df
                st.session_state["processed_engine"] = pipeline.active_engine
                st.session_state["elapsed_time"] = elapsed
                status_text.success(f"Processing successfully completed in {elapsed:.2f} seconds via {pipeline.active_engine}!")

        # Display Processed Data
        active_proc_df = st.session_state.get("processed_data")
        if active_proc_df is not None and "sentiment_label" in active_proc_df.columns:
            st.markdown("---")
            st.subheader("Processed Results Summary")

            proc_engine = st.session_state.get("processed_engine", pipeline.active_engine)
            st.markdown(f"""
            <div class="status-card">
                <strong>Execution Engine:</strong> {proc_engine}<br>
                <span style="font-size:0.9rem; color:#4B5563;">{pipeline.engine_details}</span>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            counts = active_proc_df["sentiment_label"].value_counts()
            pos = counts.get("Positive", 0)
            neu = counts.get("Neutral", 0)
            neg = counts.get("Negative", 0)
            tot = len(active_proc_df)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Processed Rows", f"{tot:,}")
            m2.metric("Positive Posts", f"{pos:,}", f"{(pos/tot*100):.1f}%")
            m3.metric("Neutral Posts", f"{neu:,}", f"{(neu/tot*100):.1f}%")
            m4.metric("Negative Posts", f"{neg:,}", f"{(neg/tot*100):.1f}%")

            # Result Preview Table with filters
            st.markdown("#### Results Explorer")
            filter_sentiment = st.multiselect(
                "Filter by Sentiment:",
                ["Positive", "Neutral", "Negative"],
                default=["Positive", "Neutral", "Negative"]
            )
            filtered_df = active_proc_df[active_proc_df["sentiment_label"].isin(filter_sentiment)]

            display_cols = [c for c in ["TweetID", "text", "cleaned_text", "compound_score", "sentiment_label", "Likes", "RetweetCount"] if c in filtered_df.columns]
            st.dataframe(filtered_df[display_cols].head(50), use_container_width=True)

            # Download CSV
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Processed Dataset (CSV)",
                data=csv_data,
                file_name="socialpulse_processed_results.csv",
                mime="text/csv",
                key="download-processed-csv"
            )

    # TAB 3: ANALYTICS & VISUALIZATIONS
    with tab_analytics:
        st.subheader("Big Data Analytics & Statistical Visualizations")

        active_proc_df = st.session_state.get("processed_data")
        if active_proc_df is None or "sentiment_label" not in active_proc_df.columns:
            st.info("Run the Sentiment Pipeline in Tab 2 or select the Processed Dataset to view analytics.")
        else:
            # 1. Sentiment Distribution Bar Chart
            st.markdown("#### 1. Sentiment Distribution")
            sent_counts_df = active_proc_df["sentiment_label"].value_counts().reset_index()
            sent_counts_df.columns = ["Sentiment", "Count"]
            sent_counts_df["Percentage"] = (sent_counts_df["Count"] / len(active_proc_df) * 100).round(2)

            color_map = {"Positive": "#10B981", "Neutral": "#6B7280", "Negative": "#EF4444"}
            fig_dist = px.bar(
                sent_counts_df,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map=color_map,
                text="Percentage",
                title="Sentiment Class Distribution"
            )
            fig_dist.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_dist.update_layout(height=350)
            st.plotly_chart(fig_dist, use_container_width=True)

            # 2. Engagement Metrics Breakdown
            st.markdown("#### 2. Engagement Breakdown by Sentiment")
            if "Likes" in active_proc_df.columns and "RetweetCount" in active_proc_df.columns:
                eng_df = active_proc_df.groupby("sentiment_label")[["Likes", "RetweetCount", "Reach"]].mean().reset_index()

                fig_eng = go.Figure()
                fig_eng.add_trace(go.Bar(
                    x=eng_df["sentiment_label"],
                    y=eng_df["Likes"],
                    name="Avg Likes",
                    marker_color="#3B82F6"
                ))
                fig_eng.add_trace(go.Bar(
                    x=eng_df["sentiment_label"],
                    y=eng_df["RetweetCount"],
                    name="Avg Retweets",
                    marker_color="#8B5CF6"
                ))
                fig_eng.update_layout(
                    barmode="group",
                    title="Average Likes and Retweets by Sentiment",
                    height=350
                )
                st.plotly_chart(fig_eng, use_container_width=True)

            # 3. Temporal Patterns
            col_hr, col_day = st.columns(2)
            with col_hr:
                if "Hour" in active_proc_df.columns:
                    st.markdown("#### 3. Post Activity by Hour")
                    hr_df = active_proc_df["Hour"].value_counts().sort_index().reset_index()
                    hr_df.columns = ["Hour", "Count"]
                    fig_hr = px.line(hr_df, x="Hour", y="Count", markers=True, title="Hourly Activity Trend")
                    fig_hr.update_layout(height=320)
                    st.plotly_chart(fig_hr, use_container_width=True)

            with col_day:
                if "Weekday" in active_proc_df.columns:
                    st.markdown("#### 4. Post Activity by Weekday")
                    day_df = active_proc_df["Weekday"].value_counts().reset_index()
                    day_df.columns = ["Weekday", "Count"]
                    fig_day = px.bar(day_df, x="Weekday", y="Count", color="Count", title="Weekday Activity")
                    fig_day.update_layout(height=320)
                    st.plotly_chart(fig_day, use_container_width=True)

    # TAB 4: EXISTING PROJECT ARTIFACTS
    with tab_artifacts:
        st.subheader("Existing Project Artifacts Gallery")
        st.markdown("""
        All artifacts generated by `01_dataset_inspection.ipynb` under `outputs/` are preserved intact.
        """)

        plots_dir = os.path.join(BASE_DIR, "outputs", "plots")
        wordclouds_dir = os.path.join(BASE_DIR, "outputs", "wordclouds")

        # Wordclouds
        st.markdown("#### Sentiment Word Clouds")
        wc_cols = st.columns(3)
        wc_map = [
            ("Positive Word Cloud", os.path.join(wordclouds_dir, "positive_wordcloud.png")),
            ("Neutral Word Cloud", os.path.join(wordclouds_dir, "neutral_wordcloud.png")),
            ("Negative Word Cloud", os.path.join(wordclouds_dir, "negative_wordcloud.png")),
        ]
        for col, (label, path) in zip(wc_cols, wc_map):
            with col:
                st.markdown(f"**{label}**")
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.info(f"File not found: {path}")

        # Core Plots
        st.markdown("#### Preserved Analysis Charts")
        p_cols = st.columns(2)
        plot_files = [
            ("Correlation Heatmap", os.path.join(plots_dir, "correlation_heatmap.png")),
            ("Weekday vs Hour Heatmap", os.path.join(plots_dir, "tweet_activity_weekday_hour_heatmap.png")),
            ("Top 15 Words", os.path.join(plots_dir, "top_15_words.png")),
            ("Sentiment Distribution (Notebook Output)", os.path.join(plots_dir, "sentiment_distribution.png")),
        ]
        for idx, (label, path) in enumerate(plot_files):
            col_target = p_cols[idx % 2]
            with col_target:
                st.markdown(f"**{label}**")
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.info(f"File not found: {path}")

        # LDA Topic Modelling
        lda_path = os.path.join(BASE_DIR, "outputs", "bda_nlp_outputs", "lda_topics.txt")
        if os.path.exists(lda_path):
            st.markdown("#### Discovered LDA Topics (5 Topics)")
            with open(lda_path, "r", encoding="utf-8") as f:
                lda_content = f.read()
            st.text_area("outputs/bda_nlp_outputs/lda_topics.txt", lda_content, height=160)
    # =========================================================================
    # TAB 5: STREAMING ALGORITHMS LAB
    # =========================================================================
    with tab_streaming:
        # --- local imports (stdlib + modules already in venv) ----------------
        import math as _math
        from src.streaming_algorithms import (
            run_bloom_filter_experiment,
            run_fm_experiment,
            run_controlled_tests,
        )

        st.markdown("## 🌊 Streaming Algorithms Lab")
        st.info(
            "**Simulated Stream Notice:** This lab replays records from the "
            "static SocialPulse dataset as if they arrived in a stream.  "
            "No live social-media ingestion is performed."
        )

        # ---- Stream configuration ------------------------------------------
        st.markdown("### ⚙️ Stream Configuration")
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            stream_size = st.slider(
                "Stream size (records)", min_value=1_000, max_value=50_000,
                value=10_000, step=1_000,
                help="Number of rows to read from the dataset as the stream.",
            )
            id_column = st.selectbox(
                "Stream field", ["TweetID", "UserID"],
                help="TweetID → mostly unique items (low FPR expected).  "
                     "UserID → many repeats (higher collision, better FP demo).",
            )
        with cfg_col2:
            reproducible = st.checkbox("Reproducible (fixed seed)", value=True)
            if reproducible:
                rng_seed = st.number_input("Random seed", value=42, step=1)
            else:
                rng_seed = None

        # ---- Load stream IDs from CSV (fast, no Spark collection) -----------
        raw_csv = os.path.join(BASE_DIR, "data", "raw", "tweets-engagement-metrics.csv")
        proc_csv = os.path.join(BASE_DIR, "data", "processed", "tweets_processed.csv")
        csv_path = raw_csv if os.path.exists(raw_csv) else proc_csv

        try:
            stream_df = pd.read_csv(
                csv_path,
                usecols=[id_column] if id_column in pd.read_csv(csv_path, nrows=0).columns else None,
                nrows=stream_size,
                dtype={id_column: str},
            )
            if id_column not in stream_df.columns:
                st.error(f"Column '{id_column}' not found in dataset. Choose another field.")
                st.stop()
            stream_ids = stream_df[id_column].dropna().astype(str).tolist()
            if reproducible and rng_seed is not None:
                random.seed(int(rng_seed))
                random.shuffle(stream_ids)
        except Exception as _e:
            st.error(f"Could not load stream data: {_e}")
            st.stop()

        st.success(
            f"✅ Loaded **{len(stream_ids):,}** records from `{id_column}` "
            f"({len(set(stream_ids)):,} distinct values)."
        )

        st.divider()

        # =====================================================================
        # BLOOM FILTER
        # =====================================================================
        st.markdown("## 🔵 Feature 1 — Bloom Filter")
        st.markdown(
            """
A **Bloom Filter** is a space-efficient probabilistic data structure that tests
whether an element is a member of a set.

> ⚠️ A **positive** result means the item is **possibly present — not definitely present**.  
> A **negative** result is always correct: the item was definitely never inserted.

It uses a **bit array of size *m*** and **k independent hash functions**.  
Each `add(item)` sets k bit positions; each `query(item)` checks all k positions.
"""
        )

        bf_col1, bf_col2 = st.columns(2)
        with bf_col1:
            m_bits = st.select_slider(
                "Bit-array size (m)",
                options=[2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000],
                value=50_000,
                help="Larger m → lower false-positive rate, more memory.",
            )
        with bf_col2:
            k_hashes = st.slider(
                "Number of hash functions (k)", min_value=1, max_value=15,
                value=5,
                help="Optimal k ≈ (m/n)·ln(2).  Too many or too few increases FPR.",
            )

        if st.button("▶ Run Bloom Filter", key="run_bf"):
            with st.spinner("Running Bloom Filter experiment…"):
                bf_result = run_bloom_filter_experiment(
                    stream_ids, m_bits=m_bits, k_hashes=k_hashes
                )

            r = bf_result
            mem_kb = r["memory_bytes"] / 1024

            st.markdown("#### Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Records processed", f"{r['n_total']:,}")
            m2.metric("Items inserted (known set)", f"{r['n_inserted']:,}")
            m3.metric("Items tested (unseen set)", f"{r['n_tested']:,}")
            m4.metric("False negatives", str(r["false_negatives"]),
                      help="Must always be 0 for a correct Bloom Filter.")

            fp1, fp2, fp3, fp4 = st.columns(4)
            fp1.metric("Possibly seen (query=True on unseen)", f"{r['possibly_seen_in_test']:,}")
            fp2.metric("False positives", f"{r['false_positives']:,}")
            fp3.metric("Observed FP rate", f"{r['false_positive_rate']:.4%}")
            fp4.metric("Theoretical FP rate", f"{r['theoretical_fpr']:.4%}")

            st.markdown(f"**Memory used by bit array:** {mem_kb:.1f} KB &nbsp;|&nbsp; "
                        f"**Equivalent Python set** for {r['n_inserted']:,} strings would use "
                        f"≈ {r['n_inserted'] * 50 / 1024:.0f} KB (estimated).")

            # FPR comparison chart (uses `go` imported at top of file)
            fig_bf = go.Figure()
            fig_bf.add_bar(
                x=["Observed FP Rate", "Theoretical FP Rate"],
                y=[r["false_positive_rate"] * 100, r["theoretical_fpr"] * 100],
                marker_color=["#EF553B", "#636EFA"],
                text=[f"{r['false_positive_rate']:.3%}", f"{r['theoretical_fpr']:.3%}"],
                textposition="outside",
            )
            fig_bf.update_layout(
                title="Observed vs Theoretical False-Positive Rate (%)",
                yaxis_title="FP Rate (%)", height=350,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bf, use_container_width=True)

            with st.expander("📖 Memory-vs-Accuracy Trade-off"):
                st.markdown(
                    f"""
| Parameter | Value |
|-----------|-------|
| Bit-array size *m* | {m_bits:,} bits ({mem_kb:.1f} KB) |
| Hash functions *k* | {k_hashes} |
| Items inserted *n* | {r['n_inserted']:,} |
| Optimal k formula | k = (m/n)·ln 2 ≈ **{(m_bits / max(r['n_inserted'],1)) * _math.log(2):.1f}** |
| Theoretical FPR | {r['theoretical_fpr']:.4%} |

**Trade-off:** Increasing *m* reduces the false-positive rate but uses more memory.
Increasing *k* (up to the optimum) also reduces FPR, but each query must check more bits.
Beyond the optimal *k* the FPR rises again because too many bits are set.
"""
                )

        st.divider()

        # =====================================================================
        # FLAJOLET-MARTIN
        # =====================================================================
        st.markdown("## 📊 Feature 2 — Flajolet–Martin Approximate Distinct Count")
        st.markdown(
            """
The **Flajolet–Martin (FM) algorithm** estimates the number of *distinct* elements
in a stream using a tiny amount of memory — no set or hash-table of all elements.

**How trailing zeros inform the estimate:**
> Hash each item to a large integer.  Count the number of **trailing zero bits**
> in that integer.  A value with k trailing zeros occurs with probability 2⁻ᵏ.
> After processing ~N distinct items, it is likely that the **maximum** trailing-zero
> count observed ≈ log₂(N).  So the estimate is **2^(max trailing zeros) ≈ N**.

Using **multiple independent hash functions** and taking the **median** of their
estimates significantly reduces the variance of a single-hash estimate.
"""
        )

        fm_col1, fm_col2 = st.columns(2)
        with fm_col1:
            num_fm_hashes = st.slider(
                "Number of FM hash functions", min_value=1, max_value=32,
                value=8,
                help="More hashes → lower variance (median is more stable). "
                     "1 hash gives a very rough estimate.",
            )

        if st.button("▶ Run Flajolet–Martin", key="run_fm"):
            with st.spinner("Running Flajolet–Martin experiment…"):
                fm_result = run_fm_experiment(stream_ids, num_hashes=num_fm_hashes)

            r = fm_result
            st.markdown("#### Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stream items processed", f"{r['n_stream_items']:,}")
            c2.metric("Exact distinct count", f"{r['exact_distinct']:,}")
            c3.metric("FM estimate (median)", f"{r['fm_estimate']:,.0f}")
            c4.metric("Estimation error", f"{r['pct_error']:.1f}%",
                      delta=f"±{r['abs_error']:.0f} items",
                      delta_color="off")

            # Per-hash estimate chart (uses `go` imported at top of file)
            estimates = r["per_hash_estimates"]
            fig_fm = go.Figure()
            fig_fm.add_trace(go.Bar(
                x=[f"h{i+1}" for i in range(len(estimates))],
                y=estimates,
                name="Per-hash estimate",
                marker_color="#00CC96",
            ))
            fig_fm.add_hline(
                y=r["exact_distinct"], line_dash="dot",
                line_color="#EF553B", annotation_text="Exact distinct count",
            )
            fig_fm.add_hline(
                y=r["fm_estimate"], line_dash="dash",
                line_color="#636EFA", annotation_text="Median estimate",
            )
            fig_fm.update_layout(
                title="Per-Hash FM Estimates vs Exact Distinct Count",
                xaxis_title="Hash function", yaxis_title="Estimated distinct count",
                height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_fm, use_container_width=True)

            with st.expander("📐 Max trailing zeros per hash function"):
                tz_data = pd.DataFrame({
                    "Hash function": [f"h{i+1}" for i in range(len(r["max_trailing_zeros"]))],
                    "Max trailing zeros (R)": r["max_trailing_zeros"],
                    "Estimate 2^R": [int(2**v) for v in r["max_trailing_zeros"]],
                })
                st.dataframe(tz_data, use_container_width=True)
                st.caption(
                    "Each row shows the maximum trailing-zero count observed by that hash "
                    "function over the entire stream, and the resulting estimate 2^R."
                )

        st.divider()

        # =====================================================================
        # CONTROLLED TESTS
        # =====================================================================
        with st.expander("🧪 Run Controlled Validation Tests"):
            st.markdown(
                "Validates both algorithms on **known inputs**: empty stream, "
                "all-same IDs, all-unique IDs, and a deliberate false-positive scenario."
            )
            if st.button("Run tests", key="run_ctrl_tests"):
                with st.spinner("Running controlled tests…"):
                    ctrl_results = run_controlled_tests()
                rows = []
                for t in ctrl_results:
                    rows.append({
                        "Test case": t["name"],
                        "BF inserted": t["bf_inserted"],
                        "BF false positives": t["bf_fp"],
                        "BF false negatives": t["bf_fn"],
                        "FM exact distinct": t["fm_exact"],
                        "FM estimate": t["fm_estimate"],
                        "Pass": "✅" if t["pass"] else "❌",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                all_pass = all(t["pass"] for t in ctrl_results)
                if all_pass:
                    st.success("All controlled tests passed.")
                else:
                    st.warning("One or more tests failed — review results above.")


if __name__ == "__main__":

    main()
