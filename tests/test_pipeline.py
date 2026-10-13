"""
Focused validation tests for SocialPulse-Analytics.
Verifies text cleaning, sentiment classification thresholds,
schema integrity, and pipeline execution.
"""

import os
import sys
import unittest
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import clean_text, classify_sentiment, analyze_sentiment_single
from src.spark_pipeline import (
    DataProcessingPipeline,
    check_spark_environment,
    RAW_SCHEMA,
    PYSPARK_AVAILABLE
)


class TestTextPreprocessing(unittest.TestCase):
    def test_url_removal(self):
        text = "Check this http://example.com and https://t.co/abc and www.test.org"
        cleaned = clean_text(text)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("www", cleaned)
        self.assertNotIn("example.com", cleaned)

    def test_mention_and_hashtag_removal(self):
        text = "@twitteruser #awesome #bigdata great project"
        cleaned = clean_text(text)
        self.assertNotIn("@twitteruser", cleaned)
        self.assertNotIn("#awesome", cleaned)
        self.assertNotIn("#bigdata", cleaned)
        self.assertIn("great", cleaned)
        self.assertIn("project", cleaned)

    def test_special_characters_and_digits(self):
        text = "Social 123 analytics! &*() 2026 test"
        cleaned = clean_text(text)
        self.assertNotIn("123", cleaned)
        self.assertNotIn("2026", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertIn("social", cleaned)
        self.assertIn("analytics", cleaned)

    def test_stopwords_removal(self):
        text = "This is a good day and the sun is shining"
        cleaned = clean_text(text)
        tokens = cleaned.split()
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)
        self.assertIn("good", tokens)


class TestSentimentClassification(unittest.TestCase):
    def test_threshold_positive(self):
        self.assertEqual(classify_sentiment(0.05), "Positive")
        self.assertEqual(classify_sentiment(0.85), "Positive")

    def test_threshold_negative(self):
        self.assertEqual(classify_sentiment(-0.05), "Negative")
        self.assertEqual(classify_sentiment(-0.65), "Negative")

    def test_threshold_neutral(self):
        self.assertEqual(classify_sentiment(0.049), "Neutral")
        self.assertEqual(classify_sentiment(-0.049), "Neutral")
        self.assertEqual(classify_sentiment(0.0), "Neutral")

    def test_vader_integration(self):
        res_pos = analyze_sentiment_single("This is absolutely wonderful and great")
        self.assertEqual(res_pos["sentiment_label"], "Positive")
        self.assertGreaterEqual(res_pos["compound_score"], 0.05)

        res_neg = analyze_sentiment_single("This is terrible horrible and very bad")
        self.assertEqual(res_neg["sentiment_label"], "Negative")
        self.assertLessEqual(res_neg["compound_score"], -0.05)


class TestSchemaAndPipeline(unittest.TestCase):
    def setUp(self):
        self.raw_path = os.path.join(REPO_ROOT, "data", "raw", "tweets-engagement-metrics.csv")
        self.proc_path = os.path.join(REPO_ROOT, "data", "processed", "tweets_processed.csv")

    def test_raw_schema_columns(self):
        if os.path.exists(self.raw_path):
            df_raw = pd.read_csv(self.raw_path, nrows=5)
            self.assertEqual(len(df_raw.columns), 20)
            self.assertIn("text", df_raw.columns)
            self.assertIn("Likes", df_raw.columns)
            self.assertIn("RetweetCount", df_raw.columns)

    def test_processed_schema_columns(self):
        if os.path.exists(self.proc_path):
            df_proc = pd.read_csv(self.proc_path, nrows=5)
            self.assertEqual(len(df_proc.columns), 24)
            for c in ["cleaned_text", "vader_scores", "compound_score", "sentiment_label"]:
                self.assertIn(c, df_proc.columns)

    def test_pipeline_end_to_end_local_sample(self):
        if os.path.exists(self.raw_path):
            pipeline = DataProcessingPipeline(prefer_spark=False)
            df_sample = pd.read_csv(self.raw_path, nrows=20)
            result_df = pipeline.run_sentiment_pipeline(df_sample)
            self.assertEqual(len(result_df), 20)
            self.assertIn("cleaned_text", result_df.columns)
            self.assertIn("compound_score", result_df.columns)
            self.assertIn("sentiment_label", result_df.columns)
            self.assertTrue(set(result_df["sentiment_label"].unique()).issubset({"Positive", "Neutral", "Negative"}))

    def test_pipeline_end_to_end_spark_sample(self):
        if os.path.exists(self.raw_path):
            pipeline = DataProcessingPipeline(prefer_spark=True)
            if pipeline.is_spark_active():
                result_df = pipeline.run_sentiment_pipeline(self.raw_path, sample_size=30)
                self.assertEqual(len(result_df), 30)
                self.assertIn("cleaned_text", result_df.columns)
                self.assertIn("vader_scores", result_df.columns)
                self.assertIn("compound_score", result_df.columns)
                self.assertIn("sentiment_label", result_df.columns)
                self.assertIn("log1p_Likes", result_df.columns)
                self.assertIn("log1p_RetweetCount", result_df.columns)
                self.assertTrue(set(result_df["sentiment_label"].unique()).issubset({"Positive", "Neutral", "Negative"}))
                from src.spark_pipeline import SparkSessionManager
                SparkSessionManager.stop_session()

    def test_spark_environment_check(self):
        env = check_spark_environment()
        self.assertIn("pyspark_installed", env)
        self.assertIn("java_installed", env)
        self.assertIn("is_compatible", env)
        self.assertTrue(env["is_compatible"])


if __name__ == "__main__":
    unittest.main()
