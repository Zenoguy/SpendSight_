
import os
import sys
import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import date
from typing import Any, Dict, List, Optional
from pathlib import Path

matplotlib.use('Agg')

from PipeLine import get_db_connection
from dotenv import load_dotenv
from psycopg2.extras import DictCursor 

load_dotenv()
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")

OUTPUT_DIR = "data/reports/pipeline_metrics"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ====================================================================
# I. DATA FETCH (Normalized to `count`)
# ====================================================================
def fetch_pipeline_metrics(conn) -> Dict[str, Any]:
    with conn.cursor(cursor_factory=DictCursor) as cur:

        # Total
        cur.execute("SELECT COUNT(*) AS count FROM transactions;")
        total = cur.fetchone()["count"]

        # By-source breakdown
        cur.execute("""
            SELECT COALESCE(classification_source::text, 'NULL') AS source,
                   COUNT(*) AS count
            FROM transactions
            GROUP BY source;
        """)
        rows = cur.fetchall()
        by_source = [{'source': row['source'], 'count': row['count']} for row in rows]

        # Per-stage routing counts
        regex_count = next((r["count"] for r in by_source if r["source"] == "regex"), 0)
        heur_count  = next((r["count"] for r in by_source if r["source"] == "heuristic"), 0)
        bert_count  = next((r["count"] for r in by_source if r["source"] == "bert"), 0)
        llm_count   = next((r["count"] for r in by_source if r["source"] == "llm"), 0)

        # Strong confidence metrics
        cur.execute("""
            SELECT COUNT(*) AS count
            FROM transactions
            WHERE classification_source = 'regex'
              AND confidence >= 0.8;
        """)
        regex_strong = cur.fetchone()["count"]

        cur.execute("""
            SELECT COUNT(*) AS count
            FROM transactions
            WHERE classification_source = 'heuristic'
              AND confidence >= 0.65;
        """)
        heuristic_strong = cur.fetchone()["count"]

        cur.execute("""
            SELECT COUNT(*) AS count
            FROM transactions
            WHERE classification_source = 'bert'
              AND confidence >= 0.6;
        """)
        bert_strong = cur.fetchone()["count"]

        cur.execute("""
            SELECT COUNT(*) AS count
            FROM transactions
            WHERE classification_source = 'llm'
              AND confidence >= 0.9;
        """)
        llm_strong = cur.fetchone()["count"]

        # Temporal Pending / UNCLEAR
        cur.execute("""
            SELECT 
                DATE_TRUNC('month', created_at) AS month_start,
                COUNT(*) AS pending_count
            FROM transactions
            WHERE category IS NULL
               OR category = 'PENDING'
               OR category = 'UNCLEAR'
            GROUP BY month_start
            ORDER BY month_start;
        """)
        rows = cur.fetchall()
        pending_by_month = [
            {
                "month_start": row["month_start"],
                "pending_count": row["pending_count"],
            }
            for row in rows
        ]

    return {
        "total": total,
        "by_source": by_source,
        "regex": regex_count,
        "heuristic": heur_count,
        "bert": bert_count,
        "llm": llm_count,
        "regex_strong": regex_strong,
        "heuristic_strong": heuristic_strong,
        "bert_strong": bert_strong,
        "llm_strong": llm_strong,
        "pending_by_month": pending_by_month,
    }


# ====================================================================
# II. VISUALIZATION FUNCTIONS
# ====================================================================

def plot_workload_distribution(metrics: Dict[str, Any], user_id: str) -> str:
    df = pd.DataFrame(metrics["by_source"])
    if df.empty: return ""

    df['source'] = df['source'].replace({
        'regex': 'Regex',
        'heuristic': 'Heuristic',
        'bert': 'MiniLM',
        'llm': 'LLM'
    })

    plt.figure(figsize=(9, 6))
    
    sns.barplot(
        x="source",
        y="count",
        hue="source",
        data=df,
        dodge=False,
        legend=False,
        palette={
            "Regex": "green",
            "Heuristic": "lightgreen",
            "MiniLM": "skyblue",
            "LLM": "salmon",
        },
    )


    plt.title("Classification Workload Distribution", fontsize=16)
    plt.xlabel("Classification Stage")
    plt.ylabel("Transactions Processed")
    plt.tight_layout()

    filename = f"{user_id}_workload_distribution.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath); plt.close()
    return filepath

def plot_pipeline_funnel(metrics: Dict[str, Any], user_id: str) -> str:
    """
    Stage-by-stage funnel:

      Stage 1: Total Input
      Stage 2: After Regex          (still unresolved after regex)
      Stage 3: After Heuristic      (unresolved after regex + heuristic)
      Stage 4: After MiniLM (BERT)  (unresolved after regex + heuristic + bert)
      Stage 5: After LLM            (final pending)

    Uses counts derived from final classification_source:
      - metrics["regex"]      = resolved at regex
      - metrics["heuristic"]  = resolved at heuristic
      - metrics["bert"]       = resolved at MiniLM
      - metrics["llm"]        = resolved at LLM
    """

    total      = metrics["total"]
    n_regex    = metrics["regex"]
    n_heur     = metrics["heuristic"]
    n_bert     = metrics["bert"]
    n_llm      = metrics["llm"]

    # Remaining workload after each stage
    after_regex      = total - n_regex
    after_heuristic  = total - n_regex - n_heur
    after_bert       = total - n_regex - n_heur - n_bert
    after_llm        = total - n_regex - n_heur - n_bert - n_llm  # final pending

    funnel = pd.DataFrame({
        "Stage": [
            "1. Total Input",
            "2. After Regex",
            "3. After Heuristic",
            "4. After MiniLM",
            "5. After LLM (Pending)",
        ],
        "Remaining": [
            total,
            after_regex,
            after_heuristic,
            after_bert,
            after_llm,
        ],
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(
        y="Stage",
        x="Remaining",
        hue="Stage",
        data=funnel,
        dodge=False,
        legend=False,
        palette="viridis",
    )

    # Add value labels at end of bars
    for idx, row in funnel.iterrows():
        plt.text(
            row["Remaining"] + max(total * 0.01, 1),  # small offset
            idx,
            str(row["Remaining"]),
            va="center",
        )

    plt.title("Stage-by-Stage Pipeline Workload Funnel", fontsize=16)
    plt.xlabel("Remaining Transactions")
    plt.ylabel("Pipeline Stage")
    plt.tight_layout()

    filename = f"{user_id}_pipeline_funnel.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath


def plot_quality_cost_comparison(metrics: Dict[str, Any], user_id: str) -> str:
    """
    Compare how many txns each stage handled vs how many were 'strong' (high confidence).

    Stages:
      - Regex
      - Heuristic
      - MiniLM (BERT)
      - LLM
    """
    df = pd.DataFrame({
        "Source": ["Regex", "Heuristic", "MiniLM", "LLM"],
        "Handled": [
            metrics["regex"],
            metrics["heuristic"],
            metrics["bert"],
            metrics["llm"],
        ],
        "Strong": [
            metrics["regex_strong"],
            metrics["heuristic_strong"],
            metrics["bert_strong"],
            metrics["llm_strong"],
        ],
    })

    melted = df.melt(
        id_vars="Source",
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Source",
        y="Value",
        hue="Metric",
        data=melted,
        palette=["#4c72b0", "#dd8452"],  # Handled=blue, Strong=orange
    )

    plt.title("Quality vs Cost Comparison by Stage", fontsize=16)
    plt.xlabel("Stage")
    plt.ylabel("Transaction Count")
    plt.tight_layout()

    filename = f"{user_id}_roi_comparison.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_temporal_failure_rate(metrics: Dict[str, Any], user_id: str) -> str:
    df = pd.DataFrame(metrics["pending_by_month"])
    if df.empty or "month_start" not in df.columns:
        return ""

    df["month_start"] = pd.to_datetime(df["month_start"])


    plt.figure(figsize=(12, 6))
    sns.lineplot(x='month_start', y='pending_count', data=df, marker='o', color='red')

    plt.title("Pending Transactions Over Time", fontsize=16)
    plt.xlabel("Month")
    plt.ylabel("Pending Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    filename = f"{user_id}_temporal_failure.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath); plt.close()
    return filepath


# ====================================================================
# III. MAIN ENTRYPOINT
# ====================================================================

def generate_pipeline_visuals(user_id: Optional[str] = None) -> Dict[str, str]:
    uid = user_id or DEFAULT_USER_ID

    conn = get_db_connection()
    try:
        metrics = fetch_pipeline_metrics(conn)
        if metrics["total"] == 0:
            print("[WARN] No transactions found. Skipping.")
            return {}

        paths = {
            "workload_distribution": plot_workload_distribution(metrics, uid),
            "pipeline_funnel": plot_pipeline_funnel(metrics, uid),
            "roi_comparison": plot_quality_cost_comparison(metrics, uid)
        }

        if metrics["pending_by_month"]:
            paths["temporal_failure"] = plot_temporal_failure_rate(metrics, uid)

        return paths

    finally:
        conn.close()


if __name__ == "__main__":
    visuals = generate_pipeline_visuals()
    print(json.dumps(visuals, indent=2))