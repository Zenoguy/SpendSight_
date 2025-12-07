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

# --- Matplotlib Headless Fix: Must be called BEFORE importing pyplot ---
matplotlib.use('Agg') 

# --- Import DB helpers (Assuming these are available in your environment) ---
from PipeLine import get_db_connection
from dotenv import load_dotenv
from psycopg2.extras import DictCursor 

load_dotenv()
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
OUTPUT_DIR = "data/reports/pipeline_metrics"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ====================================================================
# I. DATA FETCHING (Corrected for KeyError)
# ====================================================================

def fetch_pipeline_metrics(conn) -> Dict[str, Any]:
    """
    Fetches all necessary data points for visualization from the DB.
    FIX: Uses list comprehension to guarantee 'count' key mapping for Pandas.
    """
    
    from psycopg2.extras import DictCursor
    
    with conn.cursor(cursor_factory=DictCursor) as cur:
        
        # Total
        cur.execute("SELECT COUNT(*) AS count FROM transactions;")
        total = cur.fetchone()["count"]

        # 1. Classification Source Breakdown (Routing)
        cur.execute("""
            SELECT COALESCE(classification_source::text, 'NULL') AS source, COUNT(*) AS count
            FROM transactions
            GROUP BY source;
        """)
        # FIX: Explicit conversion to list of standard Python dicts to ensure column access
        by_source_fetched = cur.fetchall()
        by_source = [{'source': row['source'], 'count': row['count']} for row in by_source_fetched]
        
        # 2. Strong Confidence Counts (Quality Metric)
        # We fetch metrics using the explicit alias 'count'
        cur.execute("SELECT COUNT(*) AS count FROM transactions WHERE classification_source = 'regex' AND confidence >= 0.8;")
        regex_strong = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) AS count FROM transactions WHERE classification_source = 'bert' AND confidence >= 0.6;")
        bert_strong = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) AS count FROM transactions WHERE classification_source = 'llm' AND confidence >= 0.9;")
        llm_strong = cur.fetchone()["count"]
        
        # 3. Workload Passed (for Funnel)
        llm_handled = next((row['count'] for row in by_source if row['source'] == 'llm'), 0)
        bert_handled = next((row['count'] for row in by_source if row['source'] == 'bert'), 0)
        regex_heur_handled = sum(row['count'] for row in by_source if row['source'] in ('regex', 'heuristic'))
        
        
        # 4. Temporal Failure Analysis (Maintenance Plot)
        cur.execute("""
            SELECT 
                DATE_TRUNC('month', created_at) AS month_start,
                COUNT(*) AS pending_count
            FROM transactions
            WHERE category = 'PENDING' OR category IS NULL OR category = 'UNCLEAR'
            GROUP BY month_start
            ORDER BY month_start;
        """)
        pending_by_month: List[Dict[str, Any]] = cur.fetchall()

    return {
        "total": total,
        "by_source": by_source, # This list now contains 'count' keys guaranteed
        "regex_strong": regex_strong,
        "bert_strong": bert_strong,
        "llm_strong": llm_strong,
        "llm_handled": llm_handled,
        "bert_handled": bert_handled,
        "regex_heur_handled": regex_heur_handled,
        "pending_by_month": pending_by_month
    }


# ====================================================================
# II. VISUALIZATION FUNCTIONS (Using 'count' consistently)
# ====================================================================

def plot_workload_distribution(metrics: Dict[str, Any], user_id: str) -> str:
    """
    1. Plots the Classification Workload Distribution (Stacked Bar).
    Validates cost efficiency.
    """
    df = pd.DataFrame(metrics["by_source"])
    if df.empty: return ""
    
    # Use 'count' column
    df = df[df['count'] > 0]
    
    plt.figure(figsize=(8, 6))
    
    # Rename sources for better plotting labels
    df['source'] = df['source'].replace({'bert': 'MiniLM', 'llm': 'LLM Fallback', 'regex': 'Regex', 'heuristic': 'Heuristic'})

    sns.barplot(
        x='source', 
        y='count', 
        data=df, 
        palette={'MiniLM': 'skyblue', 'LLM Fallback': 'salmon', 'Regex': 'green', 'Heuristic': 'lightgreen'},
        order=['Regex', 'Heuristic', 'MiniLM', 'LLM Fallback']
    )
    
    plt.title("Classification Workload Distribution", fontsize=16)
    plt.xlabel("Classification Source (Cost Proxy)")
    plt.ylabel("Transactions Handled")
    plt.tight_layout()
    
    filename = f"{user_id}_workload_distribution.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath


# In pipeline_visuals.py

def plot_pipeline_funnel(metrics: Dict[str, Any], user_id: str) -> str:
    """
    2. Plots the Pipeline Funnel Chart (Workload Reduction).
    Uses sequential stages to visualize the cascading efficiency.
    """
    
    total = metrics["total"]
    
    # Calculate the remaining volume at each sequential transition point
    volume_entering_minilm = total - metrics["regex_heur_handled"]
    volume_entering_llm = volume_entering_minilm - metrics["bert_handled"]
    final_pending = volume_entering_llm - metrics["llm_handled"] # Should be 0

    # Data structure for the funnel (Volume entering the next stage)
    data = pd.DataFrame({
        'Stage': ['1. Input/Regex', '2. Entering MiniLM', '3. Entering LLM', '4. Final Pending'],
        'Volume': [total, volume_entering_minilm, volume_entering_llm, final_pending],
    })
    
    # We will use a bar chart to represent the funnel shape visually
    plt.figure(figsize=(10, 6))
    
    # Use a color map to highlight the reduction
    sns.barplot(
        x='Volume', 
        y='Stage', 
        data=data, 
        palette='magma', # Uses a darkening color scale for visual effect
        order=data['Stage'].tolist()
    )
    
    # Add labels to the bars for clarity
    for index, row in data.iterrows():
        plt.text(row.Volume + 10, index, f"{row.Volume}", color='black', ha="left", va="center")

    plt.title("Pipeline Workload Reduction Funnel (Transaction Volume)", fontsize=16)
    plt.xlabel("Transactions Volume (Workload)")
    plt.ylabel("Classification Stage")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.xlim(0, total + total * 0.1) # Set max limit slightly above total for labels
    plt.tight_layout()
    
    filename = f"{user_id}_pipeline_funnel.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def plot_quality_cost_comparison(metrics: Dict[str, Any], user_id: str) -> str:
    """
    3. Plots Accuracy vs. Cost Comparison (Grouped Bar Chart).
    Evaluates ROI and strong confidence generation.
    """
    # Create the data structure for plotting
    data = {
        'Source': ['Regex/Heuristic', 'MiniLM', 'LLM Fallback'],
        'Total Handled (Cost Proxy)': [metrics["regex_heur_handled"], metrics["bert_handled"], metrics["llm_handled"]],
        'Strong Confidence (Quality)': [metrics["regex_strong"], metrics["bert_strong"], metrics["llm_strong"]]
    }
    df = pd.DataFrame(data)
    
    df_melt = df.melt(id_vars='Source', var_name='Metric', value_name='Count')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Source', 
        y='Count', 
        hue='Metric', 
        data=df_melt,
        palette=['#4c72b0', '#dd8452'] # Blue for Handled, Orange for Strong
    )
    
    plt.title("Classification ROI: Volume Handled vs. Strong Confidence", fontsize=16)
    plt.ylabel("Transaction Count")
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    filename = f"{user_id}_roi_comparison.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath


def plot_temporal_failure_rate(metrics: Dict[str, Any], user_id: str) -> str:
    """
    4. Plots Temporal Failure Rate (Line Chart).
    Monitors maintenance and model drift over time.
    """
    df = pd.DataFrame(metrics["pending_by_month"])
    if df.empty: return ""
    
    # Ensure month_start is datetime for plotting
    df['month_start'] = pd.to_datetime(df['month_start'])
    
    plt.figure(figsize=(12, 6))
    
    # Use a line plot to show trend
    sns.lineplot(
        x=df['month_start'], 
        y=df['pending_count'], 
        marker='o', 
        color='red', 
        linewidth=2
    )
    
    plt.title("Final Pending Transactions Over Time (Model Drift)", fontsize=16)
    plt.xlabel("Month")
    plt.ylabel("Final Pending Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{user_id}_temporal_failure.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath


# ====================================================================
# III. MAIN EXECUTION
# ====================================================================

def generate_pipeline_visuals(user_id: Optional[str] = None) -> Dict[str, str]:
    """
    Main function to orchestrate data fetching and plot generation.
    """
    uid = user_id or DEFAULT_USER_ID
    if not uid:
        raise RuntimeError("User ID not provided and DEFAULT_USER_ID not set")
    
    print(f"\n[Visuals] Generating pipeline metrics for user: {uid}")
    
    conn = get_db_connection()
    try:
        metrics = fetch_pipeline_metrics(conn)
        
        if metrics["total"] == 0:
            print("[WARN] No transactions found in DB. Skipping visualization.")
            return {}

        paths = {}
        paths["workload_distribution"] = plot_workload_distribution(metrics, uid)
        paths["pipeline_funnel"] = plot_pipeline_funnel(metrics, uid)
        paths["roi_comparison"] = plot_quality_cost_comparison(metrics, uid)
        
        if metrics["pending_by_month"]:
            paths["temporal_failure"] = plot_temporal_failure_rate(metrics, uid)
            
    finally:
        conn.close()
        
    print(f"[Visuals] Pipeline metrics saved to: {OUTPUT_DIR}")
    return paths

if __name__ == "__main__":
    # Ensure client is initialized (if needed for DB connection helper)
    try:
        conn = get_db_connection()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Could not connect to DB. Cannot run visualization demo: {e}")
        sys.exit(1)
        
    visual_paths = generate_pipeline_visuals()
    print("\nGenerated Metric Chart Paths (PNG):")
    print(json.dumps(visual_paths, indent=2))