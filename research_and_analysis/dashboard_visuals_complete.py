import os
import sys
import json
from pathlib import Path 
from datetime import date
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor 

# --- Matplotlib Headless Fix ---
# Must be called BEFORE importing matplotlib.pyplot
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 


# --- Load environment variables and set constants ---
load_dotenv()
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")

# --- Import data and DB helpers (Ensure these paths are correct) ---
from dashboard_data import get_dashboard_data 
from PipeLine import get_db_connection 

# Define the directory where charts will be saved
OUTPUT_DIR = "data/reports/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================================================================
# I. HELPER FUNCTIONS (Date Filter/Params)
# NOTE: These are copied from dashboard_data.py to ensure standalone functionality
# ====================================================================

def _date_range_filter(start_date: Optional[date], end_date: Optional[date]) -> str:
    """Returns SQL fragment for date range filter."""
    clauses = []
    if start_date:
        clauses.append("txn_date >= %(start_date)s")
    if end_date:
        clauses.append("txn_date <= %(end_date)s")
    if not clauses:
        return ""
    return " AND " + " AND ".join(clauses)


def _build_params(user_id: str, start_date: Optional[date], end_date: Optional[date]) -> Dict[str, Any]:
    """Builds the dictionary of parameters for the SQL query."""
    params: Dict[str, Any] = {"user_id": user_id}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    return params


# ====================================================================
# II. DATA FETCHERS FOR NEW PLOTS
# ====================================================================

def get_monthly_category_spending(conn, user_id: str) -> pd.DataFrame:
    """
    Fetches total spend grouped by month and category for the trend comparison chart.
    
    FIX: Ensure total_spend is explicitly cast to numeric for nlargest() to work.
    """
    query = """
    SELECT
        DATE_TRUNC('month', txn_date) AS month_start,
        COALESCE(category, 'Uncategorized') AS category,
        SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END) AS total_spend
    FROM transactions
    WHERE user_id = %(user_id)s
      AND amount < 0
      AND category NOT IN ('Income', 'Transfers')
      AND txn_date < '2100-01-01'
    GROUP BY 1, 2
    ORDER BY 1, 3 DESC;
    """
    params = {"user_id": user_id}
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
    
    # FIX for nlargest TypeError: Convert amount column to numeric
    if not df.empty:
        df['total_spend'] = pd.to_numeric(df['total_spend'])
    
    return df


def get_daily_spending_data(
    conn,
    user_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetches individual transactions (only outflows) for the daily scatter chart.
    """
    date_filter = _date_range_filter(start_date, end_date)
    params = _build_params(user_id, start_date, end_date)
    
    query = f"""
        SELECT
            txn_date AS date,
            -amount AS spend_amount  -- Make outflow amounts positive
        FROM transactions
        WHERE user_id = %(user_id)s
          AND amount < 0
          AND (category IS NULL OR category NOT IN ('Income', 'Transfers'))
          AND txn_date < '2100-01-01'
          {date_filter}
        ORDER BY txn_date;
    """
    
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']) 
            # Ensure spend_amount is numeric for plotting
            df['spend_amount'] = pd.to_numeric(df['spend_amount'])
        return df


# ====================================================================
# III. CHART GENERATORS (5 PLOTS)
# ====================================================================
# --- Original Plot 1: Category Pie Chart (SRS: FR-50) ---

def generate_category_pie_chart(category_spending: List[Dict[str, Any]], user_id: str) -> str:
    """
    Generates a pie chart showing spending distribution by category.
    
    Removes data labels from the pie slices and relies on the legend for small categories.
    Consolidates small slices into 'Miscellaneous'.
    """
    if not category_spending: return ""
    
    df = pd.DataFrame(category_spending)
    
    # 1. Standardize and Consolidate small categories (e.g., less than 1%)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    total_amount = df['amount'].sum()
    df['percentage'] = (df['amount'] / total_amount) * 100
    
    threshold = 1.0
    small_slices = df[df['percentage'] < threshold]
    
    if not small_slices.empty:
        misc_amount = small_slices['amount'].sum()
        df = df[df['percentage'] >= threshold].copy()
        df_misc = pd.DataFrame([
            {'category': 'Miscellaneous', 'amount': misc_amount, 'percentage': (misc_amount / total_amount) * 100}
        ])
        df = pd.concat([df, df_misc], ignore_index=True)

    # Sort and prepare data
    df = df.sort_values(by='amount', ascending=False)
    
    labels = df['category']
    sizes = df['amount']
    percentages = df['percentage'] # Use the calculated percentages for conditional labeling

    plt.figure(figsize=(12, 8))
    
    # 2. Plotting with conditional labeling and improved aesthetics
    # autopct: Displays percentage only if it's 5% or more (improving readability)
    # pctdistance: Controls how far the percentage text is from the center
    # labeldistance: Controls how far the category name label is from the center (we rely on legend)
    
    def format_percentage(pct):
        # We only want to display the percentage number itself (e.g., 43.5%)
        # and we want the category name label only for the largest slices (optional)
        return f'{pct:.1f}%' if pct >= 5 else ''

    wedges, texts, autotexts = plt.pie(
        sizes, 
        # labels=labels, <--- REMOVED LABELS HERE
        autopct=format_percentage, 
        startangle=90, 
        wedgeprops=dict(width=0.4, edgecolor='w'),
        textprops={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'},
        pctdistance=0.8 # Keeps the percentage label inside the donut slice
    )
    
    # Add category names (labels) for the largest slices directly onto the chart if needed
    # For a clean look, we often rely entirely on the legend, but will add labels 
    # to the two largest slices (Debt, Shopping) for clear visual anchoring.
    for i, (p, text) in enumerate(zip(percentages, texts)):
        if p >= 10: # Only label slices > 10% directly
            text.set_text(labels.iloc[i])
        else:
            text.set_text('') # Remove labels for smaller slices
    
    total_amount = df['amount'].sum()
    plt.title(f"User Spending Profile: Breakdown by Category (Total Spend: ${total_amount:,.0f})", fontsize=18)
    
    # 3. Use legend for all category names (including the small ones)
    plt.legend(
        labels=[f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)], 
        loc='center right', 
        bbox_to_anchor=(1.1, 0.5), 
        fontsize=10,
        title="Category Breakdown"
    )
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout for legend placement
    
    filename = f"{user_id}_category_pie.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath
    
    # 2. Plotting with improved legibility and title
    plt.pie(
        sizes, 
        labels=labels, 
        # Make data labels legible (white/black text)
        autopct=lambda p: f'{p:.1f}%%' if p > 1 else '', # Only show label if > 1%
        startangle=90, 
        wedgeprops=dict(width=0.4, edgecolor='w'),
        # Custom text properties for better contrast and legibility
        textprops={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'} 
    )
    
    # 3. Apt Chart Title (Reflecting the analysis)
    plt.title(f"User Spending Profile: Breakdown by Category", fontsize=18)
    
    plt.legend(
        labels=[f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)], 
        # --- FIX APPLIED HERE: Shifted legend to Right Bottom ---
        loc='lower left', 
        bbox_to_anchor=(1.05, 0), 
        fontsize=10,
        title="Category Breakdown"
    )
    # Changed rect=[0, 0, 1, 1] to provide space for the legend on the right
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    filename = f"{user_id}_category_pie.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

# --- Original Plot 2: Monthly Line Chart (SRS: FR-52) ---
def generate_monthly_line_chart(monthly_spending: List[Dict[str, Any]], user_id: str) -> str:
    """
    Generates a visually enhanced line chart showing monthly spending trends.
    Uses bold color, prominent markers, and includes a simple trendline for context.
    """
    if not monthly_spending: return ""
    
    df = pd.DataFrame(monthly_spending)
    
    # 1. Calculate a simple trendline (SMA over a 3-month period)
    df['trendline'] = df['amount'].rolling(window=3, min_periods=1).mean()
    
    # Sort by a numerical representation of the month to ensure correct plotting order
    # Assuming 'month' column holds abbreviated month names (Jan, Feb, etc.)
    # We rely on the input list being ordered by date, but will use index for plotting
    
    plt.figure(figsize=(12, 6))
    
    # --- Aesthetic Implementation ---
    
    # 2. Plot the Main Trend (Bold Dark Color & Large Markers)
    plt.plot(
        df['month'], 
        df['amount'], 
        marker='o',             # Circle marker
        markersize=8,           # Large marker size
        linestyle='-',          # Solid line
        color='#006400',        # Dark Green color (Bold)
        linewidth=3,            # Thicker line
        label='Total Monthly Spend (USD)'
    )
    
    # 3. Plot the Contextual Trendline (Subtle Color)
    plt.plot(
        df['month'], 
        df['trendline'], 
        marker='', 
        linestyle='-', 
        color='#77dd77',        # Light green/subtle color
        linewidth=2,
        alpha=0.6,
        label='3-Month Moving Average Trend'
    )
    
    # 4. Final Aesthetics
    
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Total Spend (USD)", fontsize=12)
    plt.title(f"Total Monthly Spending Trends (Financial Volatility)", fontsize=18)
    
    # Use finer gridlines for a clean, professional look
    plt.grid(True, linestyle=':', alpha=0.7) 
    
    # Ensure all month labels are visible
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    
    filename = f"{user_id}_monthly_trend_enhanced.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def generate_vendor_bar_chart(vendor_spending: List[Dict[str, Any]], user_id: str) -> str:
    """
    Generates a horizontal bar chart showing spending on top vendors, 
    EXCLUDING the heavily dominating 'Unknown' category to make the rest of the data legible.
    """
    if not vendor_spending: return ""
        
    df = pd.DataFrame(vendor_spending)
    
    # 1. FIX: Filter out the "Unknown" category
    # Note: We still normalize it first in case the input data contains 'Unknown'
    df['vendor'] = df['vendor'].replace({'Unknown': 'Unknown (Needs Review)'})
    df = df[df['vendor'] != 'Unknown (Needs Review)'].copy()
    
    if df.empty:
        # Return if there's no data left after filtering 'Unknown'
        return ""

    # Sort the DataFrame by amount descending (ascending for horizontal plot)
    df = df.sort_values(by='amount', ascending=True)

    plt.figure(figsize=(10, 8)) # Increased height for horizontal bars
    
    # Use a single, consistent color for all known vendors
    bars = plt.barh(
        df['vendor'], 
        df.apply(lambda x: x['amount'] / 1000, axis=1), # Plot in thousands
        color='#4682B4' 
    )
    
    # Calculate max spend in thousands for the axis limit
    max_spend_thousands = df['amount'].max() / 1000
    
    # 2. Add Data Labels (Legibility Improvement)
    for bar in bars:
        width = bar.get_width()
        
        # Original value (for display in dollars)
        original_amount = width * 1000 
        
        # Format the label to show full dollar amount
        label = f'${original_amount:,.0f}'
        
        plt.text(
            width, # X position (end of the bar)
            bar.get_y() + bar.get_height() / 2, # Y position (center of the bar)
            label, 
            ha='left', 
            va='center',
            fontsize=9
        )
    
    plt.xlabel("Total Spend (in Thousands USD)")
    plt.ylabel("Vendor")
    # Update the title to reflect that the primary outlier has been filtered
    plt.title(f"Top {len(df)} Vendor Spending Breakdown (Excluding Uncategorized Outliers)", fontsize=16)
    
    # Set the x-axis maximum slightly higher to accommodate the labels
    plt.xlim(0, max_spend_thousands * 1.1) 
    
    plt.tight_layout()
    
    filename = f"{user_id}_vendor_bar.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath


# --- New Plot 4: Category Trend Comparison (SRS: FR-31) ---
def generate_category_trend_comparison(df: pd.DataFrame, user_id: str) -> str:
    """
    Generates a grouped bar chart comparing top category spending across months, 
    useful for trend analysis.
    """
    if df.empty: return ""

    # Filter for the top 5 categories to keep the chart readable
    top_categories = df.groupby('category')['total_spend'].sum().nlargest(5).index
    df_filtered = df[df['category'].isin(top_categories)]

    plt.figure(figsize=(14, 7))
    sns.barplot(
        x=df_filtered['month_start'].dt.strftime('%Y-%m'), 
        y=df_filtered['total_spend'], 
        hue=df_filtered['category']
    )
    plt.title(f"Monthly Spend Comparison: Top 5 Categories ({user_id})", fontsize=16)
    plt.xlabel("Month")
    plt.ylabel("Total Spend")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category', loc='upper left')
    plt.tight_layout()
    
    filename = f"{user_id}_category_trend_comparison.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath


# --- New Plot 5: Daily Volume & Value Scatter ---
def generate_daily_scatter_chart(df_daily: pd.DataFrame, user_id: str) -> str:
    """
    Generates a scatter plot showing daily transaction volume vs. magnitude, 
    with enhanced visual appeal and clarity for anomaly detection.
    """
    if df_daily.empty: return ""
        
    # Aggregate data by day for volume/count analysis
    daily_summary = df_daily.groupby('date').agg(
        total_spent=('spend_amount', 'sum'),
        txn_count=('spend_amount', 'count')
    ).reset_index()

    plt.figure(figsize=(14, 6))
    
    # 1. Plot Daily Total Spend (Changed from bar to line for cleaner trend view)
    # Use a second, thicker line to represent the total daily trend
    plt.plot(
        daily_summary['date'], 
        daily_summary['total_spent'], 
        color='#778899', # Slate Gray for a softer background look
        linewidth=2.5, 
        alpha=0.6,
        label='Daily Total Spend'
    )
    
    # 2. Plot Individual Transactions (Scatter/Bubble Plot)
    plt.scatter(
        df_daily['date'], 
        df_daily['spend_amount'], 
        s=df_daily['spend_amount'] * 0.4, # Slightly reduced size for less overlap
        color='#4B0082', # Indigo/Dark Violet for distinct anomaly highlights
        alpha=0.7,
        edgecolors='white', # White border for contrast
        linewidth=0.5,
        label='Individual Transaction Value'
    )
    
    # 3. Aesthetic Improvements
    
    # Clean Title
    plt.title("Daily Spending Volume & Value Analysis (Anomaly Detector)", fontsize=18)
    plt.xlabel("Date")
    plt.ylabel("Transaction Value (USD)")
    
    # Zoom Y-Axis: Limit to $13,000 to better show the anomaly spread above the fixed debt
    # The fixed debt transactions are around $6,000, and outliers go to $16,000.
    plt.ylim(-500, df_daily['spend_amount'].max() * 1.05) 
    
    plt.grid(True, axis='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{user_id}_daily_volume_value_enhanced.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

# ====================================================================
# IV. MAIN INTEGRATION FUNCTION
# ====================================================================

def generate_dashboard_visuals(
    user_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Dict[str, str]:
    """
    Fetches aggregated data, generates all 5 visualization files, and returns 
    a dictionary of saved file paths.
    """
    uid = user_id or os.getenv("DEFAULT_USER_ID")
    if not uid:
        raise RuntimeError("User ID not provided and DEFAULT_USER_ID not set")
    
    print(f"\n[Visuals] Generating charts for user: {uid}")
    
    # Establish connection for data fetching
    conn = get_db_connection() 
    
    try:
        # 1. Fetch Aggregated Data (from dashboard_data.py)
        dashboard_data = get_dashboard_data(uid, start_date, end_date)
        
        # 2. Fetch Granular Data for new plots
        # FIX: The missing function is now defined above.
        monthly_category_data = get_monthly_category_spending(conn, uid)
        daily_data = get_daily_spending_data(conn, uid, start_date, end_date)
        
        category_spending = dashboard_data["categorySpending"]
        monthly_spending = dashboard_data["monthlySpending"]
        vendor_spending = dashboard_data["vendorSpending"]
        
        # 3. Generate and save the charts
        chart_paths = {}
        
        chart_paths["category_pie"] = generate_category_pie_chart(category_spending, uid)
        chart_paths["monthly_trend"] = generate_monthly_line_chart(monthly_spending, uid)
        chart_paths["vendor_bar"] = generate_vendor_bar_chart(vendor_spending, uid)
        
        # New Financial Analysis Plots
        chart_paths["category_trend_comparison"] = generate_category_trend_comparison(monthly_category_data, uid)
        chart_paths["daily_scatter"] = generate_daily_scatter_chart(daily_data, uid)

    finally:
        conn.close()
        
    print(f"[Visuals] Charts saved to: {OUTPUT_DIR}")
    
    return chart_paths


# ====================================================================
# V. CLI Debug Helper
# ====================================================================

if __name__ == "__main__":
    """
    Runs the visualization generation and prints the resulting file paths.
    """
    # NOTE: This requires a running PostgreSQL database with the necessary data.
    
    try:
        # Attempt a dummy connection check
        conn = get_db_connection()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Could not connect to DB. Cannot run visualization demo: {e}")
        sys.exit(1)
        
    visual_paths = generate_dashboard_visuals()
    print("\nGenerated Chart Paths (PNG):")
    print(json.dumps(visual_paths, indent=2))