# dashboard_data.py
"""
Backend data generation for SpendSight dashboard.

Provides aggregated data for:
- Spending by category
- Monthly spending
- Top vendors
- Summary stats

Shapes are compatible with:
  mockCategorySpending, mockMonthlySpending, mockVendorSpending
used in the React dashboard.
"""

import os
import json
from datetime import date
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from psycopg2.extras import DictCursor

from PipeLine import get_db_connection  # reuse your existing DB helper

load_dotenv()
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")


# ------------- Helpers ------------- #

def _date_range_filter(start_date: Optional[date], end_date: Optional[date]) -> str:
    """
    Returns SQL fragment for date range. Adjust column name if needed
    (assumes txn_date is the transaction date column).
    """
    clauses = []
    if start_date:
        clauses.append("txn_date >= %(start_date)s")
    if end_date:
        clauses.append("txn_date <= %(end_date)s")
    if not clauses:
        return ""  # no filter
    return " AND " + " AND ".join(clauses)


def _build_params(user_id: str, start_date: Optional[date], end_date: Optional[date]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"user_id": user_id}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    return params


# ------------- Core aggregation functions ------------- #

def get_category_spending(conn, user_id: str,
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> List[Dict[str, Any]]:
    """
    Returns list of:
      { category: str, amount: float, percentage: float }
    Only considers "spend" transactions (amount < 0) and excludes clear income categories.
    """
    date_filter = _date_range_filter(start_date, end_date)
    params = _build_params(user_id, start_date, end_date)

    with conn.cursor(cursor_factory=DictCursor) as cur:
        # Sum negative amounts (expenses). We take ABS later in code.
        cur.execute(f"""
            SELECT
                COALESCE(category, 'Uncategorized') AS category,
                SUM(amount) AS total_amount
            FROM transactions
            WHERE user_id = %(user_id)s
              AND amount < 0
              AND (category IS NULL OR category NOT IN ('Income', 'Transfers'))
              {date_filter}
            GROUP BY COALESCE(category, 'Uncategorized')
            ORDER BY SUM(amount) ASC;  -- more negative = more spend
        """, params)

        rows = cur.fetchall()

    # Convert to positive "spent" values and percentages
    totals = [abs(row["total_amount"] or 0) for row in rows]
    total_spent = sum(totals) or 1.0  # avoid division by zero

    result = []
    for row in rows:
        amt = abs(row["total_amount"] or 0)
        pct = (amt / total_spent) * 100.0
        result.append({
            "category": row["category"],
            "amount": float(amt),
            "percentage": pct,
        })

    return result


def get_monthly_spending(conn, user_id: str,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> List[Dict[str, Any]]:
    """
    Returns list of:
      { month: 'Jan', amount: float }
    where amount is total spending in that month (positive number).
    """
    date_filter = _date_range_filter(start_date, end_date)
    params = _build_params(user_id, start_date, end_date)

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(f"""
            SELECT
                date_trunc('month', txn_date)::date AS month_start,
                SUM(amount) AS total_amount
            FROM transactions
            WHERE user_id = %(user_id)s
              AND amount < 0
              AND (category IS NULL OR category NOT IN ('Income', 'Transfers'))
              {date_filter}
            GROUP BY date_trunc('month', txn_date)::date
            ORDER BY month_start;
        """, params)

        rows = cur.fetchall()

    result = []
    for row in rows:
        month_start = row["month_start"]  # date
        amt = abs(row["total_amount"] or 0)
        # 'Jan', 'Feb', etc. You can change format if needed.
        month_label = month_start.strftime("%b")
        result.append({
            "month": month_label,
            "amount": float(amt),
        })
    return result


def get_top_vendors(conn, user_id: str,
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None,
                    limit: int = 10) -> List[Dict[str, Any]]:
    """
    Returns list of:
      { vendor: str, amount: float, transactions: int }

    'amount' here is total spent with that vendor (positive number).
    """
    date_filter = _date_range_filter(start_date, end_date)
    params = _build_params(user_id, start_date, end_date)
    params["limit"] = limit

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(f"""
            SELECT
                COALESCE(vendor, 'Unknown') AS vendor,
                SUM(amount) AS total_amount,
                COUNT(*) AS txn_count
            FROM transactions
            WHERE user_id = %(user_id)s
              AND amount < 0
              AND (category IS NULL OR category NOT IN ('Income', 'Transfers'))
              {date_filter}
            GROUP BY COALESCE(vendor, 'Unknown')
            ORDER BY SUM(amount) ASC   -- more negative = more spend
            LIMIT %(limit)s;
        """, params)

        rows = cur.fetchall()

    result = []
    for row in rows:
        amt = abs(row["total_amount"] or 0)
        result.append({
            "vendor": row["vendor"],
            "amount": float(amt),
            "transactions": int(row["txn_count"]),
        })
    return result


def get_summary_stats(conn, user_id: str,
                      start_date: Optional[date] = None,
                      end_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Returns:
      {
        "total_spent": float,
        "total_transactions": int,
        "top_category": { category, amount, percentage } or None
      }
    """
    category_spending = get_category_spending(conn, user_id, start_date, end_date)
    total_spent = sum(item["amount"] for item in category_spending)
    total_transactions = 0

    date_filter = _date_range_filter(start_date, end_date)
    params = _build_params(user_id, start_date, end_date)

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(f"""
            SELECT COUNT(*) AS c
            FROM transactions
            WHERE user_id = %(user_id)s
              {date_filter};
        """, params)
        row = cur.fetchone()
        total_transactions = row["c"]

    top_category = None
    if category_spending:
        top_category = max(category_spending, key=lambda x: x["amount"])

    return {
        "total_spent": float(total_spent),
        "total_transactions": int(total_transactions),
        "top_category": top_category,
    }


# ------------- Public API for Person C ------------- #

def get_dashboard_data(user_id: Optional[str] = None,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Main entrypoint for dashboard aggregation.

    Returns:
      {
        "summary": { ... },
        "categorySpending": [ {category, amount, percentage} ],
        "monthlySpending": [ {month, amount} ],
        "vendorSpending": [ {vendor, amount, transactions} ]
      }

    Keys match your React mock data shapes:
      - mockCategorySpending
      - mockMonthlySpending
      - mockVendorSpending
    """

    uid = user_id or DEFAULT_USER_ID
    if not uid:
        raise RuntimeError("User ID not provided and DEFAULT_USER_ID not set")

    conn = get_db_connection()
    try:
        category_spending = get_category_spending(conn, uid, start_date, end_date)
        monthly_spending = get_monthly_spending(conn, uid, start_date, end_date)
        vendor_spending = get_top_vendors(conn, uid, start_date, end_date, limit=10)
        summary = get_summary_stats(conn, uid, start_date, end_date)
    finally:
        conn.close()

    return {
        "summary": summary,
        "categorySpending": category_spending,
        "monthlySpending": monthly_spending,
        "vendorSpending": vendor_spending,
    }


# ------------- CLI debug helper ------------- #

if __name__ == "__main__":
    """
    Quick debug:
      python3 dashboard_data.py

    Prints JSON with all the dashboard data for DEFAULT_USER_ID.
    You can pipe this into a file and hand it to the frontend dev if needed.
    """
    data = get_dashboard_data()
    print(json.dumps(data, indent=2, default=str))
