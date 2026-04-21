import json
from datetime import date
from typing import Optional, Any, Dict

from PipeLine import get_db_connection
from dashboard_data import get_dashboard_data


# --------------------------------------------------
# 1) CREATE / UPDATE dashboard snapshot in `reports`
# --------------------------------------------------

def save_dashboard_snapshot(
    user_id: str,
    period: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> str:
    """
    Generate dashboard data for a user + period and upsert it into reports.summary_json.

    - summary_json: full dashboard_data payload
    - insights: left NULL here (can be filled later by LLM)
    - file_path: NULL (we are not storing PNGs anymore)
    """
    conn = get_db_connection()
    try:
        snapshot: Dict[str, Any] = get_dashboard_data(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (user_id, period, summary_json, insights, file_path)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (user_id, period) DO UPDATE
                SET summary_json = EXCLUDED.summary_json,
                    insights     = EXCLUDED.insights,
                    file_path    = EXCLUDED.file_path
                RETURNING report_id;
                """,
                (
                    user_id,
                    period,
                    json.dumps(snapshot),
                    None,   # insights placeholder
                    None,   # file_path placeholder
                ),
            )
            report_id = cur.fetchone()[0]

        conn.commit()
        return str(report_id)
    finally:
        conn.close()

# --------------------------------------------------
# 2) READ snapshots for the Dashboard tab
# --------------------------------------------------

def get_latest_dashboard_snapshot(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the most recent dashboard snapshot JSON for a user,
    or None if nothing exists.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary_json
                FROM reports
                WHERE user_id = %s
                  AND report_type = 'dashboard_snapshot'
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            # row[0] is already JSONB -> dict from psycopg2
            return row[0]
    finally:
        conn.close()


def get_dashboard_snapshot_for_period(
    user_id: str,
    period: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the dashboard snapshot JSON for a specific period, if it exists.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary_json
                FROM reports
                WHERE user_id = %s
                  AND period = %s
                  AND report_type = 'dashboard_snapshot'
                LIMIT 1;
                """,
                (user_id, period),
            )
            row = cur.fetchone()
            if not row:
                return None
            return row[0]
    finally:
        conn.close()


# --------------------------------------------------
# 3) Optional: convenience API for Person C
# --------------------------------------------------

def get_dashboard_data_from_snapshot_or_live(
    user_id: str,
    period: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    mode: str = "snapshot",  # "snapshot" or "live"
) -> Dict[str, Any]:
    """
    Frontend-friendly entrypoint:

    - mode='live'     -> always compute via get_dashboard_data()
    - mode='snapshot' -> try reports table first, fall back to live if missing
    """
    if mode == "live":
        return get_dashboard_data(user_id=user_id, start_date=start_date, end_date=end_date)

    # SNAPSHOT mode
    if period:
        snap = get_dashboard_snapshot_for_period(user_id, period)
        if snap:
            return snap
        # fallback to live if period not stored
        return get_dashboard_data(user_id=user_id, start_date=start_date, end_date=end_date)

    # no period: just use the latest snapshot if available
    snap = get_latest_dashboard_snapshot(user_id)
    if snap:
        return snap

    # fallback: compute from live data
    return get_dashboard_data(user_id=user_id, start_date=start_date, end_date=end_date)


# --------------------------------------------------
# 4) CLI debug
# --------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
    if not DEFAULT_USER_ID:
        raise RuntimeError("DEFAULT_USER_ID not set in environment")

    # Example period label – you can choose your own convention
    period_label = "lifetime_till_today"

    rid = save_dashboard_snapshot(
        user_id=DEFAULT_USER_ID,
        period=period_label,
        start_date=None,
        end_date=None,
    )
    print(f"[DEBUG] Saved dashboard snapshot report_id={rid}")

    snap = get_dashboard_snapshot_for_period(DEFAULT_USER_ID, period_label)
    print("\n[DEBUG] Loaded snapshot JSON keys:")
    if snap:
        print(snap.keys())
