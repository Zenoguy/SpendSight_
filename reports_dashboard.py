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
    Build dashboard JSON via get_dashboard_data(...) and store it in `reports`.

    - period: label like '2023-06', '2023-Q3', '2023-04_to_2024-03'
    - If (user_id, period, 'dashboard_snapshot') exists, it is UPDATED.
    Returns the report_id (UUID as string).
    """
    dashboard_json = get_dashboard_data(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (
                    user_id,
                    period,
                    report_type,
                    summary_json,
                    insights,
                    file_path
                )
                VALUES (
                    %s,
                    %s,
                    'dashboard_snapshot',
                    %s::jsonb,
                    NULL,
                    NULL
                )
                ON CONFLICT (user_id, period, report_type)
                DO UPDATE SET
                    summary_json = EXCLUDED.summary_json,
                    updated_at   = NOW()
                RETURNING report_id;
                """,
                (user_id, period, json.dumps(dashboard_json)),
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
