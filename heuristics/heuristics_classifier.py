# heuristics/heuristics_classifier.py

"""
Lightweight keyword-based heuristic classifier.
This is Stage-2 of the pipeline:
    Regex (strict) → Heuristics → MiniLM → LLM
"""

from typing import Tuple, Optional, Dict


def normalize(desc: str) -> str:
    return (desc or "").strip().lower()


def classify_with_heuristics(description: str) -> Tuple[str, Optional[str], float, Dict]:
    """
    Loose, keyword-based rules.
    We only call this *after* regex fails.
    """
    if not description or not description.strip():
        return "PENDING", None, 0.0, {"reason": "empty"}

    d = normalize(description)
    meta = {"matched_rule": "heuristic"}

    # Transport
    if any(k in d for k in ["uber", "ola", "rapido", "ride", "cab"]):
        return "Transport", "Cab", 0.60, meta

    # Dining / Food delivery
    if any(k in d for k in ["zomato", "swiggy", "restaurant", "resto", "hotel"]):
        return "Dining", "FoodDelivery", 0.65, meta

    # Shopping / Online
    if any(k in d for k in ["flipkart", "amazon", "myntra", "ajio"]):
        return "Shopping", "Online", 0.65, meta

    # Groceries
    if any(k in d for k in ["bigbasket", "dmart", "grocer", "supermarket"]):
        return "Groceries", "Supermarket", 0.65, meta

    # Entertainment / Streaming
    if any(k in d for k in ["netflix", "spotify", "youtube", "prime video"]):
        return "Entertainment", "Streaming", 0.70, meta

    # Income
    if any(k in d for k in ["salary", "payroll", "credited by employer"]):
        return "Income", "Salary", 0.95, meta

    # Transfers
    if any(k in d for k in ["imps", "neft", "rtgs", "fund transfer", "to upi id"]):
        return "Transfers", "ToPerson", 0.55, meta

    # ATM Withdrawal
    if any(k in d for k in ["atm wdl", "atm wdr", "atm withdrawal"]):
        return "Cash", "ATMWithdrawal", 0.70, meta

    # DEFAULT
    return "PENDING", None, 0.0, {"reason": "no_heuristic_match"}
