# regex_engine/upi_utils.py

import re
from typing import Optional, Tuple, Dict

# ----------------------------------------------------
# UPI patterns
# ----------------------------------------------------

# Detect UPI-style transactions
UPI_HINT_RE = re.compile(r"\bUPI[/\-]|/UPI/|VPA\b", re.IGNORECASE)

# Extract VPA handles like something@ybl, zomato@upi, hpcl001@okhdfcbank
VPA_RE = re.compile(r"([A-Za-z0-9.\-]+)@([A-Za-z0-9]+)", re.IGNORECASE)


# ----------------------------------------------------
# UPI merchant → category map (India-focused)
# ----------------------------------------------------
# Keys are UPPERCASE substrings we look for in the handle prefix.
# Values are (category, subcategory)
UPI_MERCHANT_MAP: Dict[str, Tuple[str, str]] = {
    # Food delivery / dining
    "ZOMATO": ("Dining", "FoodDelivery"),
    "SWIGGY": ("Dining", "FoodDelivery"),
    "DUNZO": ("Dining", "FoodDelivery"),
    "EATFIT": ("Dining", "FoodDelivery"),
    "BOX8": ("Dining", "FoodDelivery"),
    "FAASOS": ("Dining", "FoodDelivery"),
    "KFC": ("Dining", "Restaurant"),
    "MCDONALD": ("Dining", "Restaurant"),
    "DOMINOS": ("Dining", "Restaurant"),
    "PIZZA HUT": ("Dining", "Restaurant"),
    "PIZZAHUT": ("Dining", "Restaurant"),
    "CCD": ("Dining", "Cafe"),
    "CAFECOFFEEDAY": ("Dining", "Cafe"),

    # Groceries / supermarkets / quick commerce
    "DMART": ("Groceries", "Supermarket"),
    "BIGBASKET": ("Groceries", "OnlineGroceries"),
    "BBDAILY": ("Groceries", "OnlineGroceries"),
    "BB-": ("Groceries", "OnlineGroceries"),
    "JIOMART": ("Groceries", "Supermarket"),
    "JIO MART": ("Groceries", "Supermarket"),
    "ZEPT": ("Groceries", "OnlineGroceries"),   # ZEPTO*
    "BLINKIT": ("Groceries", "OnlineGroceries"),
    "NATURESBASKET": ("Groceries", "Supermarket"),
    "SPENCERS": ("Groceries", "Supermarket"),
    "MOREMEGASTORE": ("Groceries", "Supermarket"),
    "STARQUICK": ("Groceries", "Supermarket"),

    # Online shopping / fashion / electronics
    "AMAZON": ("Shopping", "Online"),
    "FLIPKART": ("Shopping", "Online"),
    "AJIO": ("Shopping", "Fashion"),
    "MYNTRA": ("Shopping", "Fashion"),
    "TATACLIQ": ("Shopping", "Online"),
    "TATANEU": ("Shopping", "Online"),
    "NYKAA": ("Shopping", "Beauty"),
    "MEESHO": ("Shopping", "Online"),
    "SNAPDEAL": ("Shopping", "Online"),
    "SHOPPERSSTOP": ("Shopping", "Fashion"),
    "LIFESTYLE": ("Shopping", "Fashion"),
    "MAXFASHION": ("Shopping", "Fashion"),
    "CROMA": ("Shopping", "Electronics"),
    "VIJAYSALES": ("Shopping", "Electronics"),
    "RELIANCE DIGITAL": ("Shopping", "Electronics"),
    "REL DIGITAL": ("Shopping", "Electronics"),
    "RELIANCE TRENDS": ("Shopping", "Fashion"),
    "PANTALOONS": ("Shopping", "Fashion"),

    # Cabs / mobility / transport
    "UBER": ("Transport", "Cab"),
    "OLA": ("Transport", "Cab"),
    "RAPIDO": ("Transport", "BikeTaxi"),
    "MERU": ("Transport", "Cab"),
    "REDBUS": ("Transport", "PublicTransport"),

    # Fuel / petrol pumps
    "HPCL": ("Transport", "Fuel"),
    "BPCL": ("Transport", "Fuel"),
    "IOCL": ("Transport", "Fuel"),
    "INDIANOIL": ("Transport", "Fuel"),
    "BHARATPET": ("Transport", "Fuel"),
    "HINDPETRO": ("Transport", "Fuel"),
    "RELIANCE PETROL": ("Transport", "Fuel"),

    # Telecom / mobile recharge / DTH
    "AIRTEL": ("Utilities", "MobileRecharge"),
    "JIO": ("Utilities", "MobileRecharge"),
    "VODAFONE": ("Utilities", "MobileRecharge"),
    "VI-": ("Utilities", "MobileRecharge"),
    "VI ": ("Utilities", "MobileRecharge"),
    "BSNL": ("Utilities", "MobileRecharge"),
    "SUN DIRECT": ("Utilities", "MobileRecharge"),
    "TATASKY": ("Utilities", "MobileRecharge"),
    "D2H": ("Utilities", "MobileRecharge"),
    "DISHTV": ("Utilities", "MobileRecharge"),

    # Streaming / entertainment
    "NETFLIX": ("Entertainment", "Streaming"),
    "SPOTIFY": ("Entertainment", "Music"),
    "HOTSTAR": ("Entertainment", "Streaming"),
    "DISNEY": ("Entertainment", "Streaming"),
    "SONYLIV": ("Entertainment", "Streaming"),
    "ZEE5": ("Entertainment", "Streaming"),
    "PRIME VIDEO": ("Entertainment", "Streaming"),

    # Wallets / payment gateways
    "PAYTM": ("Transfers", "ToBusiness"),
    "PHONEPE": ("Transfers", "ToBusiness"),
    "GOOGLEPAY": ("Transfers", "ToBusiness"),
    "GPAY": ("Transfers", "ToBusiness"),
    "AMAZONPAY": ("Transfers", "ToBusiness"),
    "MOBIKWIK": ("Transfers", "ToBusiness"),
    "FREECHARGE": ("Transfers", "ToBusiness"),
    "RAZORPAY": ("Transfers", "ToBusiness"),
    "BILLDESK": ("Transfers", "ToBusiness"),
    "CASHFREE": ("Transfers", "ToBusiness"),

    # Travel / tickets / IRCTC etc.
    "IRCTC": ("Transport", "PublicTransport"),
    "MAKEMYTRIP": ("Travel", "Fuel"),      # or Travel.Other; tune later
    "EASEMYTRIP": ("Travel", "Fuel"),
    "YATRA": ("Travel", "Fuel"),

    # Gaming / fantasy
    "DREAM11": ("Leisure", "Gaming"),
    "MPL": ("Leisure", "Gaming"),
    "RUMMYCIRCLE": ("Leisure", "Gaming"),
    "GAMING": ("Leisure", "Gaming"),

    # Govt / utilities (rough buckets)
    "BESCOM": ("Utilities", "Electricity"),
    "BSES": ("Utilities", "Electricity"),
    "TNEB": ("Utilities", "Electricity"),
    "MSEDCL": ("Utilities", "Electricity"),
    "TORRENTPOWER": ("Utilities", "Electricity"),

    # Insurance
    "LIC": ("Insurance", "Life"),
    "HDFCLIFE": ("Insurance", "Life"),
    "SBILIFE": ("Insurance", "Life"),
    "ICICIPRULIFE": ("Insurance", "Life"),

    # Generic bank-ish / bill-ish
    "BBPS": ("Utilities", "Other"),
    "ELECTRIC": ("Utilities", "Electricity"),
    "GAS": ("Utilities", "Gas"),
}


# Pre-sort keys to match more specific ones first
UPI_KEYS = sorted(UPI_MERCHANT_MAP.keys(), key=len, reverse=True)


def _looks_like_person(prefix: str) -> bool:
    """
    Very rough heuristic: treat handle prefix as a person if:
    - contains no digits
    - and doesn't contain obvious merchant keywords
    """
    p = prefix.upper()
    if any(k in p for k in UPI_KEYS):
        return False
    return not any(ch.isdigit() for ch in p)


def classify_upi(text_norm: str) -> Tuple[Optional[str], Optional[str], Optional[str], float, Dict]:
    """
    Try to classify UPI-style transactions from normalized description.

    Args:
        text_norm: uppercased, cleaned description (e.g. from normalize_desc())

    Returns:
        (category, subcategory, vendor, confidence, meta)
        If not a UPI transaction, returns (None, None, None, 0.0, {}).
    """
    # Quick check: if there's no UPI hint at all, bail out
    if not UPI_HINT_RE.search(text_norm):
        return None, None, None, 0.0, {}

    # Find a VPA handle in the text
    m = VPA_RE.search(text_norm)
    if not m:
        return None, None, None, 0.0, {}

    handle_prefix = m.group(1).upper()       # e.g. "ZOMATO", "DMART123", "RAMESH-KUMAR"
    handle_domain = m.group(2).upper()       # e.g. "YBL", "OKHDFCBANK"
    full_handle = f"{handle_prefix}@{handle_domain}"
    meta: Dict = {
        "matched_rule": "upi",
        "handle": full_handle,
    }

    # 1) Try merchant keyword map
    for key in UPI_KEYS:
        if key in handle_prefix:
            category, subcat = UPI_MERCHANT_MAP[key]
            meta["matched_merchant_key"] = key
            return category, subcat, handle_prefix.title(), 0.88, meta

    # 2) If looks like a person → treat as person transfer
    if _looks_like_person(handle_prefix):
        return "Transfers", "ToPerson", handle_prefix.title(), 0.75, {
            **meta,
            "reason": "upi_person_like",
        }

    # 3) Otherwise, generic business transfer
    return "Transfers", "ToBusiness", handle_prefix.title(), 0.70, {
        **meta,
        "reason": "upi_business_generic",
    }
