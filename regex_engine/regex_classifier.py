# regex_engine/regex_classifier.py

import re
from regex_engine.category_rules import CATEGORY_REGEX
from regex_engine.vendor_rules import VENDOR_REGEX

def classify_with_regex(description: str):
    """
    Return:
        category, subcategory, vendor, confidence, meta
    """
    if not description:
        return None, None, None, 0.0, {"reason": "empty"}

    text = description.lower().strip()
    meta = {}

    # -----------------------------
    # 1. Vendor detection
    # -----------------------------
    vendor = None
    for vendor_name, patterns in VENDOR_REGEX.items():
        for pat in patterns:
            if re.search(pat, text):
                vendor = vendor_name
                meta["vendor_hit"] = pat
                break
        if vendor:
            break

    # -----------------------------
    # 2. Category detection
    # -----------------------------
    category = None
    subcategory = None

    for cat, patterns in CATEGORY_REGEX.items():
        for pat in patterns:
            if re.search(pat, text):
                parts = cat.split(".")
                category = parts[0]
                subcategory = parts[1] if len(parts) > 1 else None
                meta["category_hit"] = pat
                break
        if category:
            break

    # -----------------------------
    # 3. Confidence estimation
    # -----------------------------
    if category or vendor:
        confidence = 0.90
    else:
        confidence = 0.0

    return category, subcategory, vendor, confidence, meta
