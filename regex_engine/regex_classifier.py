# regex_engine/regex_classifier.py

import re
from regex_engine.category_rules import CATEGORY_REGEX
from regex_engine.vendor_rules import VENDOR_REGEX
from regex_engine.vendor_map import VENDOR_CATEGORY_MAP  # <- you said you added this


# ----------------------------------------------------
# Description normalizer (already added by you)
# ----------------------------------------------------
def normalize_desc(desc: str) -> str:
    if not desc:
        return ""

    d = desc.upper()

    # Normalize whitespace
    d = re.sub(r"\s+", " ", d)

    # Fix common "glued" tokens
    glue_fixes = {
        "SALARYCREDIT": "SALARY CREDIT",
        "SMSCHRG": "SMS CHRG",
        "ATMWDR": "ATM WDR",
        "INTERNETBANGALORE": "INTERNET BANGALORE",
        "MOBILERECHARGE": "MOBILE RECHARGE",
        "LATEFINEFEES": "LATE FINE FEES",
        "VLPCHARGES": "VLP CHARGES",
        "TRANSFERTOANIL": "TRANSFER TO ANIL",  # if you want cleaner text for LLM/MiniLM
    }
    for bad, good in glue_fixes.items():
        d = d.replace(bad, good)

    # Collapse UPI noise
    d = re.sub(r"UPI/DR/[\w\d/.\-]+/", "UPI/", d)
    d = re.sub(r"UPI/\d+/", "UPI/", d)

    # Strip long numeric IDs
    d = re.sub(r"\d{10,}", " ", d)

    # Collapse double spaces
    d = re.sub(r"\s{2,}", " ", d).strip()

    return d

# ----------------------------------------------------
# Precomputed vendor keys (for fast substring search)
# ----------------------------------------------------
VENDOR_KEYS = sorted(VENDOR_CATEGORY_MAP.keys(), key=len, reverse=True)


def extract_vendor_from_map(desc_norm: str) -> str | None:
    """
    Try to find a vendor key from VENDOR_CATEGORY_MAP
    inside the normalized description (uppercased).
    """
    for key in VENDOR_KEYS:
        if key in desc_norm:
            return key
    return None


# ----------------------------------------------------
# Extra non-vendor rules (salary, EMI, ATM, fees, etc.)
# ----------------------------------------------------
SALARY_RE = re.compile(r"\bSALARY CREDIT\b|\bSALARYCREDIT\b", re.I)
ATM_WDR_RE = re.compile(r"\bATM WDR\b|\bATMWDR\b", re.I)
SMS_CHG_RE = re.compile(r"\bSMS CHRG\b|\bSMSCHRG\b|\bSMS CHARGES\b", re.I)
EMI_RE = re.compile(r"\bEMI PAYMENT\b", re.I)
ATM_DEP_RE = re.compile(r"\bATM DEP\b", re.I)
INT_PD_RE = re.compile(r"\bINT\.PD\b|\bINT PD\b|\bINTEREST\b", re.I)
QTR_CHG_RE = re.compile(r"QUARTERLY AVG BAL CHARGE|QTRLY AVG BAL CHARGE", re.I)
ELEC_RE = re.compile(r"\bELECTRICITY BILL\b", re.I)
GAS_RE = re.compile(r"\bGAS BILL\b", re.I)
INS_RE = re.compile(r"\bPMSBY\b|\bPMJJBY\b", re.I)
MOBILE_RECHARGE_RE = re.compile(r"\bMOBILE RECHARGE\b", re.I)
LATE_FEE_RE = re.compile(r"\bLATE FINE FEES\b|\bLATE FEE\b", re.I)
VLP_CHARGES_RE = re.compile(r"\bVLP CHARGES\b", re.I)


def classify_with_regex(description: str):
    """
    Main regex classifier.

    Returns:
        category: str or "PENDING"
        subcategory: str | None
        vendor: str | None
        confidence: float
        meta: dict
    """
    if not description:
        return "PENDING", None, None, 0.0, {"reason": "empty"}

    raw_text = description.strip()
    text_lower = raw_text.lower()
    text_norm = normalize_desc(raw_text)  # UPPERCASED + cleaned
    meta: dict = {}

    category: str | None = None
    subcategory: str | None = None
    vendor: str | None = None
    confidence: float = 0.0

    # ------------------------------------------------
    # 1) Vendor-based classification via vendor_map
    # ------------------------------------------------
    vendor_key = extract_vendor_from_map(text_norm)
    if vendor_key:
        cat, subcat = VENDOR_CATEGORY_MAP[vendor_key]

        category = cat
        subcategory = subcat
        # Pretty vendor name
        vendor = vendor_key.title()
        confidence = 0.9
        meta["matched_rule"] = "vendor_map"
        meta["vendor_key"] = vendor_key

        # NOTE: we *still* allow salary/EMI/etc. below to override if needed.
        # For most merchants, this is enough.
        # Fall through intentionally.

    # ------------------------------------------------
    # 2) Non-vendor semantic rules (salary, EMI, ATM, etc.)
    # ------------------------------------------------
    # These can override vendor_map if they hit.

    if SALARY_RE.search(text_norm):
        category = "Income"
        subcategory = "Salary"
        vendor = vendor or "Employer"
        confidence = max(confidence, 0.95)
        meta["matched_rule"] = "salary"

    elif EMI_RE.search(text_norm) or "HDFC-LOAN" in text_norm or "HDFC L-" in text_norm:
        category = "Debt"
        subcategory = "LoanEMI"
        vendor = vendor or "LoanProvider"
        confidence = max(confidence, 0.9)
        meta["matched_rule"] = "emi"

    elif ATM_WDR_RE.search(text_norm):
        category = "Cash"
        subcategory = "ATMWithdrawal"
        vendor = vendor or "ATM"
        confidence = max(confidence, 0.85)
        meta["matched_rule"] = "atm_wdr"

    elif ATM_DEP_RE.search(text_norm):
        category = "Cash"
        subcategory = "ATMDeposit"
        vendor = vendor or "ATM"
        confidence = max(confidence, 0.85)
        meta["matched_rule"] = "atm_dep"

    elif INT_PD_RE.search(text_norm):
        category = "Income"
        subcategory = "Interest"
        vendor = vendor or "BankInterest"
        confidence = max(confidence, 0.9)
        meta["matched_rule"] = "interest"

    elif QTR_CHG_RE.search(text_norm):
        category = "BankCharges"
        subcategory = "BalanceCharge"
        vendor = vendor or "BankFee"
        confidence = max(confidence, 0.8)
        meta["matched_rule"] = "qtr_charge"

    elif SMS_CHG_RE.search(text_norm):
        category = "BankCharges"
        subcategory = "SMS"
        vendor = vendor or "BankFee"
        confidence = max(confidence, 0.8)
        meta["matched_rule"] = "sms_charge"

    elif ELEC_RE.search(text_norm):
        category = "Utilities"
        subcategory = "Electricity"
        vendor = vendor or "ElectricityBoard"
        confidence = max(confidence, 0.9)
        meta["matched_rule"] = "electricity"

    elif GAS_RE.search(text_norm):
        category = "Utilities"
        subcategory = "Gas"
        vendor = vendor or "GasProvider"
        confidence = max(confidence, 0.9)
        meta["matched_rule"] = "gas"

    elif INS_RE.search(text_norm):
        category = "Insurance"
        subcategory = "GovtScheme"
        vendor = vendor or "GovtInsurance"
        confidence = max(confidence, 0.9)
        meta["matched_rule"] = "govt_insurance"

    # ------------------------------------------------
    # 3) Fallback to your existing CATEGORY_REGEX
    # ------------------------------------------------
    if category is None:
        for cat_label, patterns in CATEGORY_REGEX.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    parts = cat_label.split(".")
                    category = parts[0]
                    subcategory = parts[1] if len(parts) > 1 else None
                    confidence = max(confidence, 0.8)
                    meta["matched_rule"] = "category_regex"
                    meta["category_hit"] = pat
                    break
            if category:
                break

    # ------------------------------------------------
    # 4) Fallback vendor detection using VENDOR_REGEX (your old rules)
    # ------------------------------------------------
    if vendor is None:
        for vendor_name, patterns in VENDOR_REGEX.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    vendor = vendor_name
                    meta["vendor_hit"] = pat
                    meta["matched_rule"] = meta.get("matched_rule", "vendor_regex")
                    # if no category yet, at least mark as something generic
                    if category is None:
                        category = "Uncategorized"
                    confidence = max(confidence, 0.7)
                    break
            if vendor:
                break

    # ------------------------------------------------
    # 5) Final fallback → PENDING for MiniLM / LLM
    # ------------------------------------------------
    if category is None:
        category = "PENDING"
        subcategory = None
        vendor = vendor  # maybe detected vendor only
        confidence = 0.0
        meta.setdefault("reason", "no_regex_match")

    return category, subcategory, vendor, confidence, meta
