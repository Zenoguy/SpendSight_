"""
Lightweight keyword-based heuristic classifier.
This is Stage-2 of the pipeline:
    Regex (strict) -> Heuristics -> MiniLM -> LLM

Upgrades made:
 - expanded keyword lists and improved vendor matching
 - better handling for UPI merchant payments vs P2P transfers
 - explicit detection for EMI/loan repayments (EMIPAYMENT, HDFC-LOAN, etc.)
 - stronger POS/card parsing for fuel, groceries, dining
 - improved meta with matched_rule, matched_token, and context snippet
 - small numeric parsing robustness

Returns (category, subcategory, confidence, meta) like before
"""

import re
from typing import Tuple, Optional, Dict

# small helper
def normalize(desc: str) -> str:
    return (desc or "").strip().lower()

# precompile common regexes
UPI_RE = re.compile(r'\bupi\b|upi/|ipay|ipay|ip[a,y]y|paytm|phonepe|googlepay|gpay|bhim|sbipay|sbiepay|eship|ipayment', re.I)
CARD_RE = re.compile(r'\b(debit card|credit card|visa-pos|visa-ref|mastercard|pos|atm|debitcard|visapos|visa pos|master card)\b', re.I)
BANK_TRANSFER_RE = re.compile(r'\b(imps|neft|rtgs|fund transfer|transfer from|transfer to|mmid|imps/|neft/|rtgs/)\b', re.I)
TXN_REF_RE = re.compile(r'\b(ref|txn|trn|utr|id0|id064|bn\d{3,}|txnid|txn id)\b', re.I)
VENDOR_NUMBER_RE = re.compile(r'(\b\d{6,}\b)')  # long numeric tokens often part of refs
AMOUNT_RE = re.compile(r'[\d,]+\.?\d*')  # crude amount detector - more permissive

# vendor groups (for nicer meta)
WALLETS = ["paytm", "phonepe", "gpay", "googlepay", "mobikwik", "freecharge", "bharatpe"]
FOOD = ["zomato", "swiggy", "dunzo", "foodpanda", "dominos", "pizza hut", "pizza", "bigbasket" ]
TRANSPORT = ["uber", "ola", "rapido", "redbus", "ola cabs", "uber india"]
MARKETPLACES = ["amazon", "flipkart", "myntra", "ajio", "tatacliq", "amazon.in"]
GROCERY = ["bigbasket", "dmart", "more", "spencer", "reliance fresh", "natures basket", "nature's basket", "kirana"]
FUEL = ["bpcl", "indian oil", "ioctl", "hpcl", "bharat petroleum", "petrol", "fuel", "petrol pump", "petrolcircle"]
TRAVEL = ["indigo", "air india", "goair", "spicejet", "make my trip", "cleartrip", "ibibo", "booking.com", "makemytrip"]
UTILITIES = ["broadband", "bsnl", "reliance jio", "jio", "vodafone", "airtel", "electricity", "bescom", "bses", "mahavitaran"]
SUBSCRIPTIONS = ["netflix", "spotify", "prime video", "hotstar", "zee5", "youtube premium", "apple", "spotify" ]
RECURRING_FEES = ["emi", "loan emi", "equated monthly", "monthly instalment", "emipayment", "emipayment-", "emipay"]
BANK_FEES = ["bank charges", "cheque bounce", "service charge", "monthly maintenance", "mms_charge", "sms_charge", "annual_card_fee"]

# helper to find the first token in a description
def _first_match_token(d: str, tokens):
    for t in tokens:
        if t in d:
            return t
    return None


def classify_with_heuristics(description: str) -> Tuple[str, Optional[str], float, Dict]:
    """
    Returns (category, subcategory, confidence, meta)
    meta includes matched_rule, matched_token and a short context snippet
    """
    if not description or not description.strip():
        return "PENDING", None, 0.0, {"reason": "empty"}

    raw = description
    d = normalize(description)
    meta: Dict = {"matched_rule": None, "matched_token": None, "context": d[:240]}

    # --- High confidence exact / strong signals ---
    # Income / Salary
    if any(k in d for k in ["salary", "credited", "salary credit", "payroll", "salary by", "salary from"]):
        meta.update({"matched_rule": "income_exact"})
        return "Income", "Salary", 0.98, meta

    # EMI / Loan payments (explicit patterns like EMIPAYMENT-HDFC, HDFC-LOAN)
    if any(k in d for k in RECURRING_FEES) or any(k in d for k in ["hdfc-loan", "emipayment", "emipay", "loan payment", "loan emi"]):
        token = _first_match_token(d, RECURRING_FEES)
        meta.update({"matched_rule": "emi_exact", "matched_token": token})
        return "Bills", "EMI", 0.90, meta

    # Bank transfer / NEFT/IMPS (common label)
    bank_search = BANK_TRANSFER_RE.search(d)
    if bank_search:
        meta.update({"matched_rule": "bank_transfer", "matched_token": bank_search.group(0)})
        return "Transfers", "BankTransfer", 0.88, meta

    # UPI / Wallets — more nuanced: try to detect merchant UPI vs P2P
    upi_search = UPI_RE.search(d)
    if upi_search or any(w in d for w in WALLETS):
        token = (upi_search.group(0) if upi_search else _first_match_token(d, WALLETS))
        meta.update({"matched_rule": "upi_wallet", "matched_token": token})
        # if merchant keyword appears alongside UPI, treat as merchant payment
        merchant_token = _first_match_token(d, FOOD + TRANSPORT + GROCERY + MARKETPLACES + FUEL)
        if merchant_token:
            # map specific merchants
            if merchant_token in FOOD:
                return "Dining", "FoodDelivery", 0.80, meta
            if merchant_token in TRANSPORT:
                return "Transport", "Cab", 0.78, meta
            if merchant_token in GROCERY:
                return "Groceries", "Shopping", 0.78, meta
            if merchant_token in FUEL:
                return "Transport", "Fuel", 0.82, meta
            if merchant_token in MARKETPLACES:
                return "Shopping", "Online", 0.78, meta
        # refund / reversal signals
        if any(k in d for k in ["refund", "reversal", "credited"]):
            return "Transfers", "Refund", 0.80, meta
        # default to UPI transfer
        return "Transfers", "UPI", 0.75, meta

    # Card / POS transactions (merchant spend)
    card_search = CARD_RE.search(d)
    if card_search or any(k in d for k in ["visa-pos", "visa ref", "debit card", "credit card", "vpa"]):
        token = card_search.group(0) if card_search else _first_match_token(d, MARKETPLACES + GROCERY + FUEL + FOOD + TRANSPORT)
        meta.update({"matched_rule": "card_pos", "matched_token": token})
        # try to detect merchant verticals inside card desc
        if _first_match_token(d, FOOD):
            return "Dining", "Food", 0.78, meta
        if _first_match_token(d, FUEL) or any(k in d for k in ["petrol", "fuel", "petrol pump", "petrolcircle"]):
            return "Transport", "Fuel", 0.80, meta
        if _first_match_token(d, GROCERY):
            return "Groceries", "Supermarket", 0.78, meta
        # if card but merchant mentions 'service station' prefer fuel
        if "service station" in d or "petrol" in d:
            return "Transport", "Fuel", 0.80, meta
        return "Shopping", "POS", 0.70, meta

    # Food / Dining / Delivery (non-card UPI or plain text)
    if _first_match_token(d, FOOD) or any(k in d for k in ["hotel", "resto", "restaurant", "meal", "dine"]):
        token = _first_match_token(d, FOOD)
        meta.update({"matched_rule": "food_keywords", "matched_token": token})
        return "Dining", "FoodDelivery", 0.80 if token in ["zomato", "swiggy"] else 0.66, meta

    # Transport (cab/ride/public)
    if _first_match_token(d, TRANSPORT) or any(k in d for k in ["cab", "taxi", "auto", "ride", "uber", "ola"]):
        token = _first_match_token(d, TRANSPORT)
        meta.update({"matched_rule": "transport", "matched_token": token})
        return "Transport", "Cab", 0.75, meta

    # Marketplaces / e-commerce
    if _first_match_token(d, MARKETPLACES) or any(k in d for k in ["online shopping", "order", "seller", "marketplace"]):
        token = _first_match_token(d, MARKETPLACES)
        meta.update({"matched_rule": "marketplace", "matched_token": token})
        return "Shopping", "Online", 0.78, meta

    # Groceries / Supermarket heuristics
    if _first_match_token(d, GROCERY) or any(k in d for k in ["supermarket", "kirana", "grocery", "store"]):
        token = _first_match_token(d, GROCERY)
        meta.update({"matched_rule": "grocery", "matched_token": token})
        return "Groceries", "Shopping", 0.78, meta

    # Fuel / Petrol stations fallback
    if _first_match_token(d, FUEL) or any(k in d for k in ["petrol", "service station", "fuel", "petrol pump"]):
        token = _first_match_token(d, FUEL)
        meta.update({"matched_rule": "fuel", "matched_token": token})
        return "Transport", "Fuel", 0.82, meta

    # Travel / booking / airline / hotels
    if _first_match_token(d, TRAVEL) or any(k in d for k in ["booking", "irctc", "railway", "train ticket", "flight", "hotel booking", "airtel bus"]):
        meta.update({"matched_rule": "travel", "matched_token": _first_match_token(d, TRAVEL)})
        return "Travel", "TravelBooking", 0.80, meta

    # Utilities and telecom
    if _first_match_token(d, UTILITIES) or any(k in d for k in ["electricity", "water bill", "broadband", "gas bill", "mobile recharge", "recharge"]):
        meta.update({"matched_rule": "utilities", "matched_token": _first_match_token(d, UTILITIES)})
        return "Bills", "Utilities", 0.85, meta

    # Subscriptions (streaming etc)
    if _first_match_token(d, SUBSCRIPTIONS) or "subscription" in d or "monthly plan" in d:
        meta.update({"matched_rule": "subscription", "matched_token": _first_match_token(d, SUBSCRIPTIONS)})
        return "Entertainment", "Subscription", 0.88, meta

    # Bank charges / fees
    if any(k in d for k in BANK_FEES) or ("charge" in d and ("bank" in d or "service charge" in d)):
        meta.update({"matched_rule": "bank_fee"})
        return "Bills", "BankCharges", 0.88, meta

    # ATM / Cash withdrawal
    if "atm" in d or "atm wdl" in d or "cash wdl" in d or "cash withdrawal" in d:
        meta.update({"matched_rule": "atm"})
        return "Cash", "ATMWithdrawal", 0.86, meta

    # Refunds and reversals
    if any(k in d for k in ["refund", "reversal", "reversed", "credited back", "reversal txn"]):
        meta.update({"matched_rule": "refund"})
        return "Transfers", "Refund", 0.80, meta

    # Taxes / Govt payments
    if any(k in d for k in ["govt", "income tax", "tds", "tax", "gst", "cbic", "tax paid", "gstin"]):
        meta.update({"matched_rule": "taxes"})
        return "Bills", "Taxes", 0.92, meta

    # Healthcare / Pharmacy / Hospital
    if any(k in d for k in ["hospital", "clinic", "pharmacy", "medic", "dr.", "doctor", "chemist", "wellness"]):
        meta.update({"matched_rule": "health"})
        return "Health", "Medical", 0.80, meta

    # Education
    if any(k in d for k in ["school", "college", "tuition", "university", "exam", "entrance", "course fee"]):
        meta.update({"matched_rule": "education"})
        return "Education", "Tuition", 0.84, meta

    # Charity / Donations
    if any(k in d for k in ["donation", "charity", "donated", "ngo"]):
        meta.update({"matched_rule": "donation"})
        return "Giving", "Donation", 0.82, meta

    # Parking / Toll / Metro / Cab extras
    if any(k in d for k in ["parking", "toll", "metro", "parking fee"]):
        meta.update({"matched_rule": "parking_toll"})
        return "Transport", "TollParking", 0.78, meta

    # Small heuristics for merchant id patterns and references
    txn_search = TXN_REF_RE.search(d)
    if txn_search:
        ref = txn_search.group(0)
        meta.update({"matched_rule": "txn_ref", "matched_token": ref})
        # If a line has UPI and a txn ref it's likely a transfer/UPI payment
        if UPI_RE.search(d):
            return "Transfers", "UPI", 0.72, meta
        # If description contains typical merchant keywords but no better match, label as Shopping.POS
        if any(k in d for k in ["shop", "store", "merchant", "m/s", "m/s."]):
            return "Shopping", "POS", 0.65, meta

    # Numeric heuristics: small amounts under 200 often expenses like food/transport
    try:
        amounts = [float(a.replace(",", "")) for a in AMOUNT_RE.findall(raw) if re.search(r"\d", a)]
        if amounts:
            smallest = min(amounts)
            meta.update({"matched_rule": "amount_heuristic", "sample_amounts": amounts})
            if smallest < 200:
                if any(k in d for k in ["hotel", "restaurant", "canteen", "dhaba", "chai", "coffee"]):
                    return "Dining", "Food", 0.64, meta
                if _first_match_token(d, TRANSPORT):
                    return "Transport", "Cab", 0.64, meta
                return "Shopping", "Misc", 0.55, meta
    except Exception:
        # keep going; don't fail classification because amount parsing failed
        pass

    # Last-resort: attempt to salvage via vendor tokens (phonepe, paytm etc.)
    v = _first_match_token(d, WALLETS + FOOD + TRANSPORT + MARKETPLACES + GROCERY + FUEL + UTILITIES + SUBSCRIPTIONS)
    if v:
        meta.update({"matched_rule": "vendor_fallback", "matched_token": v})
        if v in WALLETS:
            return "Transfers", "UPI", 0.68, meta
        if v in FOOD:
            return "Dining", "FoodDelivery", 0.70, meta
        if v in TRANSPORT:
            return "Transport", "Cab", 0.70, meta
        if v in MARKETPLACES:
            return "Shopping", "Online", 0.72, meta
        if v in GROCERY:
            return "Groceries", "Supermarket", 0.72, meta
        if v in FUEL:
            return "Transport", "Fuel", 0.78, meta
        if v in UTILITIES:
            return "Bills", "Utilities", 0.78, meta

    # No match: return PENDING with a reason that can be used later by MiniLM/LLM
    meta.update({"matched_rule": "no_heuristic_match"})
    return "PENDING", None, 0.0, meta
