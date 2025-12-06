# nlp/miniLM_classifier.py

import numpy as np
from typing import Dict, List, Tuple
import re

from sentence_transformers import SentenceTransformer
from .taxonomy import CATEGORIES
from regex_engine.vendor_map import VENDOR_CATEGORY_MAP   # optional vendor bias


# ----------------------------------------------------------
# Text Normalizer (same logic as regex_normalizer)
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# L2 Normalize
# ----------------------------------------------------------
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / norms


# ----------------------------------------------------------
# Strong Non-ML Rules (BERT aware)
# If MiniLM sees these patterns, we override classification.
# ----------------------------------------------------------
SALARY_RE = re.compile(r"\bSALARY CREDIT\b|\bSALARYCREDIT\b", re.I)
ATM_WDR_RE = re.compile(r"\bATM WDR\b|\bATMWDR\b", re.I)
SMS_CHG_RE = re.compile(r"SMS CHRG|SMSCHRG|SMS CHARGES", re.I)

EMI_RE = re.compile(r"\bEMI PAYMENT\b", re.I)
ATM_DEP_RE = re.compile(r"\bATM DEP\b", re.I)
INT_PD_RE = re.compile(r"\bINT\.?PD\b|\bINTEREST\b", re.I)
QTR_CHG_RE = re.compile(r"QUARTERLY AVG BAL|QTRLY AVG BAL", re.I)
ELEC_RE = re.compile(r"ELECTRICITY BILL", re.I)
GAS_RE = re.compile(r"\bGAS BILL\b", re.I)
INS_RE = re.compile(r"PMSBY|PMJJBY", re.I)


# ----------------------------------------------------------
# MiniLM Classifier
# ----------------------------------------------------------
class MiniLMClassifier:
    """
    Wraps all-MiniLM-L6-v2 and applies:
      - Text normalization (UPI noise removal)
      - Strong rule overrides (salary, EMI, ATM, etc.)
      - Vendor-map category biasing
      - Prototype similarity scoring
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.labels: List[str] = []
        self.embs: np.ndarray = None
        self._build_index()
        print("[MiniLM] Model loaded and prototype index built.")

    # ----------------------------------------------------------
    def _build_index(self):
        """
        Builds the prototype embedding matrix from taxonomy.
        """
        phrases = []
        labels = []

        for cat_label, examples in CATEGORIES.items():
            for phrase in examples:
                labels.append(cat_label)
                phrases.append(phrase)

        emb = self.model.encode(phrases, convert_to_numpy=True, show_progress_bar=False)
        self.embs = _l2_normalize(emb)
        self.labels = labels

    # ----------------------------------------------------------
    def _apply_rule_overrides(self, text_norm: str):
        """
        Strong deterministic overrides BEFORE MiniLM logic.
        """
        # Order matters: salary > EMI > ATM > interest > fees > utilities > govt insurance
        if SALARY_RE.search(text_norm):
            return "Income", "Salary", 0.98, "salary_rule"

        if EMI_RE.search(text_norm):
            return "Debt", "LoanEMI", 0.95, "emi_rule"

        if ATM_WDR_RE.search(text_norm):
            return "Cash", "ATMWithdrawal", 0.85, "atm_wdr_rule"

        if ATM_DEP_RE.search(text_norm):
            return "Cash", "ATMDeposit", 0.85, "atm_dep_rule"

        if INT_PD_RE.search(text_norm):
            return "Income", "Interest", 0.92, "interest_rule"

        if QTR_CHG_RE.search(text_norm):
            return "BankCharges", "BalanceCharge", 0.80, "qtr_charge_rule"

        if SMS_CHG_RE.search(text_norm):
            return "BankCharges", "SMS", 0.80, "sms_charge_rule"

        if ELEC_RE.search(text_norm):
            return "Utilities", "Electricity", 0.90, "electricity_rule"

        if GAS_RE.search(text_norm):
            return "Utilities", "Gas", 0.88, "gas_rule"

        if INS_RE.search(text_norm):
            return "Insurance", "GovtScheme", 0.90, "govt_insurance_rule"

        return None

    # ----------------------------------------------------------
    def classify(self, text: str) -> Tuple[str, float, Dict]:
        """
        Final classification pipeline:
          1) Normalize text
          2) Apply hard-coded rule overrides
          3) Vendor-map biasing
          4) Prototype similarity scoring
        """
        if not text or not text.strip():
            return "PENDING", 0.0, {"reason": "empty_text"}

        # Normalize text (same as regex)
        text_norm = normalize_desc(text)

        # ------------------------------------------------------
        # 1) RULE OVERRIDES
        # ------------------------------------------------------
        rule = self._apply_rule_overrides(text_norm)
        if rule:
            cat, subcat, conf, rule_name = rule
            return f"{cat}.{subcat}", conf, {
                "source": "rule_override",
                "rule": rule_name,
                "normalized_text": text_norm
            }

        # ------------------------------------------------------
        # 2) Vendor-map biasing
        # ------------------------------------------------------
        vendor_hit = None
        for key in VENDOR_CATEGORY_MAP:
            if key in text_norm:
                vendor_hit = key
                break

        # Example: "ZOMATO" → ("Dining", "FoodDelivery")
        vendor_bias = None
        if vendor_hit:
            vendor_bias = VENDOR_CATEGORY_MAP[vendor_hit]

        # ------------------------------------------------------
        # 3) MiniLM Semantic Matching (prototype-based)
        # ------------------------------------------------------
        q_emb = self.model.encode([text_norm], convert_to_numpy=True, show_progress_bar=False)
        q_emb = _l2_normalize(q_emb)

        sims = np.dot(q_emb, self.embs.T)[0]
        top_idx = sims.argsort()[::-1][:5]
        top = [(self.labels[i], float(sims[i])) for i in top_idx]

        best_label, best_sim = top[0]
        second_sim = top[1][1] if len(top) > 1 else 0.0

        # ------------------------------------------------------
        # Confidence function tuned for MiniLM behaviour
        # Many statements produce similarity range ~0.3–0.55.
        # ------------------------------------------------------
        raw_conf = best_sim

        # Confidence curve: starts at 0.45 → grows until 0.8
        if raw_conf < 0.45:
            confidence = 0.0
        elif raw_conf >= 0.80:
            confidence = 1.0
        else:
            confidence = (raw_conf - 0.45) / 0.35
            confidence = round(max(0.0, min(confidence, 1.0)), 3)

        # ------------------------------------------------------
        # Vendor bias strengthening (if present)
        # ------------------------------------------------------
        if vendor_bias:
            vendor_cat, vendor_sub = vendor_bias

            if vendor_cat in best_label:
                confidence = max(confidence, 0.70)

        # ------------------------------------------------------
        # Threshold: decide if MiniLM is confident or if row → LLM
        # ------------------------------------------------------
        if raw_conf < 0.50:
            return "PENDING", confidence, {
                "source": "bert",
                "reason": "low_similarity",
                "raw_conf": raw_conf,
                "top_5": top,
                "normalized_text": text_norm,
                "vendor_bias": vendor_bias,
            }

        # ------------------------------------------------------
        # Otherwise accept the MiniLM label
        # ------------------------------------------------------
        return best_label, confidence, {
            "source": "bert",
            "raw_conf": raw_conf,
            "margin": raw_conf - second_sim,
            "top_5": top,
            "normalized_text": text_norm,
            "vendor_bias": vendor_bias,
        }
