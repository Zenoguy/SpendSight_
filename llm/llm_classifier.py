# llm/llm_classifier.py
"""
LLM Fallback classifier using Google Gemini (google-genai).

Features:
- Uses Gemini for semantic classification when available
- Falls back to lightweight rule-based classifier on errors/quota issues
- In-memory caching so the same description is not classified twice
- Batch classification for efficiency (classify_batch)
- Fully compatible with:
    category, subcategory, confidence, meta = llm_clf.classify(desc)
"""

import os
import json
from typing import List, Tuple, Optional, Dict

from dotenv import load_dotenv

load_dotenv()

# Try to import the genai SDK (google-genai)
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False


# -------------------------
# Global taxonomy prompt
# -------------------------
SYSTEM_PROMPT = """
You are an AI system that classifies bank transactions into a FIXED taxonomy.

RULES:
- ALWAYS choose exactly one category and one subcategory.
- Only choose from the taxonomy shown.
- If unclear, use category="Other" and subcategory="Misc".
- Respond ONLY with JSON. No prose. No explanations outside JSON.

TAXONOMY (category → subcategory list):

Dining: ["FoodDelivery", "Restaurant", "Cafe"]
Groceries: ["OnlineGroceries", "Supermarket"]
Shopping: ["Online", "Electronics", "Fashion", "Beauty"]
Entertainment: ["Streaming", "Music", "Gaming"]
Transport: ["Cab", "BikeTaxi", "PublicTransport", "Fuel"]
Utilities: ["Electricity", "Gas", "Internet", "MobileRecharge"]
Income: ["Salary", "Interest", "Refund"]
Debt: ["LoanEMI", "CreditCardPayment"]
BankCharges: ["BalanceCharge", "SMS", "LateFee", "Other"]
Cash: ["ATMWithdrawal", "ATMDeposit"]
Insurance: ["GovtScheme", "Life", "Health"]
Leisure: ["Gaming", "Subscriptions", "Other"]
Transfers: ["SelfAccount", "ToPerson", "ToBusiness"]
Other: ["Misc"]

OUTPUT SCHEMA (for each transaction):
{
  "category": "<category>",
  "subcategory": "<subcategory>",
  "confidence": <float between 0 and 1>,
  "rationale": "<short explanation>"
}
""".strip()


def _normalize_desc(desc: str) -> str:
    """Simple normalization used for caching keys."""
    return (desc or "").strip().upper()


class LLMFallbackClassifier:
    """
    Gemini-based classifier with:
      - single classify()
      - classify_batch()
      - in-memory caching
      - robust fallback on quota / API errors
    """

    def __init__(self, model_name: Optional[str] = None):
        # Model + API key
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL",
            "gemini-2.0-flash-lite-preview"  # safer default for free tier
        )
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.temperature = 0.2

        # In-memory cache: norm_desc -> (cat, subcat, conf, meta)
        self.cache: Dict[str, Tuple[str, Optional[str], float, Dict]] = {}

        # Initialize client if possible
        self.client = None
        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("[LLM] Gemini client initialized.")
            except Exception as e:
                print(f"[LLM WARN] Failed to initialize Gemini client: {e}")
                self.client = None
        else:
            if not GENAI_AVAILABLE:
                print("[LLM WARN] google-genai SDK not available; LLM will use rule-based fallback.")
            elif not self.api_key:
                print("[LLM WARN] GEMINI_API_KEY missing; LLM will use rule-based fallback.")

    # -------------------------
    # Rule-based fallback
    # -------------------------
    def _rule_based_fallback(
        self, desc: str
    ) -> Tuple[str, Optional[str], float, Dict]:
        """
        Deterministic lightweight fallback when LLM is unavailable or fails.
        """
        d = desc.lower()
        meta = {"method": "rule-fallback"}

        if any(k in d for k in ["uber", "ola", "ride", "cab", "rapido"]):
            return "Transport", "Cab", 0.60, meta
        if any(k in d for k in ["zomato", "swiggy", "restaurant", "resto", "hotel"]):
            return "Dining", "FoodDelivery", 0.65, meta
        if any(k in d for k in ["flipkart", "amazon", "myntra", "ajio"]):
            return "Shopping", "Online", 0.65, meta
        if any(k in d for k in ["bigbasket", "dmart", "grocer", "supermarket"]):
            return "Groceries", "Supermarket", 0.65, meta
        if any(k in d for k in ["netflix", "spotify", "youtube", "prime video"]):
            return "Entertainment", "Streaming", 0.70, meta
        if any(k in d for k in ["salary", "payroll", "credited by employer"]):
            return "Income", "Salary", 0.95, meta
        if any(k in d for k in ["imps", "neft", "rtgs", "fund transfer", "to upi id"]):
            return "Transfers", "ToPerson", 0.55, meta
        if any(k in d for k in ["atm wdl", "atm wdr", "atm withdrawal"]):
            return "Cash", "ATMWithdrawal", 0.70, meta

        # default unclear
        return "Other", "Misc", 0.40, meta

    # -------------------------
    # Internal: call Gemini for a BATCH
    # -------------------------
    def _call_gemini_batch(
        self, descriptions: List[str]
    ) -> Optional[List[Dict]]:
        """
        Calls Gemini once for a batch of descriptions.
        Returns list of dicts with keys:
            index, category, subcategory, confidence, rationale
        or None on failure.
        """
        if not self.client:
            return None

        items = [
            {"index": i, "description": d}
            for i, d in enumerate(descriptions)
        ]

        full_prompt = (
            SYSTEM_PROMPT
            + "\n\nYou will receive a JSON array named 'items', where each element has:\n"
            + '  { "index": <int>, "description": "<text>" }.\n\n'
            + "For EACH item, produce an object with:\n"
            + '  { "index": <same int>, "category": "<category>", "subcategory": "<subcategory>", '
            + '"confidence": <0-1>, "rationale": "<short explanation>" }.\n\n'
            + "Return ONLY a JSON array (no extra keys) in the same order of indices.\n\n"
            + f"items = {json.dumps(items, ensure_ascii=False)}"
        )

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[full_prompt],  # Gemini wants list of strings / parts
                config={
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as e:
            # Quota / API failure → mark client off for remainder of process
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                print(f"[LLM WARN] Gemini quota exhausted: {err_str}")
                self.client = None
            else:
                print(f"[LLM WARN] Gemini call failed: {err_str}")
            return None

        raw_text = getattr(resp, "text", None) or str(resp)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            print("[LLM WARN] Could not parse JSON from Gemini output.")
            return None

        # Expecting a JSON array
        if not isinstance(data, list):
            print("[LLM WARN] Gemini output is not a list.")
            return None

        return data

    # -------------------------
    # Public: batch classify
    # -------------------------
    def classify_batch(
        self, descriptions: List[str]
    ) -> List[Tuple[str, Optional[str], float, Dict]]:
        """
        Batch classify a list of descriptions.
        Returns list of (category, subcategory, confidence, meta) with same order.
        """
        n = len(descriptions)
        if n == 0:
            return []

        # 1) Initialize result list with None
        results: List[Optional[Tuple[str, Optional[str], float, Dict]]] = [None] * n

        # 2) Check cache & collect which ones need LLM
        to_llm_indices: List[int] = []
        to_llm_descs: List[str] = []

        for i, desc in enumerate(descriptions):
            desc_str = (desc or "").strip()
            if not desc_str:
                results[i] = ("Other", "Misc", 0.0, {"reason": "empty"})
                continue

            key = _normalize_desc(desc_str)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                to_llm_indices.append(i)
                to_llm_descs.append(desc_str)

        # 3) If no LLM needed, just return cached + trivial results
        if not to_llm_descs or self.client is None:
            # For any still None (no client), use rule-based
            for idx in to_llm_indices:
                d = descriptions[idx]
                rb_cat, rb_sub, rb_conf, rb_meta = self._rule_based_fallback(d)
                results[idx] = (rb_cat, rb_sub, rb_conf, rb_meta)
            return [r for r in results]  # type: ignore

        # 4) Call Gemini ONCE for remaining items
        gemini_output = self._call_gemini_batch(to_llm_descs)

        if gemini_output is None:
            # LLM failed → fallback rule-based for all those
            for local_pos, global_idx in enumerate(to_llm_indices):
                d = to_llm_descs[local_pos]
                rb_cat, rb_sub, rb_conf, rb_meta = self._rule_based_fallback(d)
                key = _normalize_desc(d)
                self.cache[key] = (rb_cat, rb_sub, rb_conf, rb_meta)
                results[global_idx] = (rb_cat, rb_sub, rb_conf, rb_meta)
            return [r for r in results]  # type: ignore

        # 5) Map LLM results back using "index" field
        #    Note: index is local to to_llm_descs (0..len(to_llm_descs)-1)
        by_index: Dict[int, Dict] = {}
        for obj in gemini_output:
            if isinstance(obj, dict) and "index" in obj:
                try:
                    by_index[int(obj["index"])] = obj
                except Exception:
                    continue

        for local_idx, global_idx in enumerate(to_llm_indices):
            d = to_llm_descs[local_idx]
            key = _normalize_desc(d)

            if local_idx in by_index:
                o = by_index[local_idx]
                cat = o.get("category") or "Other"
                sub = o.get("subcategory") or "Misc"
                try:
                    conf = float(o.get("confidence", 0.7))
                except Exception:
                    conf = 0.7
                meta = {
                    "rationale": o.get("rationale", ""),
                    "model": self.model_name,
                    "source": "gemini",
                }
            else:
                # Missing index in LLM output → fallback
                cat, sub, conf, meta = self._rule_based_fallback(d)
                meta["reason"] = "missing_index_from_llm"

            self.cache[key] = (cat, sub, conf, meta)
            results[global_idx] = (cat, sub, conf, meta)

        return [r for r in results]  # type: ignore

    # -------------------------
    # Public: single classify
    # -------------------------
    def classify(
        self, description: str
    ) -> Tuple[str, Optional[str], float, Dict]:
        """
        Single-text entry point.
        Uses classify_batch internally so logic is shared.
        """
        res_list = self.classify_batch([description])
        return res_list[0]


# Global instance used in UnifiedPipeline
llm_clf = LLMFallbackClassifier()
