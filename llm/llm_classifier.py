# llm/llm_classifier.py
"""
Pure LLM classifier using Google Gemini (google-genai).

- No rule-based fallback here (all heuristics live in regex stage).
- If the LLM is unavailable, errors, or returns invalid output:
    → we return ("Other", "Misc", 0.0, meta)
- Provides:
    - llm_clf.classify(description)
    - llm_clf.classify_batch([descriptions...])

Return format:
    (category: str, subcategory: Optional[str], confidence: float, meta: dict)
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
    """Simple normalization used for cache keys (if needed later)."""
    return (desc or "").strip().upper()


class LLMFallbackClassifier:
    """
    Gemini-based classifier with:
      - classify()
      - classify_batch()
    If LLM is unavailable / fails / returns bad JSON:
      → returns ("Other", "Misc", 0.0, meta)
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL",
            "gemini-2.0-flash-lite-preview",
        )
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.temperature = 0.2

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
                print("[LLM WARN] google-genai SDK not available; LLM disabled.")
            elif not self.api_key:
                print("[LLM WARN] GEMINI_API_KEY missing; LLM disabled.")

    # -------------------------
    # Internal: call Gemini for a batch
    # -------------------------
    def _call_gemini_batch(self, descriptions: List[str]) -> Optional[List[Dict]]:
        """
        Calls Gemini once for a batch of descriptions.

        Expects a JSON array like:
        [
          { "index": 0, "category": "...", "subcategory": "...", "confidence": 0.8, "rationale": "..." },
          ...
        ]

        Returns that list on success, or None on failure.
        """
        if not self.client:
            return None

        items = [{"index": i, "description": d} for i, d in enumerate(descriptions)]

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
                contents=[full_prompt],  # list[str] is valid for google-genai
                config={
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                print(f"[LLM WARN] Gemini quota exhausted: {err_str}")
                # Optionally disable further LLM calls in this process
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
        Batch classify descriptions using Gemini.
        On any error or invalid output, returns ("Other", "Misc", 0.0, meta) for that item.
        """
        n = len(descriptions)
        if n == 0:
            return []

        # If no client, short-circuit
        if not self.client:
            return [
                (
                    "Other",
                    "Misc",
                    0.0,
                    {"reason": "no_llm_client", "model": self.model_name},
                )
                for _ in descriptions
            ]

        # Call Gemini
        gemini_output = self._call_gemini_batch(descriptions)

        # If the batch call failed: everything falls back
        if gemini_output is None:
            return [
                (
                    "Other",
                    "Misc",
                    0.0,
                    {"reason": "llm_call_failed", "model": self.model_name},
                )
                for _ in descriptions
            ]

        # Build a map index -> object
        by_index: Dict[int, Dict] = {}
        for obj in gemini_output:
            if isinstance(obj, dict) and "index" in obj:
                try:
                    idx = int(obj["index"])
                    by_index[idx] = obj
                except Exception:
                    continue

        results: List[Tuple[str, Optional[str], float, Dict]] = []

        for i, desc in enumerate(descriptions):
            if i not in by_index:
                # Missing entry → fallback
                results.append(
                    (
                        "Other",
                        "Misc",
                        0.0,
                        {
                            "reason": "missing_index_in_llm_output",
                            "model": self.model_name,
                        },
                    )
                )
                continue

            o = by_index[i]
            cat = o.get("category") or "Other"
            sub = o.get("subcategory") or "Misc"

            # If the LLM itself returns "Other.Misc" that's fine; we still trust it more than nothing.
            try:
                conf = float(o.get("confidence", 0.7))
            except Exception:
                conf = 0.7

            meta = {
                "model": self.model_name,
                "rationale": o.get("rationale", ""),
                "source": "gemini",
            }

            results.append((cat, sub, conf, meta))

        return results

    # -------------------------
    # Public: single classify
    # -------------------------
    def classify(
        self, description: str
    ) -> Tuple[str, Optional[str], float, Dict]:
        """
        Single description entry point.
        Uses classify_batch internally.
        """
        if not description or not description.strip():
            return "Other", "Misc", 0.0, {"reason": "empty", "model": self.model_name}

        return self.classify_batch([description])[0]


# Global instance used in UnifiedPipeline
llm_clf = LLMFallbackClassifier()
