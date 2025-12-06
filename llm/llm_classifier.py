# nlp/llm_classifier.py
"""
LLM Fallback classifier using Google GenAI (google-genai).
This module is defensive: it will NOT crash on import if the API key is missing
or if the GenAI call fails. Instead it falls back to a lightweight rule-based
classifier that returns a safe "UNCLEAR"/low-confidence result.

Exports:
    llm_clf  -- an initialized instance of LLMFallbackClassifier
"""

import os
import json
from typing import Tuple, Optional, Dict

# pydantic used for optional validation (keeps interface stable)
from pydantic import BaseModel, Field, ValidationError

# Attempt to import the new genai SDK. If unavailable, we'll still operate in fallback mode.
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    types = None
    GENAI_AVAILABLE = False

# -------------------------
# Structured output schema
# -------------------------
class ClassificationOutput(BaseModel):
    category: str = Field(..., description="Primary category, e.g., 'Food'")
    subcategory: Optional[str] = Field(None, description="Secondary category, e.g., 'Groceries'")
    explanation: Optional[str] = Field(None, description="Short explanation/justification")


# -------------------------
# LLM Fallback Classifier
# -------------------------
class LLMFallbackClassifier:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the classifier.

        - model_name: model identifier to request from GenAI SDK.
        - If GEMINI_API_KEY is missing, the classifier will operate in fallback mode.
        """
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")

        # Defensive: only create a genai.Client if the sdk is present and key exists
        self.client = None
        if GENAI_AVAILABLE and self.api_key:
            try:
                # genai.Client is available in google-genai
                self.client = genai.Client(api_key=self.api_key)
                print("[LLM] GenAI client initialized.")
            except Exception as e:
                print(f"[LLM WARN] Failed to initialize GenAI client: {e}")
                self.client = None
        else:
            if not GENAI_AVAILABLE:
                print("[LLM WARN] google-genai SDK not available; LLM fallback will use rule-based mock.")
            else:
                print("[LLM WARN] GEMINI_API_KEY missing; LLM fallback will use rule-based mock.")

        # A simple prompt template (kept small; we validate output below)
        self.prompt_template = (
            "You are a financial transaction classification expert.\n"
            "Classify the transaction described below into a category and optional subcategory.\n"
            "Return a compact JSON object with keys: category, subcategory or null, explanation.\n\n"
            "Description: \"{description}\"\n\n"
            "Example categories: Food.Groceries, Food.Dining, Travel.Fuel, Transport.Ride-hailing, "
            "Shopping.Online, Utilities.Electricity, Salary, Transfer\n"
        )

    def _call_genai(self, description: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Try to call the GenAI SDK and return (parsed_json_dict, raw_text).
        On any failure return (None, raw_text_or_error).
        This function is defensive and tolerates multiple SDK calling patterns.
        """

        if not self.client:
            return None, "no-client"

        prompt = self.prompt_template.format(description=description)

        # Try the most-likely SDK call pattern(s). Wrap in try/except and fall back on error.
        try:
            # Preferred attempt: models.generate_content (some versions expose this)
            if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
                raw = getattr(resp, "text", None) or str(resp)
                return json.loads(raw), raw

            # Alternate attempt: responses.generate (other versions)
            if hasattr(self.client, "responses") and hasattr(self.client.responses, "generate"):
                # The responses.generate may return an object; try to locate text
                resp = self.client.responses.generate(
                    model=self.model_name,
                    input=prompt,
                )
                # Attempt to extract text safely
                raw = None
                try:
                    # Many responses have .output[0].content[0].text or .output_text
                    if hasattr(resp, "output_text"):
                        raw = resp.output_text
                    else:
                        # fallback parsing
                        raw = json.dumps(resp.__dict__)
                except Exception:
                    raw = str(resp)
                # Try parse
                try:
                    return json.loads(raw), raw
                except Exception:
                    # maybe resp.output was structured
                    return None, raw

            # If none of above available, return None
            return None, "unsupported-client-api"

        except Exception as e:
            return None, f"genai-call-failed: {e}"

    def _rule_based_fallback(self, description: str) -> Tuple[str, Optional[str], float, Dict]:
        """
        Deterministic lightweight fallback classifier when LLM is unavailable.
        Returns low-confidence predictions so the pipeline can route to human/LLM if desired.
        """
        d = description.lower()
        meta = {"method": "rule-fallback"}

        # Simple heuristics
        if any(k in d for k in ["uber", "ola", "ride", "cab"]):
            return "Transport", "Ride-hailing", 0.60, meta
        if any(k in d for k in ["starbucks", "cafe", "coffee", "dining", "restaurant"]):
            return "Food", "Dining", 0.60, meta
        if any(k in d for k in ["grocery", "supermarket", "bigbasket", "dmart", "grocer"]):
            return "Food", "Groceries", 0.60, meta
        if any(k in d for k in ["fuel", "petrol", "gasoline", "bharat petrol", "indianoil"]):
            return "Travel", "Fuel", 0.60, meta
        if any(k in d for k in ["salary", "payroll", "credit salary"]):
            return "Income", "Salary", 0.95, meta
        if any(k in d for k in ["transfer", "neft", "imps", "rtgs"]):
            return "Transfer", None, 0.55, meta

        # Default unclear
        return "PENDING", None, 0.40, meta

    def classify(self, description: str) -> Tuple[str, Optional[str], float, Dict]:
        """
        Public method to classify a single transaction description.
        Returns: (category, subcategory or None, confidence (0-1 float), meta dict)
        """
        clean_desc = (description or "").strip()
        if not clean_desc:
            return "PENDING", None, 0.0, {"reason": "empty_description"}

        # Try GenAI if client exists
        parsed, raw = self._call_genai(clean_desc)
        if parsed:
            try:
                # Validate using pydantic to ensure predictable fields
                validated = ClassificationOutput.model_validate(parsed)
                category = validated.category
                subcategory = validated.subcategory
                explanation = validated.explanation or ""
                confidence = 0.90  # optimistic but reasonable for LLM output
                meta = {"model": self.model_name, "raw": raw}
                return category, subcategory, confidence, meta
            except ValidationError as e:
                # If schema doesn't match, fall through to fallback route
                return "UNCLEAR", None, 0.50, {"error": "validation_failed", "details": str(e), "raw": raw}

        # If GenAI call not available or failed, use rule-based fallback
        return self._rule_based_fallback(clean_desc)


# Instantiate a global classifier used by the pipeline
llm_clf = LLMFallbackClassifier()
