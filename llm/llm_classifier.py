# nlp/llm_classifier.py
import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Tuple, Dict, Optional

# --- 1. Define the Required JSON Schema using Pydantic ---
# This ensures the LLM's output is structured and easy to parse.
class ClassificationOutput(BaseModel):
    """Defines the canonical output format for LLM classification."""
    category: str = Field(description="The primary category (e.g., 'Food', 'Travel').")
    subcategory: Optional[str] = Field(description="The secondary category (e.g., 'Groceries', 'Flight').")
    explanation: str = Field(description="A brief justification for the classification.")

# --- 2. LLM Fallback Classifier ---
class LLMFallbackClassifier:
    # ...
    def __init__(self, model_name="gemini-2.5-flash"):
        
        retrieved_key = os.getenv("GEMINI_API_KEY")

        # --- TEMPORARY DEBUG PRINT ---
        if retrieved_key:
            print(f"[DEBUG] API Key successfully retrieved. Length: {len(retrieved_key)}")
        else:
            print("[DEBUG] ERROR: GEMINI_API_KEY is None or empty!")
        # -----------------------------
        
        self.client = genai.Client(api_key=retrieved_key) # This line caused the error
        # ...
        # The prompt is simplified, relying on the Pydantic schema for structure
        self.prompt_template = """
            You are a financial transaction classification expert.
            Based on the description: "{description}", classify the transaction.
            Use the provided categories: [Food.Groceries, Food.Dining, Travel.Fuel, ... (full list)].
            If a subcategory is not explicitly available, you may provide a suitable one or use None.
        """
        print(f"[LLM] LLM Fallback Classifier initialized with {model_name}.")

    def classify(self, description: str) -> Tuple[str, Optional[str], float, Dict]:
        """Calls the Gemini API to classify the transaction."""
        
        prompt = self.prompt_template.format(description=description)
        
        try:
            # --- REAL LLM API CALL (Replaces MOCK) ---
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    # Instruct the model to output a JSON object matching the schema
                    response_mime_type="application/json",
                    response_schema=ClassificationOutput,
                ),
            )
            
            # The response text will be a valid JSON string matching ClassificationOutput
            json_output = response.text.strip()
            classification_data = ClassificationOutput.model_validate_json(json_output)

            # Assign a default high confidence for LLM results
            confidence = 0.95 

            meta = {
                "model": self.model_name,
                "raw_prompt": prompt,
                "llm_output": json_output # Store the full JSON output
            }
            
            return (
                classification_data.category,
                classification_data.subcategory,
                confidence,
                meta
            )

        except Exception as e:
            print(f"[LLM ERROR] Classification failed for '{description}': {e}")
            # Fallback for API failure (returning UNCLEAR status)
            return "UNCLEAR", None, 0.50, {"error": str(e), "raw_prompt": prompt}

# Initialize the LLM classifier for the pipeline
llm_clf = LLMFallbackClassifier()