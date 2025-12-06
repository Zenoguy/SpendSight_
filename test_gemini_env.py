import os
from dotenv import load_dotenv

print("🔄 Loading .env...")
load_dotenv()

key = os.getenv("GEMINI_API_KEY")

print("GEMINI_API_KEY =", key)

if not key:
    print("❌ Key NOT FOUND. Check .env and variable name.")
else:
    print("✅ Key FOUND in environment!")

# Optional: test Google GenAI initialization
try:
    from google import genai
    
    print("\n🔄 Initializing Google GenAI client...")
    client = genai.Client(api_key=key)
    print("✅ Google GenAI client initialized successfully!")
except Exception as e:
    print("❌ Google GenAI initialization failed:")
    print(e)
