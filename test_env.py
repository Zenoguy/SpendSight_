from dotenv import load_dotenv
import os

load_dotenv()
print("DEFAULT_USER_ID =", os.getenv("DEFAULT_USER_ID"))
