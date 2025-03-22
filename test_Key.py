import google.generativeai as genai
import os

# Ensure API key is correctly set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found!")
else:
    print("✅ API Key Loaded Successfully!")

# Configure Google Gemini API
genai.configure(api_key=api_key)

# Test API Connection
try:
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content("Hello, how are you?")
    print("✅ API Response:", response.text)
except Exception as e:
    print("❌ API Connection Error:", str(e))
