# minimal_gemini_test.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key: {api_key[:8] if api_key else 'None'}...")

if api_key:
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured")
        
        # Try the gemini-pro model
        print("🧪 Testing gemini-pro model...")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say 'Hello, World!' in 5 different languages")
        print("✅ gemini-pro model working! Response:")
        print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
    except Exception as e:
        print(f"❌ Error with gemini-pro: {e}")
        
        # Try gemini-1.0-pro as fallback
        try:
            print("🧪 Testing gemini-1.0-pro model...")
            model = genai.GenerativeModel('gemini-1.0-pro')
            response = model.generate_content("Say 'Hello, World!' in 5 different languages")
            print("✅ gemini-1.0-pro model working! Response:")
            print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
        except Exception as e2:
            print(f"❌ Error with gemini-1.0-pro: {e2}")
            
            # Try listing available models
            try:
                print("📋 Listing available models...")
                for m in genai.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        print(f"  ✅ {m.name}")
            except Exception as e3:
                print(f"❌ Error listing models: {e3}")
else:
    print("No API key found")