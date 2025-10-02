# comprehensive_gemini_test.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
print(f"Using API Key: {api_key[:8] if api_key else 'None'}...")

if api_key:
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured")
        
        # Test the model used in your application
        print("🧪 Testing models/gemini-2.0-flash...")
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # Test 1: Simple response
        response = model.generate_content("What is 2+2?")
        print("✅ Simple test working! Response:")
        print(f"   {response.text.strip()}")
        
        # Test 2: JSON format response (similar to what your app uses)
        print("\n🧪 Testing JSON format response...")
        prompt = """
        Provide your response in this exact JSON format:
        {
          "action": "BID",
          "bid_amount": 50,
          "reasoning": "Simple test bid",
          "confidence": 0.8
        }
        Make a bid decision for an item worth $100 with current highest bid at $40.
        """
        response = model.generate_content(prompt)
        print("✅ JSON format test working! Response:")
        print(f"   {response.text[:200]}{'...' if len(response.text) > 200 else ''}")
        
        # Test 3: Check if response contains JSON
        if '{' in response.text and '}' in response.text:
            print("✅ Response contains JSON format as expected")
        else:
            print("⚠️  Response may not contain proper JSON format")
            
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No API key found")