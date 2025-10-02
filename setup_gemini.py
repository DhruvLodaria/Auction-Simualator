#!/usr/bin/env python3
"""
Gemini API Setup Helper
This script helps you configure your Gemini API key for the bidding game.
"""

import os
import sys

def setup_gemini_api():
    """Interactive setup for Gemini API key"""
    print("🤖 Gemini API Setup Helper")
    print("=" * 40)
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print("❌ .env file not found!")
        print("Creating .env file...")
        with open(env_file, 'w') as f:
            f.write("# Gemini API Configuration\n")
            f.write("GEMINI_API_KEY=your_actual_api_key_here\n")
            f.write("\n# Flask Configuration\n")
            f.write("FLASK_ENV=development\n")
            f.write("FLASK_DEBUG=True\n")
        print("✅ Created .env file")
    
    # Read current .env content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Check current API key
    current_key = None
    for line in content.split('\n'):
        if line.startswith('GEMINI_API_KEY='):
            current_key = line.split('=', 1)[1]
            break
    
    print("\n🔑 Current API Key Status:")
    if not current_key or current_key == 'your_actual_api_key_here':
        print("❌ No valid API key configured")
    else:
        print(f"✅ API key configured: {current_key[:10]}...")
    
    print("\n📋 Steps to get your Gemini API key:")
    print("1. Visit: https://aistudio.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key (starts with AIzaSy...)")
    
    # Ask user for API key
    print("\n🔧 Enter your Gemini API key:")
    print("(Press Enter to skip if you want to set it manually)")
    
    new_key = input("API Key: ").strip()
    
    if new_key:
        # Validate key format
        if not new_key.startswith('AIzaSy') or len(new_key) < 30:
            print("⚠️  Warning: This doesn't look like a valid Gemini API key")
            print("   Valid keys start with 'AIzaSy' and are ~39 characters long")
            confirm = input("Continue anyway? (y/n): ").lower()
            if confirm != 'y':
                print("❌ Setup cancelled")
                return False
        
        # Update .env file
        lines = content.split('\n')
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('GEMINI_API_KEY='):
                lines[i] = f"GEMINI_API_KEY={new_key}"
                updated = True
                break
        
        if not updated:
            lines.append(f"GEMINI_API_KEY={new_key}")
        
        # Write back to file
        with open(env_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ API key updated in .env file")
        
        # Test the API keyimport os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key found: {api_key[:10]}..." if api_key else "No API key found")

if api_key:
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Test the API
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test.")
        print("✅ API key test successful!")
        print(f"Response: {response.text[:50]}...")
    except Exception as e:
        print(f"❌ API key test failed: {e}")
else:
    print("❌ No API key found in environment variables")
    print("\n🧪 Testing API key...")
    try:
            import google.generativeai as genai
            genai.configure(api_key=new_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Hello")
            print("✅ API key test successful!")
            return True
    except ImportError:
            print("⚠️  google-generativeai package not installed")
            print("   Run: pip install google-generativeai")
            return False
    except Exception as e:
            print(f"❌ API key test failed: {e}")
            if "API_KEY_INVALID" in str(e):
                print("   Please check your API key at https://aistudio.google.com/app/apikey")
        return False
            else:
        print("\n📝 Manual setup instructions:")
        print(f"1. Edit the file: {os.path.abspath(env_file)}")
        print("2. Replace 'your_actual_api_key_here' with your real API key")
        print("3. Save the file and restart the application")
        return False

def main():
    """Main function"""
    try:
        success = setup_gemini_api()
        
        print("\n" + "=" * 40)
        if success:
            print("🎉 Setup complete! You can now:")
            print("   1. Run: python app.py")
            print("   2. Select 'Gemini AI' agent in the game")
            print("   3. Enjoy enhanced AI-powered bidding!")
        else:
            print("⚠️  Setup incomplete. Please:")
            print("   1. Get your API key from https://aistudio.google.com/app/apikey")
            print("   2. Update the .env file manually")
            print("   3. Run this script again to test")
        
        print("\n💡 Need help? Check GEMINI_SETUP.md for detailed instructions")
        
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")

if __name__ == "__main__":
    main()