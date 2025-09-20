# Gemini API Integration Setup Guide

## Overview
This project now supports Google's Gemini AI as an intelligent bidding agent. The Gemini AI agent uses advanced language model capabilities to make strategic bidding decisions based on market context and historical patterns.

## Setup Instructions

### 1. Get a Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 2. Configure the API Key
**Option A: Environment Variable (Recommended)**
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and replace `your_gemini_api_key_here` with your actual API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

**Option B: System Environment Variable**
Set the environment variable in your system:
- Windows: `set GEMINI_API_KEY=your_actual_api_key_here`
- Linux/Mac: `export GEMINI_API_KEY=your_actual_api_key_here`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Setup
1. Start the application: `python app.py`
2. Open the web interface
3. Look for "Gemini AI" in the agent selection panel
4. If the API key is configured correctly, you'll see the Gemini AI agent available

## Features

### Gemini AI Agent Capabilities
- **Strategic Analysis**: Uses market context and historical data to make informed decisions
- **Adaptive Learning**: Adjusts risk tolerance based on performance
- **Reasoning Display**: Shows the AI's decision-making process in the game log
- **Confidence Indicators**: Visual indicators showing the AI's confidence level
- **Fallback Strategy**: Automatically falls back to rule-based bidding if API fails

### UI Enhancements
- **Reasoning Display**: See why the Gemini AI made each bidding decision
- **Confidence Indicators**: 💪 (high confidence), 🤔 (medium), 😰 (low)
- **AI Strategy Tags**: Clear identification of AI-powered decisions

## Troubleshooting

### Gemini AI Agent Not Available
- Check if `GEMINI_API_KEY` environment variable is set
- Verify the API key is valid and active
- Ensure `google-generativeai` package is installed

### API Rate Limits
- Gemini API has usage limits; the agent will fallback to rule-based strategy if limits are exceeded
- Consider upgrading your Gemini API plan for higher limits

### Error Handling
- If the Gemini API fails, the agent automatically falls back to a simple competitive strategy
- Check the console logs for detailed error messages

## Usage Tips

1. **Multi-Agent Games**: Try combining Gemini AI with other strategy agents for complex market dynamics
2. **Strategy Analysis**: Watch the reasoning display to understand how AI adapts to different scenarios
3. **Performance Tuning**: The AI adjusts its risk tolerance based on win/loss ratio over time

## API Costs
- Gemini API usage is charged per request
- Monitor your usage in Google AI Studio
- The free tier includes generous limits for testing

## Security Notes
- Never commit your actual API key to version control
- Use the `.env` file method for local development
- Consider using secure environment variable management for production

# Gemini API Setup Guide

## Getting Your Gemini API Key

1. **Go to Google AI Studio**: Visit https://makersuite.google.com/app/apikey
2. **Sign in**: Use your Google account to sign in
3. **Create API Key**: Click "Create API Key" button
4. **Copy the Key**: Copy the generated API key

## Setting Up the API Key

### Option 1: Environment Variable (Recommended)
```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_actual_api_key_here"

# Windows (Command Prompt)
set GEMINI_API_KEY=your_actual_api_key_here

# Linux/Mac
export GEMINI_API_KEY="your_actual_api_key_here"
```

### Option 2: .env File
Edit the `.env` file in your project directory:
```
GEMINI_API_KEY=your_actual_api_key_here
```

## Restart the Application
After setting the API key, restart the Flask application:
```bash
python app.py
```

You should see: "✅ Gemini API ready for dynamic software generation and bid analysis"

## Features Enabled with Gemini API
- AI-generated software items for bidding
- Intelligent bid analysis and commentary
- Enhanced AI reasoning in game logs
- Gemini AI agent with strategic bidding
