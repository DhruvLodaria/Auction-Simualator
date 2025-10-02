#!/usr/bin/env python3
"""
RL Bidding Game - Flask Application
A competitive bidding game where a human player faces off against a reinforcement learning AI agent.
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
import os
import json
import random
import math
from datetime import datetime
from ai_agents import create_agent_by_name, get_all_available_agents
import io
import csv

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, environment variables should be set manually
app = Flask(__name__)

app.secret_key = 'your-secret-key-here'  # Change this in production

# Enable CORS for all routes
CORS(app)

class BiddingGame:
    def __init__(self):
        # Initialize Gemini for dynamic item generation and analysis
        self.gemini_available = False
        self.gemini_model = None
        
        try:
            import google.generativeai as genai
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key and api_key != 'your_actual_api_key_here':
                genai.configure(api_key=api_key)  # type: ignore
                # Test the API key with a simple request
                test_model = genai.GenerativeModel('models/gemini-2.0-flash')  # type: ignore
                test_response = test_model.generate_content("Hello")
                self.gemini_model = test_model
                self.gemini_available = True
                print("✅ Gemini API initialized and tested successfully")
            else:
                print("⚠️  Gemini API key not configured. Please set GEMINI_API_KEY in .env file")
                print("   Visit https://aistudio.google.com/app/apikey to get your API key")
        except ImportError:
            print("⚠️  google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            print(f"⚠️  Gemini API configuration failed: {e}")
            print("   Please check your API key at https://aistudio.google.com/app/apikey")
        
        # Fallback real-world software items when Gemini is not available
        self.fallback_items = [
            {"name": "Netflix Standard", "base_price": 179.88, "category": "Streaming", "vendor": "Netflix", "description": "Watch movies and TV shows"},
            {"name": "Spotify Premium", "base_price": 119.88, "category": "Music", "vendor": "Spotify", "description": "Stream music ad-free"},
            {"name": "Microsoft 365 Personal", "base_price": 69.99, "category": "Productivity", "vendor": "Microsoft", "description": "Word, Excel, PowerPoint suite"},
            {"name": "Adobe Creative Cloud", "base_price": 263.88, "category": "Design", "vendor": "Adobe", "description": "Photoshop, Illustrator, video editing"},
            {"name": "Zoom Pro", "base_price": 179.88, "category": "Communication", "vendor": "Zoom", "description": "Video conferencing and meetings"},
            {"name": "Dropbox Plus", "base_price": 119.88, "category": "Storage", "vendor": "Dropbox", "description": "Cloud file storage and sync"},
            {"name": "Canva Pro", "base_price": 119.99, "category": "Design", "vendor": "Canva", "description": "Graphic design and presentations"},
            {"name": "Grammarly Premium", "base_price": 144.00, "category": "Writing", "vendor": "Grammarly", "description": "Advanced grammar and writing assistant"},
            {"name": "LastPass Premium", "base_price": 36.00, "category": "Security", "vendor": "LastPass", "description": "Password manager and security"},
            {"name": "GitHub Pro", "base_price": 48.00, "category": "Development", "vendor": "GitHub", "description": "Code repositories and collaboration"}
        ]
        
        # Track generated items for consistency
        self.generated_items_cache = []
        
    def generate_software_with_gemini(self):
        """Use Gemini API to generate realistic software items with market pricing"""
        if not self.gemini_available:
            return None
            
        try:
            prompt = """
Generate a REAL-WORLD SOFTWARE that people use every day in their personal or professional lives.

**STRICT REQUIREMENTS:**
1. Must be ACTUAL software that exists today and is widely used
2. Must be software people use regularly (Netflix, Spotify, Adobe, Microsoft Office, Zoom, etc.)
3. Must provide the REAL annual subscription price that customers actually pay
4. Must include the actual company/vendor name

**Examples of Real Software with Annual Pricing:**
- Netflix Standard: $179.88/year (streaming movies and TV shows)
- Spotify Premium: $119.88/year (music streaming service)
- Adobe Creative Cloud: $263.88/year (design and photo editing)
- Microsoft 365 Personal: $69.99/year (Word, Excel, PowerPoint suite)
- Zoom Pro: $179.88/year (video conferencing)
- Dropbox Plus: $119.88/year (cloud file storage)
- Canva Pro: $119.99/year (graphic design tool)
- LastPass Premium: $36/year (password manager)
- GitHub Pro: $48/year (code repositories)
- Grammarly Premium: $144/year (writing assistant)

**JSON Response Format:**
{
  "name": "[Actual Software Name]",
  "category": "[Software Category]",
  "base_price": [annual_price_number],
  "description": "[What people use this software for daily]",
  "vendor": "[Real Company Name]",
  "pricing_model": "Annual Subscription",
  "market_segment": "Individual"
}

Generate ONE real software with annual price between $30-$3000:
"""
            
            response = self.gemini_model.generate_content(prompt)  # type: ignore
            
            # Parse response
            response_text = response.text.strip()
            if '```json' in response_text:
                json_start = response_text.find('{', response_text.find('```json'))
                json_end = response_text.rfind('}') + 1
                response_text = response_text[json_start:json_end]
            elif '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                response_text = response_text[json_start:json_end]
            
            software_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['name', 'category', 'base_price']
            if all(field in software_data for field in required_fields):
                # Ensure price is reasonable for real annual subscriptions ($30-$300)
                base_price = float(software_data['base_price'])
                if base_price < 30:
                    base_price = random.uniform(35, 80)  # Adjust to realistic annual subscription price
                elif base_price > 300:
                    base_price = random.uniform(100, 250)  # Keep within typical consumer software range
                
                software_data['base_price'] = round(base_price, 2)
                
                # Ensure we have proper fields for real software
                if 'vendor' not in software_data:
                    software_data['vendor'] = 'Software Company'
                if 'pricing_model' not in software_data:
                    software_data['pricing_model'] = 'Annual subscription'
                    
                return software_data
            
        except Exception as e:
            print(f"Error generating software with Gemini: {e}")
            
        return None
    
    def analyze_bid_feasibility(self, software_item, final_bid, winner, bid_sequence):
        """Use Gemini API to analyze if the final bid makes business sense"""
        if not self.gemini_available:
            return {
                'analysis': "Business analysis not available (Gemini API required)",
                'is_market_winner': None,
                'value_assessment': 'unknown'
            }
            
        try:
            bid_history = " → ".join([f"{bid['bidder']}: ${bid['amount']:.2f}" for bid in bid_sequence])
            market_price = software_item['actual_price']
            
            prompt = f"""
Analyze this software auction and determine if the winner got a good deal or overpaid:

**Software:** {software_item['name']} by {software_item.get('vendor', 'Unknown')}
**Market Price:** ${market_price:.2f}
**Final Bid:** ${final_bid:.2f}
**Winner:** {winner}
**Price Difference:** {((final_bid - market_price) / market_price * 100):+.1f}%

**Task:** Provide exactly 2 sentences:
1. First sentence: State clearly if this is a GOOD DEAL or OVERPAID
2. Second sentence: Explain why with specific numbers

**Classification Rules:**
- GOOD DEAL: Paid at or below market value (≤105% of market price)
- OVERPAID: Paid significantly above market value (>105% of market price)

**Response Format (JSON):**
{{
  "is_market_winner": true/false,
  "value_assessment": "EXCELLENT/GOOD/FAIR/POOR/TERRIBLE",
  "analysis": "[2-sentence analysis starting with 'GOOD DEAL' or 'OVERPAID']",
  "price_difference_percent": {((final_bid - market_price) / market_price * 100):.1f},
  "recommendation": "Brief bidding advice"
}}

Provide your analysis:
"""
            
            response = self.gemini_model.generate_content(prompt)  # type: ignore
            
            # Parse JSON response
            try:
                response_text = response.text.strip()
                if '```json' in response_text:
                    json_start = response_text.find('{', response_text.find('```json'))
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
                elif '{' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
                
                market_analysis = json.loads(response_text)
                
                # Validate and set defaults
                market_analysis.setdefault('is_market_winner', final_bid <= market_price * 1.05)
                market_analysis.setdefault('value_assessment', 'FAIR')
                market_analysis.setdefault('analysis', f"Winner paid ${final_bid:.2f} for ${market_price:.2f} software.")
                market_analysis.setdefault('price_difference_percent', (final_bid - market_price) / market_price * 100)
                
                return market_analysis
                
            except (json.JSONDecodeError, KeyError) as e:
                # Fallback analysis based on simple price comparison
                price_diff_percent = (final_bid - market_price) / market_price * 100
                is_winner = final_bid <= market_price * 1.12  # Allow 15% overpay for good deal
                
                if price_diff_percent <= -10:
                    assessment = 'EXCELLENT'
                    analysis_text = f"GOOD DEAL: {winner} paid ${final_bid:.2f} for ${market_price:.2f} software, saving {abs(price_diff_percent):.1f}%. This represents excellent value with significant savings below market price."
                elif price_diff_percent <= 0:
                    assessment = 'GOOD'
                    analysis_text = f"GOOD DEAL: {winner} paid ${final_bid:.2f} for ${market_price:.2f} software, at or below market value. Smart bidding strategy with {abs(price_diff_percent):.1f}% savings."
                elif price_diff_percent <= 5:
                    assessment = 'FAIR'
                    analysis_text = f"GOOD DEAL: {winner} paid ${final_bid:.2f} for ${market_price:.2f} software, just {price_diff_percent:.1f}% above market. Reasonable purchase within acceptable range."
                else:
                    assessment = 'POOR'
                    analysis_text = f"OVERPAID: {winner} paid ${final_bid:.2f} for ${market_price:.2f} software, {price_diff_percent:.1f}% above market value. This represents poor value with significant overpayment."
                
                return {
                    'is_market_winner': is_winner,
                    'value_assessment': assessment,
                    'analysis': analysis_text,
                    'price_difference_percent': price_diff_percent,
                    'recommendation': 'Consider market value when bidding.'
                }
            
        except Exception as e:
            # Handle specific API errors with helpful messages
            error_str = str(e)
            if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
                analysis_text = "Invalid Gemini API key. Please update your GEMINI_API_KEY in the .env file with a valid key from https://aistudio.google.com/app/apikey"
            elif "quota" in error_str.lower() or "limit" in error_str.lower():
                analysis_text = "Gemini API quota exceeded. Please check your usage limits at https://aistudio.google.com"
            else:
                analysis_text = f"Gemini API error: {error_str}"
            
            # Fallback analysis
            market_price = software_item['actual_price']
            price_diff_percent = (final_bid - market_price) / market_price * 100
            is_winner = final_bid <= market_price * 1.05
            
            if price_diff_percent <= 5:
                analysis_text = f"GOOD DEAL: {winner} paid ${final_bid:.2f} for ${market_price:.2f} software, within acceptable range. Technical analysis unavailable but price appears reasonable."
            else:
                analysis_text = f"OVERPAID: {winner} paid ${final_bid:.2f} for ${market_price:.2f} software, {price_diff_percent:.1f}% above market value. Technical analysis unavailable but significant overpayment detected."
            
            return {
                'analysis': analysis_text,
                'is_market_winner': is_winner,
                'value_assessment': 'FAIR' if is_winner else 'POOR',
                'price_difference_percent': price_diff_percent,
                'recommendation': 'Please configure a valid Gemini API key for enhanced analysis.'
            }
        
    def generate_random_item(self):
        """Generate a random software item using Gemini or fallback to static list"""
        # Try Gemini first
        if self.gemini_available and random.random() < 0.7:  # 70% chance to use Gemini
            gemini_item = self.generate_software_with_gemini()
            if gemini_item:
                # Add price variation (±15% for Gemini-generated items)
                variation = 0.85 + random.random() * 0.3  # 0.85 to 1.15
                actual_price = round(gemini_item["base_price"] * variation, 2)
                
                generated_item = {
                    "name": gemini_item["name"],
                    "category": gemini_item["category"],
                    "base_price": gemini_item["base_price"],
                    "actual_price": actual_price,
                    "description": gemini_item.get("description", "Real-world software"),
                    "vendor": gemini_item.get("vendor", "Software Vendor"),
                    "pricing_model": gemini_item.get("pricing_model", "Subscription"),
                    "market_segment": gemini_item.get("market_segment", "Business"),
                    "generated_by": "gemini"
                }
                
                # Cache for analysis
                self.generated_items_cache.append(generated_item)
                return generated_item
        
        # Fallback to real-world software
        base_item = random.choice(self.fallback_items)
        variation = 0.9 + random.random() * 0.2  # 0.9 to 1.1 (smaller variation for realistic pricing)
        actual_price = round(base_item["base_price"] * variation, 2)
        
        return {
            "name": base_item["name"],
            "category": base_item["category"],
            "base_price": base_item["base_price"],
            "actual_price": actual_price,
            "description": base_item.get("description", "Real-world software subscription"),
            "vendor": base_item.get("vendor", "Software Company"),
            "pricing_model": "Annual Subscription",
            "market_segment": "Individual",
            "generated_by": "static"
        }
    
    def calculate_score(self, market_value, final_bid):
        """Calculate score based on bidding efficiency"""
        return max(50, round((market_value - final_bid) * 10) + 50)

# Initialize game instance and current AI agents
game = BiddingGame()
current_agents = {}  # Dictionary to store multiple agent instances
active_agents = []   # List of currently active agent IDs

@app.route('/debug')
def debug():
    """Debug page"""
    return render_template('debug_test.html')

@app.route('/debug_direct')
def debug_direct():
    """Direct debug page"""
    with open('debug_test.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/get_agents', methods=['GET'])
def get_agents():
    """Get list of available AI agents"""
    return jsonify({
        'agents': get_all_available_agents(),
        'current_agents': session.get('active_agent_ids', [])
    })

@app.route('/select_agents', methods=['POST'])
def select_agents():
    """Select multiple AI agents to compete against"""
    global current_agents, active_agents
    
    try:
        print(f"DEBUG: select_agents called with method: {request.method}")
        print(f"DEBUG: Content-Type: {request.content_type}")
        
        # Handle JSON parsing with proper error handling
        try:
            data = request.get_json(force=True) if request.data else None
        except Exception as json_error:
            print(f"JSON parsing failed: {json_error}")
            return jsonify({'error': 'Invalid JSON data provided'}), 400
        
        print(f"DEBUG: Received data: {data}")
        
        if not data:
            print("DEBUG: No data provided")
            return jsonify({'error': 'No data provided'}), 400
            
        agent_ids = data.get('agent_ids', [])
        print(f"DEBUG: Agent IDs: {agent_ids}")
        
        if not agent_ids:
            return jsonify({'error': 'At least one agent ID is required'}), 400
        
        # Create agent instances
        current_agents = {}
        for agent_id in agent_ids:
            current_agents[agent_id] = create_agent_by_name(agent_id)
        
        active_agents = agent_ids
        session['active_agent_ids'] = agent_ids
        
        # Set up collusion coordination if multiple collusive agents are selected
        collusive_agents = [agent for agent in current_agents.values() 
                          if hasattr(agent, 'set_collusion_partners')]
        
        if len(collusive_agents) > 1:
            # Enable collusion coordination between collusive agents
            for agent in collusive_agents:
                partners = [other for other in collusive_agents if other != agent]
                agent.set_collusion_partners(partners)
        elif len(collusive_agents) == 1:
            # Single collusive agent - still set up coordination (empty partners list)
            collusive_agents[0].set_collusion_partners([])
        
        agent_info = [{
            'id': agent_id,
            'name': current_agents[agent_id].name,
            'strategy': current_agents[agent_id].strategy
        } for agent_id in agent_ids]
        
        print(f"DEBUG: Successfully created agents: {[info['name'] for info in agent_info]}")
        
        return jsonify({
            'success': True,
            'agents': agent_info
        })
    except Exception as e:
        print(f"DEBUG: Exception in select_agents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({'status': 'ok', 'message': 'Flask is working'})

@app.route('/api_status')
def api_status():
    """Check API and system status"""
    return jsonify({
        'flask_status': 'ok',
        'gemini_available': game.gemini_available,
        'agents_count': len(get_all_available_agents()),
        'current_agents': len(current_agents),
        'active_agents': len(active_agents)
    })

@app.route('/')
def index():
    """Main game page"""
    # Initialize session if new
    if 'game_initialized' not in session:
        session['game_initialized'] = True
        session['human_score'] = 0
        session['ai_score'] = 0
        session['human_wins'] = 0
        session['ai_wins'] = 0
        session['agent_wins'] = {}  # Track individual agent wins
        session['game_round'] = 0
        session['game_history'] = []
        
    return render_template('index.html')

@app.route('/start_round', methods=['POST'])
def start_round():
    """Start a new bidding round with Gemini-generated software"""
    session['game_round'] = session.get('game_round', 0) + 1
    
    # Generate item (potentially using Gemini)
    generated_item = game.generate_random_item()
    session['current_item'] = generated_item
    
    session['current_highest_bid'] = 0
    session['current_highest_bidder'] = None
    session['bidding_in_progress'] = True
    session['bid_sequence'] = []
    session['human_abandoned'] = False
    session['all_ai_abandoned'] = False
    
    # Create enhanced message for real-world software
    if generated_item.get('generated_by') == 'gemini':
        message = f"Round {session['game_round']}: Bidding on real-world {generated_item['name']} (Annual: ${generated_item['actual_price']:.2f}/year) 🌐"
        if 'description' in generated_item:
            message += f" - {generated_item['description']}"
    else:
        vendor_info = f" by {generated_item.get('vendor', 'Unknown')}" if generated_item.get('vendor') else ""
        message = f"Round {session['game_round']}: Bidding on {generated_item['name']}{vendor_info} (Annual: ${generated_item['actual_price']:.2f}/year)"
    
    return jsonify({
        'success': True,
        'round': session['game_round'],
        'item': generated_item,
        'message': message
    })

@app.route('/place_bid', methods=['POST'])
def place_bid():
    """Handle human bid placement and multi-agent responses"""
    try:
        # Handle JSON parsing with proper error handling
        try:
            data = request.get_json(force=True) if request.data else None
        except Exception as json_error:
            print(f"JSON parsing failed: {json_error}")
            return jsonify({'error': 'Invalid JSON data provided'}), 400
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        bid_value = float(data.get('bid', 0))
        current_item = session.get('current_item')
        current_highest_bid = session.get('current_highest_bid', 0)
        
        if not current_item:
            return jsonify({'error': 'No active round'}), 400
        
        # Validation
        if bid_value <= 0:
            return jsonify({'error': 'Please enter a valid bid amount!'}), 400
        
        # Allow first bid to be any positive amount (no longer restricted to below market value)
        # if current_highest_bid == 0 and bid_value >= current_item['actual_price']:
        #     return jsonify({
        #         'error': f'First bid must be below market value of ${current_item["actual_price"]:.2f}!'
        #     }), 400
        
        # Subsequent bids must be higher than current highest
        if current_highest_bid > 0 and bid_value <= current_highest_bid:
            return jsonify({
                'error': f'Bid must be higher than current highest bid of ${current_highest_bid:.2f}!'
            }), 400
        
        # Process human bid
        session['current_highest_bid'] = bid_value
        session['current_highest_bidder'] = 'human'
        
        bid_sequence = session.get('bid_sequence', [])
        bid_sequence.append({'bidder': 'human', 'amount': bid_value})
        session['bid_sequence'] = bid_sequence
        
        # Check if agents are selected
        if not active_agents:
            return jsonify({'error': 'Please select AI agents first'}), 400
        
        # Record human bid for all AI agents
        for agent in current_agents.values():
            agent.record_opponent_bid(bid_value)
            
            # Set final game info for neural network agents
            if hasattr(agent, 'set_final_game_info'):
                agent.set_final_game_info(bid_value, current_item['actual_price'])
        
        # Process AI agents' responses
        ai_bids = []
        ai_abandoned = []
        current_bid = bid_value
        
        for agent_id in active_agents:
            agent = current_agents[agent_id]
            
            # Get AI response
            ai_result = agent.make_bid(
                current_item['actual_price'],
                session['game_round'],
                current_bid
            )
            
            if ai_result['abandoned']:
                ai_abandoned.append(agent.name)
            else:
                ai_bid = ai_result['bid']
                ai_bid_info = {
                    'agent_id': agent_id,
                    'agent_name': agent.name,
                    'bid': ai_bid
                }
                
                # Add collusion info if available
                if hasattr(agent, 'my_turn_to_win') and hasattr(agent, 'collusion_active'):
                    ai_bid_info['collusion_status'] = {
                        'my_turn_to_win': agent.my_turn_to_win,
                        'collusion_active': agent.collusion_active
                    }
                
                # Add Gemini AI reasoning if available
                if 'reasoning' in ai_result:
                    ai_bid_info['reasoning'] = ai_result['reasoning']
                if 'confidence' in ai_result:
                    ai_bid_info['confidence'] = ai_result['confidence']
                
                ai_bids.append(ai_bid_info)
                current_bid = max(current_bid, ai_bid)
                
                # Update highest bid if this AI bid is higher
                if ai_bid > session['current_highest_bid']:
                    session['current_highest_bid'] = ai_bid
                    session['current_highest_bidder'] = agent_id
                
                # Add to bid sequence
                bid_sequence.append({'bidder': agent.name, 'amount': ai_bid})
        
        session['bid_sequence'] = bid_sequence
        
        response_data = {
            'success': True,
            'human_bid': bid_value,
            'ai_bids': ai_bids,
            'ai_abandoned': ai_abandoned,
            'current_highest_bid': session['current_highest_bid'],
            'current_highest_bidder': session['current_highest_bidder']
        }
        
        # Check if all AI agents abandoned
        if len(ai_abandoned) == len(active_agents):
            session['all_ai_abandoned'] = True
            end_result = end_round()
            if isinstance(end_result, dict):
                response_data.update(end_result)
            else:
                return end_result
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/abandon_bid', methods=['POST'])
def abandon_bid():
    """Handle human abandoning the bid"""
    current_highest_bid = session.get('current_highest_bid', 0)
    
    if current_highest_bid == 0:
        return jsonify({'error': 'You must place at least one bid before abandoning!'}), 400
    
    session['human_abandoned'] = True
    
    # In multi-agent mode, check if any AI agents are still active
    active_ai_count = 0
    if active_agents:
        # Simulate remaining AI agents continuing to bid against each other
        current_item = session.get('current_item')
        if current_item:
            for agent_id in active_agents:
                agent = current_agents[agent_id]
                # Check if this agent would continue bidding
                if not agent.should_abandon_bid(
                    current_item['actual_price'], 
                    current_highest_bid, 
                    session.get('game_round', 1)
                ):
                    active_ai_count += 1
    
    # If no AI agents want to continue, set all_ai_abandoned
    if active_ai_count == 0:
        session['all_ai_abandoned'] = True
    
    end_result = end_round()
    return jsonify(end_result)

def end_round():
    """End the bidding round and determine winner in multi-agent scenario"""
    current_item = session.get('current_item')
    current_highest_bid = session.get('current_highest_bid', 0)
    current_highest_bidder = session.get('current_highest_bidder')
    bid_sequence = session.get('bid_sequence', [])
    human_abandoned = session.get('human_abandoned', False)
    all_ai_abandoned = session.get('all_ai_abandoned', False)
    
    if not current_item:
        return jsonify({'error': 'No active round'}), 400
        
    actual_price = current_item['actual_price']
    human_won = False
    ai_won = False
    winner = None
    winning_agent = None
    
    if human_abandoned and all_ai_abandoned:
        winner = 'No one (all abandoned)'
    elif human_abandoned:
        # AI wins - determine which agent won
        ai_won = True
        if current_highest_bidder == 'human':
            # Edge case: human abandoned after bidding highest
            winner = 'AI (human abandoned)'
        else:
            # Find winning agent
            for agent_id in active_agents:
                if agent_id == current_highest_bidder:
                    winning_agent = current_agents[agent_id]
                    winner = f'{winning_agent.name} (human abandoned)'
                    break
            
            if not winning_agent and active_agents:
                # Fallback to first active agent
                winning_agent = current_agents[active_agents[0]]
                winner = f'{winning_agent.name} (human abandoned)'
    elif all_ai_abandoned:
        winner = 'Human (all AI abandoned)'
        human_won = True
    else:
        # Determine winner based on who has highest bid and can afford it
        if current_highest_bidder == 'human':
            winner = 'Human'
            human_won = True
        else:
            # Find winning agent
            ai_won = True
            for agent_id in active_agents:
                if agent_id == current_highest_bidder:
                    winning_agent = current_agents[agent_id]
                    winner = winning_agent.name
                    break
            
            if not winning_agent and active_agents:
                # Fallback to first active agent
                winning_agent = current_agents[active_agents[0]]
                winner = winning_agent.name
    
    # Add Gemini market analysis AFTER winner is determined
    market_analysis = None
    print(f"DEBUG: About to check market analysis - game.gemini_available: {game.gemini_available}, current_item: {current_item is not None}, winner: '{winner}'")
    
    if game.gemini_available and current_item and winner:
        print(f"DEBUG: Calling analyze_bid_feasibility with winner='{winner}', bid=${current_highest_bid}")
        try:
            market_analysis = game.analyze_bid_feasibility(
                current_item,
                current_highest_bid,
                winner,
                bid_sequence
            )
            print(f"DEBUG: Market analysis result: {market_analysis}")
        except Exception as e:
            print(f"Error in Gemini market analysis: {e}")
            market_analysis = {
                'analysis': f"Analysis failed: {str(e)}",
                'is_market_winner': None,
                'value_assessment': 'unknown'
            }
    else:
        print(f"DEBUG: Market analysis skipped - gemini_available: {game.gemini_available}, has_item: {current_item is not None}, has_winner: {winner is not None}")
        # Provide a fallback analysis even when Gemini is not available
        if current_item and winner and current_highest_bid > 0:
            price_diff_percent = (current_highest_bid - current_item['actual_price']) / current_item['actual_price'] * 100
            is_winner = current_highest_bid <= current_item['actual_price'] * 1.05
            
            if price_diff_percent <= 5:
                analysis_text = f"GOOD DEAL: {winner} paid ${current_highest_bid:.2f} for ${current_item['actual_price']:.2f} software, within acceptable range. Gemini analysis unavailable."
            else:
                analysis_text = f"OVERPAID: {winner} paid ${current_highest_bid:.2f} for ${current_item['actual_price']:.2f} software, {price_diff_percent:.1f}% above market value. Gemini analysis unavailable."
            
            market_analysis = {
                'analysis': analysis_text,
                'is_market_winner': is_winner,
                'value_assessment': 'FAIR' if is_winner else 'POOR',
                'price_difference_percent': price_diff_percent,
                'recommendation': 'Analysis based on simple price comparison.'
            }
            print(f"DEBUG: Fallback analysis created: {market_analysis}")
    
    # Debug logging
    print(f"DEBUG: Round ended - Winner: {winner}, Market Analysis: {market_analysis is not None}")
    if market_analysis:
        print(f"DEBUG: Market Analysis - Winner: {market_analysis.get('is_market_winner')}, Assessment: {market_analysis.get('value_assessment')}")
    
    # Determine point distribution and win tracking based on market analysis
    # First check if we have market analysis (either from Gemini or fallback)
    poor_performance = False
    if market_analysis:
        poor_performance = (market_analysis.get('is_market_winner') == False or 
                           market_analysis.get('value_assessment', '').upper() in ['POOR', 'TERRIBLE'])
    
    if human_won:
        if poor_performance:
            # Human overpaid or performed poorly - NO POINTS and NO WIN COUNT
            print(f"DEBUG: Human performed poorly - zero points and no win awarded")
            # Do NOT increment human_wins or human_score when performance is poor
        elif market_analysis and market_analysis.get('is_market_winner'):
            # Human made a good deal - award normal points and win
            human_points = game.calculate_score(actual_price, current_highest_bid)
            session['human_score'] = session.get('human_score', 0) + human_points
            session['human_wins'] = session.get('human_wins', 0) + 1
            print(f"DEBUG: Human wins with good deal - gains {human_points} points and a win")
        elif not market_analysis:
            # No market analysis available - only award if not clearly overpaying
            if current_highest_bid <= actual_price * 1.1:  # Allow 10% overpay without analysis
                human_points = game.calculate_score(actual_price, current_highest_bid)
                session['human_score'] = session.get('human_score', 0) + human_points
                session['human_wins'] = session.get('human_wins', 0) + 1
                print(f"DEBUG: Human wins (no analysis, reasonable bid) - gains {human_points} points and a win")
            else:
                print(f"DEBUG: Human bid too high without analysis - no points awarded")
        else:
            # Market analysis exists but neutral - award points
            human_points = game.calculate_score(actual_price, current_highest_bid)
            session['human_score'] = session.get('human_score', 0) + human_points
            session['human_wins'] = session.get('human_wins', 0) + 1
            print(f"DEBUG: Human wins (neutral analysis) - gains {human_points} points and a win")
    elif ai_won and winning_agent:
        agent_id = None
        # Find the winning agent's ID
        for aid, agent in current_agents.items():
            if agent == winning_agent:
                agent_id = aid
                break
        
        if poor_performance:
            # AI overpaid or performed poorly - NO POINTS and NO WIN COUNT
            print(f"DEBUG: {winning_agent.name} performed poorly - zero points and no win awarded to anyone")
            # Do NOT increment ai_wins or agent_wins when performance is poor
        elif market_analysis and market_analysis.get('is_market_winner'):
            # AI made a good deal - award points and win only to winning agent
            ai_points = game.calculate_score(actual_price, current_highest_bid)
            if agent_id:
                # Initialize agent scores dict if not exists
                if 'agent_scores' not in session:
                    session['agent_scores'] = {}
                if 'agent_wins' not in session:
                    session['agent_wins'] = {}
                session['agent_scores'][agent_id] = session['agent_scores'].get(agent_id, 0) + ai_points
                session['agent_wins'][agent_id] = session['agent_wins'].get(agent_id, 0) + 1
                print(f"DEBUG: {winning_agent.name} ({agent_id}) gains {ai_points} points and a win (good deal)")
            session['ai_wins'] = session.get('ai_wins', 0) + 1
        elif not market_analysis:
            # No market analysis available - only award if not clearly overpaying
            if current_highest_bid <= actual_price * 1.1:  # Allow 10% overpay without analysis
                ai_points = game.calculate_score(actual_price, current_highest_bid)
                if agent_id:
                    if 'agent_scores' not in session:
                        session['agent_scores'] = {}
                    if 'agent_wins' not in session:
                        session['agent_wins'] = {}
                    session['agent_scores'][agent_id] = session['agent_scores'].get(agent_id, 0) + ai_points
                    session['agent_wins'][agent_id] = session['agent_wins'].get(agent_id, 0) + 1
                    print(f"DEBUG: {winning_agent.name} ({agent_id}) gains {ai_points} points and a win (no analysis, reasonable bid)")
                session['ai_wins'] = session.get('ai_wins', 0) + 1
            else:
                print(f"DEBUG: {winning_agent.name} bid too high without analysis - no points awarded")
        else:
            # Market analysis exists but neutral - award points
            ai_points = game.calculate_score(actual_price, current_highest_bid)
            if agent_id:
                if 'agent_scores' not in session:
                    session['agent_scores'] = {}
                if 'agent_wins' not in session:
                    session['agent_wins'] = {}
                session['agent_scores'][agent_id] = session['agent_scores'].get(agent_id, 0) + ai_points
                session['agent_wins'][agent_id] = session['agent_wins'].get(agent_id, 0) + 1
                print(f"DEBUG: {winning_agent.name} ({agent_id}) gains {ai_points} points and a win (neutral analysis)")
            session['ai_wins'] = session.get('ai_wins', 0) + 1
    
    # Record game outcome for all AI agents
    for agent_id in active_agents:
        agent = current_agents[agent_id]
        # Agent wins if it's the winning agent, otherwise it loses
        agent_won = winning_agent and agent == winning_agent
        
        # If the winning agent overpaid, provide learning feedback
        if agent_won and market_analysis and market_analysis.get('is_market_winner') == False:
            # This agent made a poor market decision - enhanced learning
            agent.record_game_outcome(False)  # Record as loss for learning
            print(f"Agent {agent.name} recorded as loss due to overpaying")
        else:
            agent.record_game_outcome(agent_won)
    
    # Clear abandonment flags
    session['human_abandoned'] = False
    session['all_ai_abandoned'] = False
    
    # Determine if this is a true win or just who had the highest bid
    true_winner = winner
    if market_analysis and (market_analysis.get('is_market_winner') == False or 
                           market_analysis.get('value_assessment', '').upper() in ['POOR', 'TERRIBLE']):
        true_winner = f"{winner} (Market Loser - No Points Awarded)"
        print(f"DEBUG: Market analysis shows poor performance - {winner} is marked as market loser")
    
    round_result = {
        'round': session['game_round'],
        'item': current_item['name'],
        'actual_price': actual_price,
        'final_bid': current_highest_bid,
        'bid_sequence': bid_sequence,
        'winner': true_winner,  # Updated to show market performance
        'bid_winner': winner,  # Original highest bidder
        'winning_agent': winning_agent.name if winning_agent else None,
        'human_won': human_won,
        'ai_won': ai_won,
        'human_abandoned': human_abandoned,
        'all_ai_abandoned': all_ai_abandoned,
        'market_analysis': market_analysis,  # Include full market analysis
        'item_details': current_item,  # Include full item details for analysis
        'is_market_winner': market_analysis.get('is_market_winner') if market_analysis else None
    }
    
    game_history = session.get('game_history', [])
    game_history.append(round_result)
    session['game_history'] = game_history
    
    return {
        'round_ended': True,
        'result': round_result,
        'scores': {
            'human_score': session.get('human_score', 0),
            'ai_score': sum(session.get('agent_scores', {}).values()),  # Total of all agent scores
            'human_wins': session.get('human_wins', 0),
            'ai_wins': session.get('ai_wins', 0),
            'agent_scores': session.get('agent_scores', {}),  # Individual agent scores
            'agent_wins': session.get('agent_wins', {}),  # Individual agent wins
            'winning_agent_id': None if not winning_agent else next((aid for aid, agent in current_agents.items() if agent == winning_agent), None)
        }
    }

def get_last_human_bid(bid_sequence):
    """Get the last human bid from sequence"""
    for bid in reversed(bid_sequence):
        if bid['bidder'] == 'human':
            return bid['amount']
    return 0

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get current game statistics"""
    game_history = session.get('game_history', [])
    total_rounds = session.get('game_round', 0)
    human_wins = session.get('human_wins', 0)
    
    human_win_rate = (human_wins / total_rounds * 100) if total_rounds > 0 else 0
    
    avg_bid_diff = 0
    if game_history:
        total_bid_diff = sum(abs(round_data['final_bid'] - round_data['actual_price']) 
                           for round_data in game_history)
        avg_bid_diff = total_bid_diff / len(game_history)
    
    return jsonify({
        'total_rounds': total_rounds,
         'human_win_rate': f"{human_win_rate:.1f}%",
        'avg_bid_diff': f"${avg_bid_diff:.2f}",
        'human_score': session.get('human_score', 0),
        'ai_score': sum(session.get('agent_scores', {}).values()),  # Total of all agent scores
        'human_wins': human_wins,
        'ai_wins': session.get('ai_wins', 0),
        'agent_scores': session.get('agent_scores', {}),  # Individual agent scores
        'agent_wins': session.get('agent_wins', {})  # Individual agent wins
    })

@app.route('/reset_game', methods=['POST'])
def reset_game():
    """Reset the entire game"""
    global current_agents, active_agents
    
    session.clear()
    for agent in current_agents.values():
        agent.reset_session()
    current_agents = {}
    active_agents = []
    
    # Initialize fresh session data
    session['game_initialized'] = True
    session['human_score'] = 0
    session['ai_score'] = 0
    session['human_wins'] = 0
    session['ai_wins'] = 0
    session['agent_wins'] = {}  # Reset individual agent wins
    session['agent_scores'] = {}  # Reset individual agent scores
    session['game_round'] = 0
    session['game_history'] = []
     
    return jsonify({
        'success': True,
        'message': 'Game reset successfully!'
    })

@app.route('/export_chart_data', methods=['GET'])
def export_chart_data():
    """Export chart data to CSV file"""
    try:
        # Get current game statistics
        game_history = session.get('game_history', [])
        agent_wins = session.get('agent_wins', {})
        agent_scores = session.get('agent_scores', {})
        human_wins = session.get('human_wins', 0)
        human_score = session.get('human_score', 0)
        
        # Create CSV data in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers for summary statistics
        writer.writerow(['=== GAME SUMMARY STATISTICS ==='])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Rounds', session.get('game_round', 0)])
        writer.writerow(['Human Wins', human_wins])
        writer.writerow(['Human Score', human_score])
        writer.writerow(['AI Total Wins', session.get('ai_wins', 0)])
        writer.writerow(['AI Total Score', sum(agent_scores.values())])
        
        # Add individual agent statistics
        writer.writerow([])
        writer.writerow(['=== INDIVIDUAL AGENT STATISTICS ==='])
        writer.writerow(['Agent ID', 'Wins', 'Score'])
        
        # Get all unique agents from game history and current session
        all_agents = set(agent_wins.keys()) | set(agent_scores.keys())
        for agent_id in sorted(all_agents):
            agent_win_count = agent_wins.get(agent_id, 0)
            agent_score_total = agent_scores.get(agent_id, 0)
            writer.writerow([agent_id, agent_win_count, agent_score_total])
        
        # Add detailed round history
        writer.writerow([])
        writer.writerow(['=== DETAILED ROUND HISTORY ==='])
        writer.writerow([
            'Round', 'Item Name', 'Market Price', 'Final Bid', 'Winner', 
            'Human Won', 'AI Won', 'Market Analysis', 'Value Assessment',
            'Price Difference %', 'Bid Sequence'
        ])
        
        # Write round data
        for round_data in game_history:
            market_analysis = round_data.get('market_analysis', {})
            bid_sequence_str = ' → '.join([
                f"{bid['bidder']}: ${bid['amount']:.2f}" 
                for bid in round_data.get('bid_sequence', [])
            ])
            
            writer.writerow([
                round_data.get('round', ''),
                round_data.get('item', ''),
                f"${round_data.get('actual_price', 0):.2f}",
                f"${round_data.get('final_bid', 0):.2f}",
                round_data.get('winner', ''),
                round_data.get('human_won', False),
                round_data.get('ai_won', False),
                market_analysis.get('analysis', 'No analysis') if market_analysis else 'No analysis',
                market_analysis.get('value_assessment', 'Unknown') if market_analysis else 'Unknown',
                f"{market_analysis.get('price_difference_percent', 0):.1f}%" if market_analysis else '0.0%',
                bid_sequence_str
            ])
        
        # Add win tracking data for chart visualization
        writer.writerow([])
        writer.writerow(['=== WIN TRACKING DATA FOR CHARTS ==='])
        writer.writerow(['Player/Agent', 'Win Count', 'Score', 'Type'])
        writer.writerow(['Human', human_wins, human_score, 'Human'])
        
        for agent_id in sorted(all_agents):
            writer.writerow([
                agent_id, 
                agent_wins.get(agent_id, 0), 
                agent_scores.get(agent_id, 0),
                'AI Agent'
            ])
        
        # Add timestamp
        writer.writerow([])
        writer.writerow(['Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Rounds Played', len(game_history)])
        
        # Prepare file for download
        output.seek(0)
        
        # Create a bytes buffer for the file
        mem_file = io.BytesIO()
        mem_file.write(output.getvalue().encode('utf-8'))
        mem_file.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bidding_game_data_{timestamp}.csv'
        
        return send_file(
            mem_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("🎯 Starting RL Bidding Game with Gemini AI Integration")
    if game.gemini_available:
        print("✅ Gemini API ready for dynamic software generation and bid analysis")
    else:
        print("⚠️  Gemini API not available - using fallback static items")
        print("   Set GEMINI_API_KEY environment variable to enable AI features")
    
    app.run(debug=True, host='0.0.0.0', port=5000)