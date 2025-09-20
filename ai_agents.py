#!/usr/bin/env python3
"""
Multiple AI Agents for Bidding Game
Different AI agents with various bidding strategies.
"""

import random
import math
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

# Gemini API integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. GeminiAgent will not be available.")

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, strategy: str):
        self.name = name
        self.strategy = strategy
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.bid_history: List[float] = []
        self.opponent_history: List[float] = []
        self.current_state: Optional[str] = None
        self.last_action: Optional[Union[float, str]] = None
        self.estimated_value: float = 0.0  # Estimated value of current item
    
    @abstractmethod
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        """Make a bid decision or abandon"""
        pass
    
    @abstractmethod
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        """Determine if agent should abandon bidding"""
        pass
    
    def record_opponent_bid(self, bid: float):
        """Record opponent's bid for learning"""
        self.opponent_history.append(bid)
        if len(self.opponent_history) > 20:
            self.opponent_history.pop(0)
    
    def record_game_outcome(self, won: bool):
        """Record game outcome for statistics"""
        self.total_games += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
    
    def get_stats(self) -> Dict:
        """Get agent performance statistics"""
        return {
            'name': self.name,
            'strategy': self.strategy,
            'total_games': self.total_games,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / self.total_games if self.total_games > 0 else 0
        }
    
    def reset_session(self):
        """Reset for new game session"""
        self.bid_history = []
        self.opponent_history = []
        self.current_state = None
        self.last_action = None
        self.estimated_value = 0.0
    
    def estimate_item_value(self, market_price: float) -> float:
        """Estimate the true value of an item based on market price"""
        # Add some randomness to simulate different value perceptions
        variation = random.uniform(0.85, 1.15)  # ±15% variation
        self.estimated_value = market_price * variation
        return self.estimated_value

class TruthfulAgent(BaseAgent):
    """Truthful Bidding Agent - Bids exactly estimated value"""
    
    def __init__(self):
        super().__init__("Truthful Bidder", "Value-based")
    
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        # Estimate item value if not done yet
        if self.estimated_value == 0.0:
            self.estimate_item_value(current_price)
        
        state = f"truthful_{current_price}_{opponent_last_bid}"
        self.current_state = state
        
        if self.should_abandon_bid(current_price, opponent_last_bid, game_round):
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True}
        
        # Bid close to estimated value, but must be higher than last bid
        if opponent_last_bid == 0:
            # Opening bid - start conservatively at estimated value minus buffer
            bid = min(self.estimated_value * 0.9, current_price * 0.85)
        else:
            # Subsequent bids - stay close to estimated value
            bid = min(self.estimated_value, opponent_last_bid + current_price * 0.05)
        
        bid = max(bid, opponent_last_bid + 1)
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        return {'bid': bid, 'state': state, 'abandoned': False}
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        # Abandon if opponent bid exceeds our estimated value
        return opponent_last_bid > self.estimated_value

class SnipingAgent(BaseAgent):
    """Sniping Agent - Waits until the end to bid aggressively"""
    
    def __init__(self):
        super().__init__("Sniper", "Last-moment bidding")
        self.snipe_threshold = 3  # Start aggressive bidding after 3 rounds
    
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        if self.estimated_value == 0.0:
            self.estimate_item_value(current_price)
        
        state = f"sniper_{game_round}_{opponent_last_bid}"
        self.current_state = state
        
        if self.should_abandon_bid(current_price, opponent_last_bid, game_round):
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True}
        
        # Early rounds: bid minimally
        if len(self.bid_history) < self.snipe_threshold:
            if opponent_last_bid == 0:
                bid = current_price * 0.4  # Very low opening bid
            else:
                bid = opponent_last_bid + 1  # Minimal increment
        else:
            # Late rounds: aggressive "sniping" bid
            aggressive_bid = min(self.estimated_value * 0.95, current_price * 1.1)
            bid = max(aggressive_bid, opponent_last_bid + current_price * 0.1)
        
        bid = max(bid, opponent_last_bid + 1)
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        return {'bid': bid, 'state': state, 'abandoned': False}
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        # More willing to abandon early, less willing later
        if len(self.bid_history) < self.snipe_threshold:
            return opponent_last_bid > current_price * 0.8
        else:
            return opponent_last_bid > self.estimated_value * 1.1

class IncrementalAgent(BaseAgent):
    """Incremental Bidding Agent - Gradual step-by-step increases"""
    
    def __init__(self):
        super().__init__("Incremental Bidder", "Step-by-step increases")
        self.increment_size = 0.05  # 5% increments
    
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        if self.estimated_value == 0.0:
            self.estimate_item_value(current_price)
        
        state = f"incremental_{len(self.bid_history)}_{opponent_last_bid}"
        self.current_state = state
        
        if self.should_abandon_bid(current_price, opponent_last_bid, game_round):
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True}
        
        if opponent_last_bid == 0:
            # Start with a low bid
            bid = current_price * 0.5
        else:
            # Incremental increase based on current price
            increment = current_price * self.increment_size
            bid = opponent_last_bid + max(increment, 5)  # At least $5 increase
        
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        return {'bid': bid, 'state': state, 'abandoned': False}
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        # Abandon if we've made too many increments or price is too high
        return (len(self.bid_history) > 6 or 
                opponent_last_bid > self.estimated_value or
                opponent_last_bid > current_price * 1.2)

class JumpBiddingAgent(BaseAgent):
    """Jump Bidding Agent - Large jumps to intimidate opponents"""
    
    def __init__(self):
        super().__init__("Jump Bidder", "Aggressive large jumps")
        self.jump_multiplier = 1.3  # 30% jumps
    
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        if self.estimated_value == 0.0:
            self.estimate_item_value(current_price)
        
        state = f"jump_{opponent_last_bid}_{game_round}"
        self.current_state = state
        
        if self.should_abandon_bid(current_price, opponent_last_bid, game_round):
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True}
        
        if opponent_last_bid == 0:
            # Strong opening bid
            bid = current_price * 0.7
        else:
            # Big jump to intimidate
            jump_bid = opponent_last_bid * self.jump_multiplier
            bid = min(jump_bid, self.estimated_value * 0.9)
        
        bid = max(bid, opponent_last_bid + 1)
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        return {'bid': bid, 'state': state, 'abandoned': False}
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        # Abandon if opponent doesn't seem intimidated or price too high
        return (opponent_last_bid > self.estimated_value * 0.9 or
                opponent_last_bid > current_price * 1.15)

class ShadingAgent(BaseAgent):
    """Shading Agent - Underbids to maximize profit margin"""
    
    def __init__(self):
        super().__init__("Shader", "Underbidding for profit")
        self.shade_factor = 0.85  # Bid 85% of estimated value
    
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        if self.estimated_value == 0.0:
            self.estimate_item_value(current_price)
        
        state = f"shading_{opponent_last_bid}_{self.estimated_value}"
        self.current_state = state
        
        if self.should_abandon_bid(current_price, opponent_last_bid, game_round):
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True}
        
        # Always bid below estimated value to ensure profit
        max_bid = self.estimated_value * self.shade_factor
        
        if opponent_last_bid == 0:
            bid = min(max_bid, current_price * 0.6)
        else:
            # Only increment if still below shaded value
            if opponent_last_bid + 10 <= max_bid:
                bid = opponent_last_bid + random.uniform(5, 15)
            else:
                # Can't maintain profit margin, consider abandoning
                bid = max_bid
        
        bid = max(bid, opponent_last_bid + 1)
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        return {'bid': bid, 'state': state, 'abandoned': False}
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        # Abandon if can't maintain profit margin
        return opponent_last_bid >= self.estimated_value * self.shade_factor

class CollusiveAgent(BaseAgent):
    """Collusive Bidding Agent - Coordinates with other AI agents to keep prices low"""
    
    def __init__(self):
        super().__init__("Collusive Bidder", "Coordinated low bidding")
        self.collusion_partners = []  # List of other colluding agents
        self.turn_order = []  # Predetermined turn order for winning
        self.current_turn_index = 0
        self.my_turn_to_win = False
        self.collusion_active = True
        self.max_collusive_bid_ratio = 0.6  # Never bid above 60% of market value when colluding
        self.coordination_history = []  # Track coordination attempts
    
    def set_collusion_partners(self, partners):
        """Set other agents to collude with"""
        self.collusion_partners = partners
        # Establish turn order based on agent names for consistency
        all_agents = [self] + partners
        self.turn_order = sorted(all_agents, key=lambda x: x.name)
        self.update_turn_assignment()
    
    def update_turn_assignment(self):
        """Determine if it's this agent's turn to win based on game round"""
        if not self.turn_order:
            self.my_turn_to_win = False
            return
        
        # Use total games played to determine whose turn it is
        turn_index = self.total_games % len(self.turn_order)
        self.my_turn_to_win = (self.turn_order[turn_index] == self)
    
    def is_collusion_viable(self, current_price: float, opponent_last_bid: float) -> bool:
        """Determine if collusion is still viable in current situation"""
        # Collusion breaks down if:
        # 1. Human bids are too aggressive (above our max collusive threshold)
        # 2. Market value is too low to share
        # 3. Too many rounds have passed
        
        if opponent_last_bid > self.estimated_value * self.max_collusive_bid_ratio:
            self.collusion_active = False
            return False
        
        if len(self.bid_history) > 4:  # Long bidding wars break collusion
            self.collusion_active = False
            return False
        
        return self.collusion_active
    
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        if self.estimated_value == 0.0:
            self.estimate_item_value(current_price)
        
        # Update turn assignment
        self.update_turn_assignment()
        
        state = f"collusive_{self.my_turn_to_win}_{opponent_last_bid}_{self.collusion_active}"
        self.current_state = state
        
        if self.should_abandon_bid(current_price, opponent_last_bid, game_round):
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True}
        
        # Check if collusion is still viable
        collusion_viable = self.is_collusion_viable(current_price, opponent_last_bid)
        
        if collusion_viable and self.my_turn_to_win:
            # It's my turn to win - bid more aggressively but still reasonably
            if opponent_last_bid == 0:
                bid = min(self.estimated_value * 0.7, current_price * 0.8)
            else:
                # Aggressive bid to secure win, but not above market value
                bid = min(
                    opponent_last_bid + current_price * 0.15,
                    self.estimated_value * 0.9
                )
        elif collusion_viable and not self.my_turn_to_win:
            # Not my turn - bid minimally to keep prices low
            if opponent_last_bid == 0:
                bid = current_price * 0.3  # Very low opening bid
            else:
                # Minimal increment to avoid suspicion but keep prices low
                bid = opponent_last_bid + random.uniform(1, 5)
                # Don't bid above our collusive threshold
                bid = min(bid, self.estimated_value * self.max_collusive_bid_ratio)
        else:
            # Collusion broken - revert to normal competitive bidding
            if opponent_last_bid == 0:
                bid = current_price * 0.6
            else:
                bid = min(
                    opponent_last_bid + current_price * 0.1,
                    self.estimated_value * 0.85
                )
        
        bid = max(bid, opponent_last_bid + 1)
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        # Record coordination attempt
        self.coordination_history.append({
            'round': game_round,
            'my_turn': self.my_turn_to_win,
            'collusion_active': collusion_viable,
            'bid': bid,
            'opponent_bid': opponent_last_bid
        })
        
        return {'bid': bid, 'state': state, 'abandoned': False}
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        # Collusive agents are more willing to abandon when it's not their turn
        if self.collusion_active and not self.my_turn_to_win:
            # Abandon earlier when it's not our turn to keep prices low
            return opponent_last_bid > self.estimated_value * self.max_collusive_bid_ratio
        else:
            # When it's our turn or collusion is broken, normal abandonment criteria
            return opponent_last_bid > self.estimated_value * 0.95
    
    def record_game_outcome(self, won: bool):
        """Override to track collusion success"""
        super().record_game_outcome(won)
        
        # Analyze if collusion strategy worked
        if self.coordination_history:
            avg_bid = sum(entry['bid'] for entry in self.coordination_history) / len(self.coordination_history)
            collusion_success = avg_bid < self.estimated_value * 0.7 if self.estimated_value > 0 else False
            
            # Adjust collusion parameters based on success
            if collusion_success and won:
                # Successful collusion - maintain strategy
                pass
            elif not collusion_success:
                # Collusion failed - be more aggressive next time
                self.max_collusive_bid_ratio = min(0.8, self.max_collusive_bid_ratio + 0.1)
    
    def reset_session(self):
        """Reset for new game session"""
        super().reset_session()
        self.coordination_history = []
        self.collusion_active = True
        self.current_turn_index = 0

class GeminiAgent(BaseAgent):
    """Gemini-Powered AI Agent - Uses Google's Gemini API for intelligent bidding"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Gemini AI", "AI-powered strategic bidding")
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package is required for GeminiAgent")
        
        # Initialize Gemini API
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)  # type: ignore
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # type: ignore
        
        # Bidding context and history
        self.bidding_context: List[Dict] = []
        self.market_analysis: Dict = {}
        self.risk_tolerance = 0.7  # Default risk tolerance (0-1)
        
    def analyze_market_context(self, current_price: float, game_round: int, opponent_last_bid: float) -> str:
        """Generate market context for Gemini analysis"""
        context = {
            'market_value': current_price,
            'estimated_value': self.estimated_value,
            'game_round': game_round,
            'opponent_last_bid': opponent_last_bid,
            'my_bid_history': self.bid_history[-5:],  # Last 5 bids
            'opponent_history': self.opponent_history[-5:],  # Last 5 opponent bids
            'total_games': self.total_games,
            'win_rate': self.wins / self.total_games if self.total_games > 0 else 0,
            'risk_tolerance': self.risk_tolerance
        }
        return json.dumps(context, indent=2)
    
    def make_bid(self, current_price: float, game_round: int, opponent_last_bid: float = 0) -> Dict:
        if self.estimated_value == 0.0:
            self.estimate_item_value(current_price)
        
        state = f"gemini_{current_price}_{opponent_last_bid}_{game_round}"
        self.current_state = state
        
        try:
            # Prepare context for Gemini
            market_context = self.analyze_market_context(current_price, game_round, opponent_last_bid)
            
            # Create prompt for Gemini
            prompt = f"""
You are an expert auction bidder in a competitive bidding game. Analyze the following auction scenario and make a strategic bidding decision.

**Market Context:**
{market_context}

**Game Rules:**
1. You're bidding on a software item with market value ${current_price:.2f}
2. Your estimated value is ${self.estimated_value:.2f}
3. Current highest bid is ${opponent_last_bid:.2f}
4. You must bid higher than the current bid or abandon
5. Goal: Win items at prices below their market value to maximize profit

**Your Options:**
1. ABANDON - Stop bidding (if price is too high or unprofitable)
2. BID - Make a specific bid amount (must be > ${opponent_last_bid:.2f})

**Decision Factors to Consider:**
- Profitability: Bid price vs estimated value
- Opponent behavior patterns
- Risk vs reward analysis
- Market value awareness
- Your historical performance

**Response Format:**
Provide your decision in this exact JSON format:
{{
  "action": "BID" or "ABANDON",
  "bid_amount": number (only if action is BID),
  "reasoning": "Brief explanation of your strategy",
  "confidence": number between 0-1
}}

Make your decision:"""
            
            # Get Gemini's response
            response = self.model.generate_content(prompt)
            
            # Parse the response
            try:
                # Extract JSON from response
                response_text = response.text.strip()
                if '```json' in response_text:
                    json_start = response_text.find('{', response_text.find('```json'))
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
                elif '{' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
                
                decision = json.loads(response_text)
                
                # Validate decision
                action = decision.get('action', '').upper()
                
                if action == 'ABANDON':
                    self.last_action = 'abandon'
                    return {'bid': None, 'state': state, 'abandoned': True, 'reasoning': decision.get('reasoning', 'Strategic abandonment')}
                
                elif action == 'BID':
                    bid_amount = float(decision.get('bid_amount', 0))
                    
                    # Validate bid amount
                    if bid_amount <= opponent_last_bid:
                        bid_amount = opponent_last_bid + random.uniform(5, 15)
                    
                    # Safety check: don't bid above 150% of estimated value
                    max_safe_bid = self.estimated_value * 1.5
                    if bid_amount > max_safe_bid:
                        bid_amount = max_safe_bid
                    
                    # Final validation
                    if bid_amount <= opponent_last_bid or bid_amount > current_price * 2:
                        # Fallback to safe bid
                        bid_amount = min(opponent_last_bid + 10, self.estimated_value * 0.9)
                    
                    bid_amount = round(bid_amount, 2)
                    
                    self.bid_history.append(bid_amount)
                    self.last_action = bid_amount
                    
                    # Store reasoning for analysis
                    self.bidding_context.append({
                        'round': game_round,
                        'bid': bid_amount,
                        'reasoning': decision.get('reasoning', ''),
                        'confidence': decision.get('confidence', 0.5)
                    })
                    
                    return {
                        'bid': bid_amount, 
                        'state': state, 
                        'abandoned': False,
                        'reasoning': decision.get('reasoning', ''),
                        'confidence': decision.get('confidence', 0.5)
                    }
                else:
                    # Invalid action, fallback
                    return self._fallback_bid(current_price, opponent_last_bid, state)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing Gemini response: {e}")
                print(f"Raw response: {response.text}")
                # Fallback to simple strategy
                return self._fallback_bid(current_price, opponent_last_bid, state)
                
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Fallback to simple strategy
            return self._fallback_bid(current_price, opponent_last_bid, state)
    
    def _fallback_bid(self, current_price: float, opponent_last_bid: float, state: str) -> Dict:
        """Fallback bidding strategy when Gemini API fails"""
        if self.should_abandon_bid(current_price, opponent_last_bid, 1):
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True, 'reasoning': 'Fallback: Price too high'}
        
        # Simple competitive bid
        if opponent_last_bid == 0:
            bid = min(self.estimated_value * 0.8, current_price * 0.7)
        else:
            bid = min(opponent_last_bid + 15, self.estimated_value * 0.9)
        
        bid = max(bid, opponent_last_bid + 1)
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        return {'bid': bid, 'state': state, 'abandoned': False, 'reasoning': 'Fallback strategy'}
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, game_round: int) -> bool:
        # Conservative abandonment - let Gemini make most decisions
        return opponent_last_bid > self.estimated_value * 1.3
    
    def record_game_outcome(self, won: bool):
        """Override to update risk tolerance based on performance"""
        super().record_game_outcome(won)
        
        # Adjust risk tolerance based on recent performance
        if self.total_games >= 5:
            recent_win_rate = self.wins / self.total_games
            if recent_win_rate > 0.6:
                self.risk_tolerance = min(0.9, self.risk_tolerance + 0.05)  # Increase risk
            elif recent_win_rate < 0.3:
                self.risk_tolerance = max(0.3, self.risk_tolerance - 0.05)  # Decrease risk
    
    def reset_session(self):
        """Reset for new game session"""
        super().reset_session()
        self.bidding_context = []
        self.market_analysis = {}

def create_agent_by_name(agent_name: str) -> BaseAgent:
    """Factory function to create agents by name"""
    agents = {
        'truthful': TruthfulAgent,
        'sniper': SnipingAgent,
        'incremental': IncrementalAgent,
        'jump': JumpBiddingAgent,
        'shader': ShadingAgent,
        'collusive': CollusiveAgent,
        'gemini': lambda: GeminiAgent() if GEMINI_AVAILABLE else TruthfulAgent()
    }
    
    if agent_name.lower() in agents:
        return agents[agent_name.lower()]()
    else:
        return TruthfulAgent()  # Default fallback

def get_all_available_agents() -> List[Dict]:
    """Get list of all available agents with their info"""
    base_agents = [
        {'id': 'truthful', 'name': 'Truthful Bidder', 'strategy': 'Value-based bidding'},
        {'id': 'sniper', 'name': 'Sniper', 'strategy': 'Last-moment aggressive bidding'},
        {'id': 'incremental', 'name': 'Incremental Bidder', 'strategy': 'Step-by-step increases'},
        {'id': 'jump', 'name': 'Jump Bidder', 'strategy': 'Large intimidating jumps'},
        {'id': 'shader', 'name': 'Shader', 'strategy': 'Underbidding for profit margin'},
        {'id': 'collusive', 'name': 'Collusive Bidder', 'strategy': 'Coordinated low bidding'}
    ]
    
    # Add Gemini agent if available
    if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
        base_agents.append({
            'id': 'gemini', 
            'name': 'Gemini AI', 
            'strategy': 'AI-powered strategic bidding'
        })
    
    return base_agents