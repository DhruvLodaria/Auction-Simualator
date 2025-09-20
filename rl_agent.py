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
    def __init__(self):
        # Q-table for state-action values
        self.q_table: Dict[str, float] = {}
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2  # exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        
        # Experience buffer for replay
        self.experience_buffer: List[Dict] = []
        self.buffer_size = 1000
        
        # Game statistics
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        
        # Bidding strategy parameters
        self.bid_history: List[float] = []
        self.opponent_history: List[float] = []
        self.price_history: List[float] = []
        
        # Current state tracking
        self.current_state: Optional[str] = None
        self.last_action: Optional[Union[float, str]] = None
        
        # Load any existing model
        self.load_model()
    
    def get_state(self, current_price: float, opponent_last_bid: float, 
                  game_round: int, my_last_bid: float = 0) -> str:
        """Get state representation for current game situation"""
        # Normalize values to create discrete states
        price_range = self._get_price_range(current_price)
        opponent_bid_range = self._get_bid_range(opponent_last_bid, current_price)
        my_bid_range = self._get_bid_range(my_last_bid, current_price)
        round_phase = min(game_round // 5, 4)  # 0-4
        
        # Create composite state
        state = f"{price_range}_{opponent_bid_range}_{my_bid_range}_{round_phase}"
        return state
    
    def _get_price_range(self, price: float) -> str:
        """Discretize price into ranges"""
        if price < 50:
            return 'low'
        elif price < 200:
            return 'medium'
        elif price < 500:
            return 'high'
        else:
            return 'premium'
    
    def _get_bid_range(self, bid: float, price: float) -> str:
        """Discretize bid relative to price"""
        if bid == 0:
            return 'none'
        
        ratio = bid / price
        if ratio < 0.8:
            return 'under'
        elif ratio < 1.1:
            return 'fair'
        elif ratio < 1.5:
            return 'over'
        else:
            return 'extreme'
    
    def get_actions(self, current_price: float, last_bid: float = 0) -> List[float]:
        """Get available actions (bid amounts relative to price and last bid)"""
        if last_bid == 0:
            # First bid must be below market value
            starting_bid_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
            return [ratio * current_price for ratio in starting_bid_ratios]
        else:
            # Subsequent bids are incremental increases
            base_increments = [
                1,    # +$1
                5,    # +$5
                10,   # +$10
                25,   # +$25
                50,   # +$50
                current_price * 0.05,  # +5% of market value
                current_price * 0.1,   # +10% of market value
                current_price * 0.15   # +15% of market value
            ]
            return [last_bid + increment for increment in base_increments]
    
    def select_action(self, state: str, current_price: float, last_bid: float = 0) -> float:
        """Select action using epsilon-greedy strategy"""
        actions = self.get_actions(current_price, last_bid)
        
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Exploitation: best known action
        best_action = actions[0]
        best_value = self.get_q_value(state, best_action)
        
        for action in actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action
    
    def get_q_value(self, state: str, action: Union[float, str]) -> float:
        """Get Q-value for state-action pair"""
        key = f"{state}_{self._action_to_key(action)}"
        return self.q_table.get(key, 0.0)
    
    def set_q_value(self, state: str, action: Union[float, str], value: float):
        """Set Q-value for state-action pair"""
        key = f"{state}_{self._action_to_key(action)}"
        self.q_table[key] = value
    
    def _action_to_key(self, action: Union[float, str]) -> str:
        """Convert action to discrete key"""
        if action == 'abandon':
            return 'abandon'
        return str(round(float(action) * 100))  # Round to cents
    
    def should_abandon_bid(self, current_price: float, opponent_last_bid: float, 
                          game_round: int) -> bool:
        """Determine if AI should abandon bidding based on stopping criteria"""
        # If this is the first bid and human hasn't bid yet, AI should make an opening bid
        if opponent_last_bid == 0:
            return False  # Always make opening bid
        
        # Criteria 1: Economic rationality - abandon if approaching or exceeding market value significantly
        if opponent_last_bid > current_price * 1.3:
            return True  # Definitely abandon if 30% over market value
        
        # Criteria 2: Market value threshold with probability
        if opponent_last_bid > current_price:
            overage = (opponent_last_bid - current_price) / current_price
            abandon_probability = min(0.8, overage * 2)  # Higher probability as overage increases
            if random.random() < abandon_probability:
                return True
        
        # Criteria 3: Risk assessment based on opponent behavior
        if len(self.opponent_history) >= 3:
            recent_bids = self.opponent_history[-3:]
            avg_bid = sum(recent_bids) / len(recent_bids)
            
            # If opponent is consistently aggressive and current bid is high
            if avg_bid > current_price * 0.8 and opponent_last_bid > avg_bid * 1.2:
                if random.random() < 0.3:  # 30% chance to abandon
                    return True
        
        # Criteria 4: Learning-based decision using Q-values
        state = self.get_state(current_price, opponent_last_bid, game_round, 0)
        abandon_reward = self.get_q_value(state, 'abandon')
        continue_actions = self.get_actions(current_price, opponent_last_bid)
        
        max_continue_value = float('-inf')
        for action in continue_actions:
            q_value = self.get_q_value(state, action)
            max_continue_value = max(max_continue_value, q_value)
        
        # If abandoning has higher expected value, do it
        if abandon_reward > max_continue_value + 10:  # Small bias towards continuing
            return True
        
        # Criteria 5: Progressive risk tolerance as bids approach market value
        bid_ratio = opponent_last_bid / current_price
        if bid_ratio > 0.85:
            abandon_probability = min(0.6, (bid_ratio - 0.85) * 4)  # Increases as we approach market value
            if random.random() < abandon_probability:
                return True
        
        # Criteria 6: Win/loss ratio consideration
        if self.total_games > 5:
            win_rate = self.wins / self.total_games
            # If AI is losing too much, be more conservative
            if win_rate < 0.3 and opponent_last_bid > current_price * 0.9:
                if random.random() < 0.4:
                    return True
        
        return False  # Continue bidding
    
    def update_q_value(self, state: str, action: Union[float, str], reward: float, 
                      next_state: Optional[str] = None):
        """Update Q-values based on experience"""
        current_q = self.get_q_value(state, action)
        
        # Find maximum Q-value for next state
        max_next_q = 0.0
        if next_state:
            next_actions = self.get_actions(100, 0)  # Use dummy values for next state actions
            next_actions_str = ['abandon']  # Include abandon as an action
            
            for next_action in next_actions + next_actions_str:
                next_q = self.get_q_value(next_state, next_action)
                max_next_q = max(max_next_q, next_q)
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.set_q_value(state, action, new_q)
    
    def calculate_reward(self, my_bid: float, opponent_bid: float, actual_price: float, 
                        won: bool, abandoned: bool = False) -> float:
        """Calculate reward based on game outcome"""
        reward = 0.0
        
        if abandoned:
            # Reward for smart abandoning
            if opponent_bid > actual_price * 1.5:
                reward += 30  # Good decision to abandon
            elif opponent_bid < actual_price * 1.1:
                reward -= 40  # Bad decision to abandon early
            
            # Small penalty for giving up
            reward -= 10
        elif won:
            # Base reward for winning
            reward += 100
            
            # Bonus for efficient bidding (not overbidding too much)
            overbid_amount = my_bid - actual_price
            if overbid_amount > 0:
                efficiency = max(0, 1 - (overbid_amount / actual_price))
                reward += efficiency * 50
            
            # Bonus for beating opponent by smaller margin (strategic play)
            margin = my_bid - opponent_bid
            if margin > 0 and margin < actual_price * 0.1:
                reward += 25
        else:
            # Penalty for losing
            reward -= 50
            
            # Smaller penalty if bid was close
            if my_bid > 0:
                diff = abs(my_bid - opponent_bid)
                if diff < actual_price * 0.05:
                    reward += 20  # Reduce penalty for close bids
        
        # Penalty for extremely high bids (risk management)
        if my_bid > actual_price * 2:
            reward -= 30
        
        return reward
    
    def train_from_experience(self):
        """Train the agent with experience replay"""
        if len(self.experience_buffer) < 10:
            return
        
        # Sample random experiences for training
        batch_size = min(5, len(self.experience_buffer))
        for _ in range(batch_size):
            experience = random.choice(self.experience_buffer)
            
            self.update_q_value(
                experience['state'],
                experience['action'],
                experience['reward'],
                experience.get('next_state')
            )
    
    def learn(self, state: str, action: Union[float, str], reward: float, 
             next_state: Optional[str] = None):
        """Learn from game result"""
        # Add to experience buffer
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        }
        self.experience_buffer.append(experience)
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
        
        # Update Q-value
        self.update_q_value(state, action, reward, next_state)
        
        # Train from past experiences
        self.train_from_experience()
        
        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def make_bid(self, current_price: float, game_round: int, 
                opponent_last_bid: float = 0) -> Dict:
        """Make a bid decision or abandon"""
        my_last_bid = self.bid_history[-1] if self.bid_history else 0
        state = self.get_state(current_price, opponent_last_bid, game_round, my_last_bid)
        
        # Store current state for learning
        self.current_state = state
        
        # Check if AI should abandon based on stopping criteria
        should_abandon = self.should_abandon_bid(current_price, opponent_last_bid, game_round)
        if should_abandon:
            self.last_action = 'abandon'
            return {'bid': None, 'state': state, 'abandoned': True}
        
        bid = self.select_action(state, current_price, opponent_last_bid)
        
        # Add some adaptive behavior based on opponent patterns
        bid = self._adapt_bid_to_opponent(bid, current_price, opponent_last_bid)
        
        # Ensure bid follows game rules
        if opponent_last_bid == 0:
            # First bid must be below market value
            bid = min(bid, current_price * 0.95)
            bid = max(bid, current_price * 0.3)  # At least 30% of market value
        else:
            # Must bid higher than opponent's last bid
            bid = max(bid, opponent_last_bid + 1)
        
        # Cap at reasonable maximum
        bid = min(bid, current_price * 2)
        
        # Round to 2 decimal places
        bid = round(bid, 2)
        
        self.bid_history.append(bid)
        self.last_action = bid
        
        return {'bid': bid, 'state': state, 'abandoned': False}
    
    def _adapt_bid_to_opponent(self, base_bid: float, current_price: float, 
                              opponent_last_bid: float) -> float:
        """Adapt bid based on opponent behavior patterns"""
        if len(self.opponent_history) < 3:
            return base_bid
        
        # Analyze opponent's recent bidding pattern
        recent_bids = self.opponent_history[-3:]
        avg_opponent_bid = sum(recent_bids) / len(recent_bids)
        opponent_trend = self._calculate_trend(recent_bids)
        
        adjusted_bid = base_bid
        
        # If opponent is consistently aggressive, be more conservative
        if avg_opponent_bid > current_price * 1.3:
            adjusted_bid *= 0.9
        
        # If opponent is trending upward, increase bid slightly
        if opponent_trend > 0.1:
            adjusted_bid *= 1.05
        
        # If opponent just made a very low bid, this might be an opportunity
        if opponent_last_bid < current_price * 0.9:
            adjusted_bid *= 1.1
        
        return adjusted_bid
    
    def _calculate_trend(self, bids: List[float]) -> float:
        """Calculate trend in recent bids"""
        if len(bids) < 2:
            return 0.0
        
        trend = 0.0
        for i in range(1, len(bids)):
            if bids[i-1] > 0:  # Avoid division by zero
                trend += (bids[i] - bids[i-1]) / bids[i-1]
        
        return trend / (len(bids) - 1)
    
    def record_opponent_bid(self, bid: float):
        """Record opponent's bid for learning"""
        self.opponent_history.append(bid)
        
        # Keep history manageable
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
            'total_games': self.total_games,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / self.total_games if self.total_games > 0 else 0,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }
    
    def reset_session(self):
        """Reset for new game session"""
        self.bid_history = []
        self.opponent_history = []
        self.price_history = []
        self.current_state = None
        self.last_action = None
    
    def save_model(self, filename: str = 'rl_bidding_agent.pkl'):
        """Save model state to file"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'stats': self.get_stats(),
            'experience_buffer': self.experience_buffer[-100:]  # Save only recent experiences
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"RL Agent model saved to {filename}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    def load_model(self, filename: str = 'rl_bidding_agent.pkl'):
        """Load model state from file"""
        if not os.path.exists(filename):
            print(f"No saved model found at {filename}")
            return
        
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data.get('q_table', {})
            self.epsilon = model_data.get('epsilon', self.epsilon)
            
            if 'stats' in model_data:
                stats = model_data['stats']
                self.total_games = stats.get('total_games', 0)
                self.wins = stats.get('wins', 0)
                self.losses = stats.get('losses', 0)
            
            if 'experience_buffer' in model_data:
                self.experience_buffer = model_data['experience_buffer']
            
            print(f"RL Agent model loaded successfully from {filename}")
        except Exception as e:
            print(f"Failed to load saved model: {e}")