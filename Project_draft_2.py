!pip install gymnasium
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import json
from flask import Flask, render_template_string, jsonify
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from threading import Thread, Lock
import time



# --- Auction Environment Class ---
# This class defines the core logic of the English auction.
# It inherits from gymnasium.Env to be compatible with standard RL libraries.
class EnglishAuctionEnv(gym.Env):
    def __init__(self, num_agents=3, min_val=100, max_val=200, min_increment=5, max_bid_cap=300):
        super(EnglishAuctionEnv, self).__init__()
        
        # Action Space: Each agent can choose to bid or not bid.
        # 0: Don't bid, 1: Bid
        self.action_space = spaces.MultiBinary(num_agents)
        
        # Observation Space: An agent observes the current price and its own valuation.
        # For simplicity in a multi-agent setting, we combine these.
        self.observation_space = spaces.Dict({
            "current_bid": spaces.Box(low=0, high=max_bid_cap, shape=(1,), dtype=np.float32),
            "agents_active": spaces.MultiBinary(num_agents),
            "valuations": spaces.Box(low=min_val, high=max_val, shape=(num_agents,), dtype=np.float32)
        })

        self.num_agents = num_agents
        self.min_val = min_val
        self.max_val = max_val
        self.min_increment = min_increment
        self.max_bid_cap = max_bid_cap
        self.auction_in_progress = False

        # State variables
        self.valuations = np.zeros(self.num_agents, dtype=np.float32)
        self.active_agents = np.ones(self.num_agents, dtype=np.int8)
        self.current_bid = 0.0
        self.last_bidder = -1  # Index of the last bidder
        self.winning_agent = -1
        self.bid_history = []
        self.turn_history = []

    def _get_obs(self):
        """Returns the current observation."""
        return {
            "current_bid": np.array([self.current_bid], dtype=np.float32),
            "agents_active": self.active_agents.copy(),
            "valuations": self.valuations.copy()
        }

    def _get_info(self):
        """Returns auxiliary information about the environment state."""
        return {
            "last_bidder": self.last_bidder,
            "bid_history": self.bid_history,
            "turn_history": self.turn_history,
            "winning_agent": self.winning_agent
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all state variables for a new auction
        self.valuations = self.np_random.uniform(self.min_val, self.max_val, self.num_agents).astype(np.float32)
        self.active_agents = np.ones(self.num_agents, dtype=np.int8)
        self.current_bid = 0.0
        self.last_bidder = -1
        self.winning_agent = -1
        self.bid_history = []
        self.turn_history = []
        self.auction_in_progress = True
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, actions):
        """
        Takes actions from all agents and updates the environment.
        Actions is a list/array of 0s and 1s, one for each agent.
        """
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        done = False
        
        # If auction is already over, no more steps
        if not self.auction_in_progress:
            return self._get_obs(), rewards, done, False, self._get_info()

        valid_bids = []
        
        for i in range(self.num_agents):
            if self.active_agents[i] == 1:
                # Agent's decision: to bid or not
                bid_action = actions[i]
                
                # Check if agent's valuation is too low to continue
                if self.valuations[i] < self.current_bid + self.min_increment:
                    bid_action = 0  # Force agent to drop out if bid is irrational
                    
                if bid_action == 1:
                    new_bid = self.current_bid + self.min_increment
                    if new_bid <= self.max_bid_cap:
                        valid_bids.append((i, new_bid))
                        self.current_bid = new_bid
                        self.last_bidder = i
                        # Reward for making a valid bid
                        rewards[i] += 1
                else:
                    self.active_agents[i] = 0 # Agent drops out
                    rewards[i] -= 1 # Small penalty for dropping out

        # Log the current turn's actions
        self.turn_history.append({
            "active_agents": list(self.active_agents),
            "current_bid": self.current_bid,
            "last_bidder": self.last_bidder,
            "valid_bids_this_turn": valid_bids
        })
        
        # End condition: Only one or zero agents left active
        if np.sum(self.active_agents) <= 1:
            done = True
            self.auction_in_progress = False
            
            # Determine the winner
            if np.sum(self.active_agents) == 1:
                self.winning_agent = np.argmax(self.active_agents)
                winning_price = self.current_bid
                value_gained = self.valuations[self.winning_agent]
                
                # Reward the winner based on utility
                reward_for_winner = value_gained - winning_price
                rewards[self.winning_agent] += reward_for_winner
                
                self.bid_history.append({
                    "bidder": self.winning_agent,
                    "bid_amount": winning_price
                })
            else:
                self.winning_agent = -1 # No winner if everyone drops out

        # Check if no agent has made a valid bid and at least one is active
        if len(valid_bids) == 0 and np.sum(self.active_agents) > 0:
            done = True
            self.auction_in_progress = False
            self.winning_agent = self.last_bidder
            if self.winning_agent != -1:
                rewards[self.winning_agent] += self.valuations[self.winning_agent] - self.current_bid

        # Check for max bid cap
        if self.current_bid >= self.max_bid_cap:
            done = True
            self.auction_in_progress = False
            self.winning_agent = self.last_bidder
            if self.winning_agent != -1:
                rewards[self.winning_agent] += self.valuations[self.winning_agent] - self.current_bid
        
        observation = self._get_obs()
        info = self._get_info()
        
        # Truncated is always false in this simplified environment
        return observation, rewards, done, False, info

    def render(self, mode="console"):
        """Prints the current state to the console."""
        print(f"Current Bid: ${self.current_bid:.2f}")
        print(f"Last Bidder: Agent {self.last_bidder + 1 if self.last_bidder != -1 else 'N/A'}")
        print(f"Active Agents: {[i + 1 for i, status in enumerate(self.active_agents) if status == 1]}")
        if not self.auction_in_progress:
            print(f"Auction Ended. Winner: Agent {self.winning_agent + 1 if self.winning_agent != -1 else 'N/A'}")
            print("-" * 20)

# --- Flask Web Server and RL Training ---

app = Flask(__name__)
# Global variables to share state between Flask and the simulation thread
auction_data = {
    "current_bid": 0.0,
    "last_bidder": -1,
    "winning_agent": -1,
    "history": [],
    "agents": []
}
data_lock = Lock()

def run_simulation(env):
    """
    This function runs the RL training loop and simulation in a separate thread.
    """
    global auction_data

    # Wrap the environment to be compatible with single-agent RL algorithms.
    # We will treat the joint actions and observations as a single vector.
    # This is a common, albeit simplified, approach for multi-agent problems with single-agent libraries.
    class MultiAgentWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            # Flatten observation and action spaces
            obs_space_size = sum(s.shape[0] if isinstance(s, spaces.Box) else s.n for s in self.env.observation_space.values())
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32)
            self.action_space = self.env.action_space
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._flatten_obs(obs), info
        
        def step(self, actions):
            obs, rewards, done, truncated, info = self.env.step(actions)
            # In a multi-agent scenario, we might want a single reward signal.
            # Here we sum the rewards, a simple approach.
            total_reward = np.sum(rewards)
            return self._flatten_obs(obs), total_reward, done, truncated, info
            
        def _flatten_obs(self, obs):
            # Flatten the dictionary observation into a single vector
            flat_obs = np.concatenate([v.flatten() for v in obs.values()])
            return flat_obs

    print("Starting RL training...")
    
    # Create the environment for training
    env_train = MultiAgentWrapper(env)
    
    # Define and train the PPO model
    model = PPO("MlpPolicy", env_train, verbose=1, learning_rate=0.0003, n_steps=2048)
    model.learn(total_timesteps=100000)
    print("Training finished!")
    
    print("Running simulation with trained agents...")
    
    # Run a continuous loop of auctions to display on the web interface
    while True:
        # Reset the environment for a new round
        obs_dict, info = env.reset()
        
        # Initial data update for the web interface
        with data_lock:
            auction_data["current_bid"] = obs_dict["current_bid"][0]
            auction_data["last_bidder"] = -1
            auction_data["winning_agent"] = -1
            auction_data["history"] = []
            auction_data["agents"] = [{"valuation": val, "is_active": True} for val in obs_dict["valuations"]]
        
        # Run the auction
        done = False
        while not done:
            # Get the flat observation for the trained model
            flat_obs = env_train._flatten_obs(obs_dict)
            actions, _states = model.predict(flat_obs, deterministic=True)
            
            # Step the environment with the agents' actions
            obs_dict, rewards, done, truncated, info = env.step(actions)
            
            # Update data for the web interface
            with data_lock:
                auction_data["current_bid"] = obs_dict["current_bid"][0]
                auction_data["last_bidder"] = info["last_bidder"]
                auction_data["winning_agent"] = info["winning_agent"]
                auction_data["history"] = info["bid_history"]
                for i in range(len(auction_data["agents"])):
                    auction_data["agents"][i]["is_active"] = bool(obs_dict["agents_active"][i])

            env.render()
            time.sleep(2) # Pause for 2 seconds to see the bids
        
        print("Auction complete. Starting a new one...")
        time.sleep(5) # Wait before starting the next auction

@app.route("/")
def index():
    """Serves the main HTML page for the auction visualization."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/data")
def get_data():
    """Provides real-time auction data as a JSON object."""
    with data_lock:
        return jsonify(auction_data)

if __name__ == "__main__":
    # Initialize the environment
    NUM_AGENTS = 3
    env = EnglishAuctionEnv(num_agents=NUM_AGENTS, min_val=100, max_val=200, min_increment=5, max_bid_cap=300)
    
    # Start the simulation thread
    simulation_thread = Thread(target=run_simulation, args=(env,))
    simulation_thread.daemon = True
    simulation_thread.start()
    
    # Start the Flask web server
    app.run(debug=True, use_reloader=False)
