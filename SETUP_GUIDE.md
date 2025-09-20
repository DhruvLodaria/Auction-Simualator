# RL Bidding Game - Python Implementation Guide

## 🚀 **Quick Start**

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Game:**
   ```bash
   python app.py
   ```

3. **Open Browser:**
   Navigate to `http://127.0.0.1:5000`

4. **Start Playing:**
   - Click "Start New Round"
   - Place bids below market value
   - Try to make the AI abandon first!

## 📁 **File Structure**

```
├── app.py                  # Flask web server & game logic
├── rl_agent.py            # Q-learning AI implementation
├── templates/
│   └── index.html         # Web interface (HTML + JavaScript)
├── requirements.txt       # Python dependencies
├── test_game.py          # Test suite
└── README.md             # Full documentation
```

## 🧠 **How the AI Works**

### Q-Learning Algorithm
- **State Space**: Price range + bid ratios + game phase
- **Action Space**: 8 different bid amounts (30-90% of market value)
- **Reward System**: +100 for winning, bonuses for efficiency
- **Learning Rate**: 0.1 with epsilon-greedy exploration

### Smart Abandonment Criteria
1. **Economic Rationality**: Abandons at 130%+ of market value
2. **Market Awareness**: Progressive abandon probability at 85%+
3. **Risk Assessment**: Analyzes opponent bidding patterns
4. **Performance Adaptation**: Adjusts based on win/loss ratio

## 🎮 **Game Mechanics**

### Bidding Rules
- First bid must be **below** market value
- Each bid must be **higher** than the previous
- Either player can abandon after first bid
- Winner = last bidder before opponent abandons

### Scoring System
```
Points = (Market Value - Final Bid) × 10 + 50
Minimum: 50 points
```

### AI Learning Process
1. **Exploration Phase**: Random strategy testing (ε = 20%)
2. **Learning Phase**: Pattern recognition and Q-value updates
3. **Exploitation Phase**: Uses learned strategies (ε → 5%)
4. **Persistent Memory**: Saves knowledge to `rl_bidding_agent.pkl`

## 🔧 **Technical Details**

### Flask API Endpoints
- `POST /start_round` - Begin new bidding round
- `POST /place_bid` - Submit human bid
- `POST /abandon_bid` - Human abandons
- `GET /get_stats` - Retrieve game statistics
- `POST /reset_game` - Reset all progress

### Session Management
- Player scores and win counts
- Current round state
- Bidding sequence history
- AI learning persistence

## 🐛 **Troubleshooting**

### Common Issues
1. **Port 5000 in use**: Change port in `app.py` (line 299)
2. **Permission errors**: Run as administrator
3. **Module not found**: Ensure `pip install -r requirements.txt`
4. **AI not learning**: Check `rl_bidding_agent.pkl` file creation

### Debug Mode
Flask runs in debug mode by default. Check terminal for errors.

## 🎯 **Game Strategy Tips**

### For Players
- Start with 60-70% of market value
- Watch AI patterns and adapt
- Risk vs reward: higher bids = more points but riskier
- AI becomes smarter over time!

### AI Behavior
- **Early games**: Explores randomly
- **Mid games**: Recognizes patterns
- **Late games**: Sophisticated counter-strategies

## 📊 **Monitoring AI Learning**

The AI's learning progress can be observed through:
- Changing epsilon value (exploration → exploitation)
- Q-table size growth
- Win rate improvements over time
- More strategic abandonment decisions

---

**Ready to challenge the AI?** 🤖 vs 👤

Start the server and visit `http://127.0.0.1:5000` to begin!