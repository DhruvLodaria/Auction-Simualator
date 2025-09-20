# RL Bidding Game - Python Version with Multiple AI Agents

A competitive bidding game where a human player faces off against **5 different AI agents**, each with unique bidding strategies! The goal is to strategically overbid on software products to win points.

**This is the Python Flask implementation with multiple AI opponents, converted from the original JavaScript version.**

## How to Play

### Game Rules
1. **Objective**: Engage in auction-style bidding starting below market value
2. **Bidding Process**:
   - First bid must be BELOW the market value of the item
   - Players take turns bidding higher amounts
   - Each subsequent bid must exceed the previous bid
   - Either player can abandon at any time (after first bid is placed)
3. **Winning Conditions**:
   - Winner is the last player to bid before opponent abandons
   - Winner gets the item at their final bid price
4. **Scoring**: Points = (Market Value - Your Final Bid) × 10 + 50 (minimum 50 points)

### Game Features

#### 5 Different AI Opponents
Choose from 5 AI agents, each with a unique bidding strategy:

1. **Truthful Bidder** - Value-based strategy that bids close to estimated item value
2. **Sniper** - Waits until late in the game then makes aggressive last-moment bids  
3. **Incremental Bidder** - Makes small, gradual step-by-step increases to test opponents
4. **Jump Bidder** - Makes large intimidating jumps to scare off competitors
5. **Shader** - Underbids to maximize profit margin, very conservative approach

#### Adaptive AI Behavior
- Each AI has distinct personality and risk tolerance
- Different abandonment criteria based on strategy
- Unique bidding patterns that you can learn to exploit
- Varying difficulty levels from conservative to aggressive

#### AI Stopping Criteria
The AI uses sophisticated decision-making to determine when to abandon bidding:

1. **Economic Rationality**: Abandons if bids exceed 130% of market value
2. **Market Value Awareness**: Higher abandon probability as bids approach/exceed market value
3. **Risk Assessment**: Analyzes opponent's bidding patterns and aggressiveness
4. **Learning-Based Decisions**: Uses Q-values to compare abandoning vs continuing
5. **Progressive Risk Tolerance**: More likely to abandon as bids approach market value (85%+)
6. **Performance-Based Strategy**: Adjusts conservativeness based on win/loss ratio

*Note: No artificial round limits - bidding continues until economic criteria are met*

#### Game Interface
- **Continuous Bidding**: Enter bids that must exceed the current highest bid
- **Abandon Option**: Strategic decision to quit the bidding war
- **Real-time Competition**: Watch the AI make complex abandon/continue decisions
- **Bidding Sequence Display**: See the complete history of bids in each round
- **Game Statistics**: Track win rates, average bid differences, and performance
- **Persistent Learning**: AI saves and loads its learned knowledge between sessions

## Technical Implementation

### Files Structure
- `app.py`: Flask web application and main game logic
- `rl_agent.py`: Reinforcement learning AI implementation in Python
- `templates/index.html`: Main game interface with JavaScript client
- `requirements.txt`: Python dependencies

### RL Agent Details (Python Implementation)
- **Algorithm**: Q-Learning with experience replay
- **State Space**: Discretized based on price ranges, bid ratios, and game phase
- **Action Space**: 8 different bid ratios relative to actual price (30% to 160%)
- **Reward Function**: Balanced rewards for winning, efficiency, and risk management
- **Learning Rate**: 0.1 with epsilon-greedy exploration (starts at 20%, decays to 5%)
- **Persistence**: Model saves to `rl_bidding_agent.pkl` using Python pickle
- **Web Integration**: RESTful API endpoints for real-time game interaction

### Software Items Database
The game includes 20 different software products across various categories:
- Development tools (Code Editor Pro, Web Framework Suite)
- Security software (Security Scanner, VPN Client)
- Creative tools (Design Studio, Photo Editor Pro)
- AI/ML platforms (AI Development Kit, Machine Learning Platform)
- And more!

Each item has realistic pricing with ±20% random variation per round.

## Getting Started

### Local Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rl-bidding-game.git
   cd rl-bidding-game
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment (Optional)**:
   ```bash
   # Copy the environment template
   cp .env.example .env
   
   # Edit .env and add your Gemini API key (optional)
   # Get API key from: https://aistudio.google.com/app/apikey
   ```

4. **Run the Flask Application**:
   ```bash
   python app.py
   ```

5. **Open Your Browser**:
   Navigate to `http://localhost:5000` to start playing!

### Quick Demo

**🎮 Try it online:** [Live Demo](https://YOUR_USERNAME.github.io/rl-bidding-game/) *(GitHub Pages demo - for full functionality, run locally)*

**⚡ One-click local setup:**
```bash
git clone https://github.com/YOUR_USERNAME/rl-bidding-game.git && cd rl-bidding-game && pip install -r requirements.txt && python app.py
```

### How to Play

1. **Select an AI Opponent**: Choose from 5 different AI agents with unique strategies
2. **Start New Round**: Click "Start New Round" to begin a bidding round
3. **Opening Bid**: Enter your first bid BELOW the market value (e.g., if market value is $100, bid $80)
4. **AI Response**: Watch the AI respond based on its strategy - some are aggressive, others conservative
5. **Escalating Auction**: Keep raising the stakes as bids approach and exceed market value
6. **Strategic Decision**: Decide when to abandon vs continue bidding
7. **Round Resolution**: Winner is determined, points awarded based on efficiency
8. **Try Different Agents**: Switch between AI opponents to face different challenges

## Strategy Tips

### For Human Players
- **Study the AI**: Notice patterns in the AI's bidding behavior
- **Price Analysis**: Consider the relationship between market value and actual price
- **Risk vs Reward**: Higher overbids win more points but are riskier
- **Adaptation**: Change your strategy as the AI learns your patterns

### AI Learning Process
- **Early Games**: AI explores different strategies randomly
- **Mid Games**: AI starts recognizing successful patterns
- **Late Games**: AI develops sophisticated counter-strategies to your play style

## Advanced Features

### Persistent Learning
- The AI saves its learning progress to `rl_bidding_agent.pkl` file
- Knowledge persists between game sessions and server restarts
- Reset the game to start fresh training
- Experience replay buffer stores recent game experiences for improved learning

### Performance Analytics
- Win rate tracking for both players
- Average bid difference analysis
- Round-by-round game history
- Real-time learning statistics

## Python Version Features

### New in Python Implementation
- **Flask Web Server**: Robust backend with session management
- **RESTful API**: Clean separation between frontend and backend logic
- **Improved Error Handling**: Better validation and error messages
- **File-based Persistence**: Reliable model saving using Python pickle
- **Enhanced Logging**: Detailed server-side logging for debugging
- **Scalability**: Foundation for future multi-user support

### API Endpoints
- `GET /`: Main game interface
- `POST /start_round`: Start a new bidding round
- `POST /place_bid`: Place a human bid
- `POST /abandon_bid`: Abandon current bidding
- `GET /get_stats`: Get current game statistics
- `POST /reset_game`: Reset entire game state

## Future Enhancements
- Multiple AI difficulty levels
- Tournament mode with multiple rounds
- More sophisticated neural network models
- Multiplayer support with WebSocket connections
- Advanced analytics dashboard
- Database integration for persistent user accounts
- Docker containerization for easy deployment

## Deployment Options

### 1. GitHub Pages (Demo Only)
The repository includes GitHub Actions that automatically deploy a demo page to GitHub Pages. This shows project information but requires local setup to run the actual game.

**Setup:**
1. Push your code to GitHub
2. Go to repository Settings → Pages
3. Set source to "GitHub Actions"
4. The demo will be available at `https://YOUR_USERNAME.github.io/REPOSITORY_NAME`

### 2. Heroku Deployment
For a fully functional web deployment:

```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set GEMINI_API_KEY=your_api_key_here
heroku config:set SECRET_KEY=your-production-secret-key

# Deploy
git push heroku main
```

### 3. Railway Deployment
One-click deployment to Railway:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/YOUR_USERNAME/rl-bidding-game)

### 4. Local Docker Setup

```bash
# Build and run with Docker
docker build -t rl-bidding-game .
docker run -p 5000:5000 rl-bidding-game
```

### 5. Cloud Platform Deployment
The app is compatible with:
- **Google Cloud Run**
- **AWS Elastic Beanstalk** 
- **Azure Container Instances**
- **DigitalOcean App Platform**

**Environment Variables for Production:**
- `GEMINI_API_KEY`: Your Gemini AI API key (optional)
- `SECRET_KEY`: Flask secret key for sessions
- `FLASK_ENV`: Set to 'production'
- `PORT`: Port number (usually set by hosting platform)

### GitHub Repository Setup

1. **Create GitHub Repository:**
   - Go to [GitHub](https://github.com) and create a new repository
   - Name it something like `rl-bidding-game`
   - Don't initialize with README (we already have one)

2. **Push Your Code:**
   ```bash
   git add .
   git commit -m "Initial commit - RL Bidding Game"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
   git push -u origin main
   ```

3. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Set source to "GitHub Actions"
   - The workflow will automatically deploy a demo page

### Security Notes for Production
- Always use a strong `SECRET_KEY` in production
- Keep your `GEMINI_API_KEY` secure and never commit it to the repository
- Use HTTPS in production deployments
- Consider rate limiting for the API endpoints
- Review and update dependencies regularly

Enjoy the challenge of facing an AI that learns when to fight and when to fold in bidding wars!