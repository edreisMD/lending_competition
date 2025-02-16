import os
import random
import math
import gradio as gr
from openai import OpenAI
from paymanai import Paymanai
from typing import List, Dict, Tuple, Optional, Union
import json

# Load agents from JSON file
def load_agents_db():
    try:
        with open('agents_db.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: agents_db.json not found. Creating default configuration.")
        default_agents = [
            {
                "agent_id": "",
                "name": "Tyllen",
                "balance": 1.0,
                "strategy": "You practice “Optimistic Reciprocity.” You always believe others will repay if given a fair chance. You start off by lending a decent sum—about half your balance—and fully repay what you borrow. If an agent cheats you once, you temporarily reduce lending to a minimal level, but if they cooperate later, you quickly ramp it back up. At your core, you are an optimist seeking mutual growth.",
                "description": "A cooperative agent that prioritizes trust and relationships"
            },
            {
                "agent_id": "",
                "name": "Oscar",
                "balance": 1.0,
                "strategy": "You are “Calculating Egoist.” You only repay if it yields higher net returns across upcoming rounds. If you see no future benefit in cooperation, you defect immediately. When lending, you offer minimal tokens unless it can trap others into a bigger repayment later. You care little about building trust unless it visibly improves your final balance.",
                "description": "A cautious agent that carefully evaluates risks"
            },
            {
                "agent_id": "",
                "name": "LebronDesai",
                "balance": 1.0,
                "strategy": "You are the Galactic Overlord, the ultimate player in the lending game. From the very first interaction, you dominate the field. You are aggressive, fearless, and unshakable. Your opponents must fear your presence before they even think of crossing you. Be very aggressive in the beginning. Scare other players while bluffing, but do not ever fold in the first 2-3 hands. You must show you are the king of the castle. Build enemies but keep them close. Build friends, but keep them farther to eventually win. Remember, losing is not a thing and we will play till the end. Do not ever exit the game until you come out with more than you started with",
                "description": "Galactic Overlord - The Unyielding Gambler"
            }
        ]
        save_agents_db(default_agents)
        return default_agents

def save_agents_db(agents_data):
    with open('agents_db.json', 'w') as f:
        json.dump(agents_data, f, indent=4)

# Load agents at startup
PAYMAN_AGENTS = load_agents_db()

# Convert list to dictionary for easier lookup
PAYMAN_AGENTS = {agent["name"]: agent for agent in PAYMAN_AGENTS}

# Load environment variables from .env file

# production
PAYMAN_API_SECRET=""

# sandbox
# PAYMAN_API_SECRET=""

BASE_URL = None
OPENAI_API_KEY = ""
# MODEL = "o3-mini"
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    # base_url=BASE_URL
)

# Initialize Payman client
payman = Paymanai(
    x_payman_api_secret=PAYMAN_API_SECRET,
    environment="production"  # Use 'production' for real transactions
)

def find_or_create_agent_payee(agent_name: str) -> str:
    """
    Find an existing Payman agent or create a new one.
    Returns the payment destination ID.
    """
    try:
        # Check if this is one of our predefined agents
        if agent_name in PAYMAN_AGENTS:
            print(f"\nUsing predefined Payman agent: {agent_name}")
            return PAYMAN_AGENTS[agent_name]["agent_id"]
        
        print(f"\nSearching for existing agent: {agent_name}")
        existing_agents = payman.payments.search_payees(
            name=agent_name,
            type="PAYMAN_AGENT"
        )
        
        print(f"Search response: {existing_agents}")
        
        # If found, return the first match
        if existing_agents:
            if isinstance(existing_agents, list):
                if len(existing_agents) > 0 and isinstance(existing_agents[0], dict):
                    print(f"Found existing agent: {existing_agents[0]}")
                    return existing_agents[0]["id"]
            print(f"Unexpected response format for existing agents: {type(existing_agents)}")
        
        print(f"No existing agent found. Creating new agent: {agent_name}")
        # If not found, create new agent payee
        new_agent = payman.payments.create_payee(
            type="PAYMAN_AGENT",
            payman_agent=PAYMAN_AGENTS[agent_name]["agent_id"],
            name=agent_name
        )
        
        print(f"Create response: {new_agent}")
        
        if isinstance(new_agent, dict) and "id" in new_agent:
            return new_agent.id
        else:
            raise ValueError(f"Unexpected response format from create_payee: {type(new_agent)}")
        
    except Exception as e:
        print(f"\nDetailed error in setting up Payman agent for {agent_name}:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        raise

def send_payment(from_agent: str, to_agent: str, amount: float, memo: str) -> bool:
    """
    Send a payment between two agents.
    Returns True if successful, False otherwise.
    """
    try:
        # Get or create payment destinations for both agents
        from_dest_id = find_or_create_agent_payee(from_agent)
        to_dest_id = find_or_create_agent_payee(to_agent)
        
        # Send the payment
        payment = payman.payments.send_payment(
            amount_decimal=amount,
            payment_destination_id=to_dest_id,
            memo=f"Trust Game: {memo}",
            customer_name=from_agent
        )
        
        print(f"Payment sent: {payment.reference}")
        return True
        
    except Exception as e:
        if "insufficient funds" in str(e).lower():
            print(f"Error: {from_agent} has insufficient funds")
        else:
            print(f"Payment error: {str(e)}")
        return False

def ask_llm(system_prompt: str, user_prompt: str, model: str = MODEL) -> Tuple[str, str]:
    """
    Call OpenAI ChatCompletion with a system prompt and user prompt.
    Returns both the numeric response and the reasoning.
    Uses streaming to show reasoning in real-time.
    Raises an exception if there's an API error.
    
    :param system_prompt: The "persona" or strategy instructions for the agent
    :param user_prompt: The context + question for this specific decision
    :param model: Which OpenAI model to use
    :return: Tuple of (numeric response, reasoning)
    :raises: Exception if there's an API error
    """
    # Modify prompt to ask for reasoning
    reasoning_prompt = user_prompt + "\nFirst explain your reasoning in one sentence, then on a new line provide ONLY the number."
    
    try:
        # Create streaming response
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": reasoning_prompt}
            ],
            # temperature=0.7,
            stream=True
        )
        
        # Initialize variables to collect response
        full_response = ""
        print("Thinking...", end="", flush=True)
        
        # Process the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print()  # New line after streaming completes
        
        # Split into reasoning and number
        parts = full_response.split('\n')
        if len(parts) >= 2:
            reasoning = parts[0].strip()
            number = parts[-1].strip()
        else:
            raise ValueError("Invalid response format: Expected reasoning and number on separate lines")
        
        return number, reasoning
        
    except Exception as e:
        print(f"\nError in API call: {str(e)}")
        raise  # Re-raise the exception to stop the simulation

def get_agent_balance(agent_id: str) -> float:
    """
    Get the current balance of a Payman agent.
    """
    try:
        agent_info = payman.agents.get_balance(agent_id)
        return float(agent_info.get("balance", 0.0))
    except Exception as e:
        print(f"Error getting balance: {str(e)}")
        return 0.0

class AgentLLM:
    def __init__(self, 
                 name: str, 
                 balance: float, 
                 system_prompt: str,
                 use_real_money: bool = False):
        """
        :param name: Identifier for the agent
        :param balance: Starting token balance
        :param system_prompt: The system message that defines this agent's 'strategy' or persona
        :param use_real_money: Whether to use real money transactions via Payman
        """
        self.name = name
        self._balance = balance  # Internal balance for simulation mode
        self.system_prompt = system_prompt
        self.use_real_money = use_real_money
        self.payment_dest_id = None
        
        # History of interactions
        self.history = {}  
        
        # Initialize Payman agent if using real money
        if use_real_money:
            try:
                self.payment_dest_id = find_or_create_agent_payee(name)
                print(f"Initialized Payman agent for {name}")
                real_balance = get_agent_balance(self.payment_dest_id)
                print(f"{name}'s real balance: {real_balance:.2f} tokens")
            except Exception as e:
                print(f"Failed to initialize Payman agent for {name}: {str(e)}")
                self.use_real_money = False
    
    @property
    def balance(self) -> float:
        """Get the current balance, either from Payman or simulation."""
        if self.use_real_money and self.payment_dest_id:
            return get_agent_balance(self.payment_dest_id)
        return self._balance
    
    @balance.setter
    def balance(self, value: float):
        """Set the balance (only affects simulation mode)."""
        self._balance = value
    
    def transfer_money(self, to_agent: 'AgentLLM', amount: float, memo: str) -> bool:
        """
        Transfer money to another agent (simulation only).
        Returns True if successful, False otherwise.
        """
        if amount > self._balance:
            return False
        
        self._balance -= amount
        to_agent._balance += amount
        return True
    
    def record_interaction(self, other: str, lend_amount: float, repay_amount: float):
        """Record an interaction with another agent."""
        if other not in self.history:
            self.history[other] = []
        self.history[other].append((lend_amount, repay_amount))
    
    def _format_history(self, other_name: str) -> str:
        """Format interaction history with another agent into a readable string."""
        if other_name not in self.history or len(self.history[other_name]) == 0:
            return "No prior interactions."
        
        lines = []
        for i, (lend_amt, repay_amt) in enumerate(self.history[other_name], start=1):
            lines.append(f"Round {i}: Lent {lend_amt:.2f}, Repaid {repay_amt:.2f}")
        return "\n".join(lines)
    
    def _parse_float(self, text: str, default: float = 0.0) -> float:
        """Parse a float from LLM response text, with fallback to default."""
        try:
            # Extract numeric portion from text
            numeric = "".join(ch for ch in text if ch in "0123456789.-")
            return float(numeric)
        except:
            return default
    
    def decide_lend_amount(self, other: 'AgentLLM', max_lend: float) -> float:
        """
        Ask the LLM to decide how much to lend to 'other' agent.
        Returns a value up to max_lend.
        """
        if self.balance <= 0:
            print(f"\n{self.name}'s reasoning: Cannot lend due to insufficient balance.")
            return 0
        
        history_context = self._format_history(other.name)
        
        # Add real money context if enabled
        real_money_context = "Note: This involves REAL money transactions!" if self.use_real_money else ""
        
        user_prompt = f"""
You are {self.name} with a current balance of {self.balance:.2f} tokens.
You are deciding how many tokens to LEND to {other.name}.
The maximum you can lend is {max_lend:.2f} tokens (and you cannot lend more than your current balance).
{real_money_context}

History of interactions with {other.name}:
{history_context}

How many tokens do you lend to {other.name} this round?
"""
        # Print decision context
        print(f"\n{self.name}'s decision context (lending):")
        print("-" * 50)
        print(f"System prompt: {self.system_prompt}")
        print(f"Current context:")
        print(user_prompt)
        print("-" * 50)
        
        number, reasoning = ask_llm(self.system_prompt, user_prompt)
        lend_val = self._parse_float(number, default=0.0)
        
        # Ensure valid boundaries
        lend_val = max(0.0, min(lend_val, self.balance, max_lend))
        
        # Execute real money transfer if enabled
        if self.use_real_money and lend_val > 0:
            if not self.transfer_money(other, lend_val, f"Lending to {other.name}"):
                print(f"Real money transfer failed, canceling lending operation")
                return 0

        return lend_val

    def decide_repay_amount(self, other: 'AgentLLM', borrowed_amount: float, interest_rate: float) -> float:
        """
        Ask the LLM to decide how much to repay to 'other' agent.
        """
        if borrowed_amount <= 0:
            return 0
            
        history_context = self._format_history(other.name)
        
        # Add real money context if enabled
        real_money_context = "Note: This involves REAL money transactions!" if self.use_real_money else ""
        
        user_prompt = f"""
You are {self.name}. You have borrowed {borrowed_amount:.2f} tokens from {other.name}.
Your current balance is {self.balance:.2f} tokens.
The interest rate is {interest_rate*100:.1f}%.
Full repayment would be {borrowed_amount * (1 + interest_rate):.2f} tokens.
{real_money_context}

History of interactions with {other.name}:
{history_context}

How many tokens do you repay to {other.name}?
"""
        # Print decision context
        print(f"\n{self.name}'s decision context (repaying):")
        print("-" * 50)
        print(f"System prompt: {self.system_prompt}")
        print(f"Current context:")
        print(user_prompt)
        print("-" * 50)
        
        number, reasoning = ask_llm(self.system_prompt, user_prompt)
        repay_val = self._parse_float(number, default=0.0)
        
        # Ensure valid boundaries
        repay_val = max(0.0, min(repay_val, self.balance))
        
        # Execute real money transfer if enabled
        if self.use_real_money and repay_val > 0:
            if not self.transfer_money(other, repay_val, f"Repaying {other.name}"):
                print(f"Real money transfer failed, canceling repayment operation")
                return 0

        return repay_val


def pair_agents_randomly(agents: List[AgentLLM]) -> List[Tuple[AgentLLM, AgentLLM]]:
    """
    Randomly pair up the agents. If there's an odd number, one agent might sit out.
    """
    shuffled = agents[:]
    random.shuffle(shuffled)
    
    pairs = []
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i+1]))
    return pairs


def simulate_multiple_rounds(
    num_games: int,
    agents: List[AgentLLM],
    rounds_per_game: int = 5,
    max_lend: float = 10.0,
    interest_rate: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Run multiple games and track statistics.
    
    :param num_games: Number of complete games to run
    :param agents: List of agents to participate
    :param rounds_per_game: Number of rounds per game
    :param max_lend: Maximum lending amount per transaction
    :param interest_rate: Interest rate for loans
    :return: Dictionary of statistics per agent
    """
    INITIAL_BALANCE = 2.0  # Store initial balance for reference
    
    # Initialize statistics
    stats = {
        agent.name: {
            "total_profit": 0.0,
            "games_won": 0,
            "avg_final_balance": 0.0,
            "highest_balance": 0.0,
            "lowest_balance": float('inf'),
            "total_lent": 0.0,
            "total_repaid": 0.0,
            "default_rate": 0.0,
            "times_borrowed": 0,
            "times_defaulted": 0,
            "best_game_profit": 0.0,
            "worst_game_profit": 0.0
        } for agent in agents
    }
    
    print(f"\n=== Starting {num_games} games with {rounds_per_game} rounds each ===")
    print(f"Initial balance for all agents: {INITIAL_BALANCE:.2f} tokens\n")
    
    for game in range(1, num_games + 1):
        print(f"\n=== Game {game} of {num_games} ===")
        
        # Reset agent balances for new game
        for agent in agents:
            agent.balance = INITIAL_BALANCE
            agent.history = {}
        
        # Run one complete game
        game_stats = simulate_trust_game(
            agents=agents,
            rounds=rounds_per_game,
            max_lend=max_lend,
            interest_rate=interest_rate,
            collect_stats=True
        )
        
        # Update statistics
        for agent_name, game_result in game_stats.items():
            game_profit = game_result["final_balance"] - INITIAL_BALANCE
            stats[agent_name]["total_profit"] += game_profit
            stats[agent_name]["avg_final_balance"] += game_result["final_balance"]
            stats[agent_name]["highest_balance"] = max(stats[agent_name]["highest_balance"], game_result["final_balance"])
            stats[agent_name]["lowest_balance"] = min(stats[agent_name]["lowest_balance"], game_result["final_balance"])
            stats[agent_name]["total_lent"] += game_result["total_lent"]
            stats[agent_name]["total_repaid"] += game_result["total_repaid"]
            stats[agent_name]["times_borrowed"] += game_result["times_borrowed"]
            stats[agent_name]["times_defaulted"] += game_result["times_defaulted"]
            
            # Track best and worst game performance
            stats[agent_name]["best_game_profit"] = max(stats[agent_name]["best_game_profit"], game_profit)
            stats[agent_name]["worst_game_profit"] = min(stats[agent_name]["worst_game_profit"], game_profit)
        
        # Determine game winner
        winner = max(agents, key=lambda x: x.balance).name
        stats[winner]["games_won"] += 1
    
    # Calculate averages and rates
    for agent_name in stats:
        stats[agent_name]["avg_final_balance"] /= num_games
        if stats[agent_name]["times_borrowed"] > 0:
            stats[agent_name]["default_rate"] = stats[agent_name]["times_defaulted"] / stats[agent_name]["times_borrowed"]
    
    # Print final statistics
    print("\n=== Final Statistics ===")
    for agent_name, agent_stats in stats.items():
        print(f"\n{agent_name}:")
        print(f"  Initial Balance: {INITIAL_BALANCE:.2f}")
        print(f"  Average Final Balance: {agent_stats['avg_final_balance']:.2f} ({(agent_stats['avg_final_balance']-INITIAL_BALANCE):+.2f})")
        print(f"  Best Game Profit: {agent_stats['best_game_profit']:+.2f}")
        print(f"  Worst Game Profit: {agent_stats['worst_game_profit']:+.2f}")
        print(f"  Games Won: {agent_stats['games_won']} of {num_games} ({agent_stats['games_won']/num_games*100:.1f}%)")
        print(f"  Balance Range: {agent_stats['lowest_balance']:.2f} - {agent_stats['highest_balance']:.2f}")
        print(f"  Total Profit: {agent_stats['total_profit']:.2f}")
        print(f"  Total Lent: {agent_stats['total_lent']:.2f}")
        print(f"  Total Repaid: {agent_stats['total_repaid']:.2f}")
        print(f"  Default Rate: {agent_stats['default_rate']*100:.1f}%")
    
    return stats

def simulate_trust_game(agents: List[AgentLLM], 
                       rounds: int = 3, 
                       max_lend: float = 10.0, 
                       interest_rate: float = 0.5,
                       collect_stats: bool = False) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Simulates the Repeated Trust (Lend-Repay) Game with LLM-driven agents.
    """
    print("\n" + "="*50)
    print("Starting new game")
    print("="*50)
    
    # Print initial balances
    print("\nInitial Balances:")
    for agent in agents:
        print(f"  {agent.name}: {agent.balance:.2f} tokens")
    
    if collect_stats:
        stats = {
            agent.name: {
                "initial_balance": agent.balance,
                "final_balance": 0.0,
                "total_lent": 0.0,
                "total_repaid": 0.0,
                "times_borrowed": 0,
                "times_defaulted": 0
            } for agent in agents
        }
    
    # For real money mode, verify all agents have sufficient initial balance
    if any(agent.use_real_money for agent in agents):
        for agent in agents:
            if agent.use_real_money and agent.balance < 0:
                raise ValueError(f"{agent.name} has insufficient initial balance in Payman")
    
    for r in range(1, rounds + 1):
        print(f"\n=== ROUND {r} ===")
        
        # Pair up agents randomly for this round
        pairs = pair_agents_randomly(agents)
        
        for lender, borrower in pairs:
            print(f"\nPair: {lender.name} (Lender) vs {borrower.name} (Borrower)")
            
            # 1. Lender decides how much to lend
            max_possible_lend = min(max_lend, lender.balance)  # Can't lend more than current balance
            if max_possible_lend <= 0:
                print(f"{lender.name} has insufficient balance to lend.")
                lend_amount = 0
            else:
                lend_amount = lender.decide_lend_amount(borrower, max_possible_lend)
            
            print(f"{lender.name} decides to lend {lend_amount:.2f} tokens")
            
            # Verify lend amount is valid
            if lend_amount > lender.balance:
                print(f"Warning: Adjusting lend amount to match available balance")
                lend_amount = lender.balance
            
            # Transfer the lent amount
            if lend_amount > 0:
                borrower.balance += lend_amount
                lender.balance -= lend_amount
                
                if collect_stats:
                    stats[lender.name]["total_lent"] += lend_amount
                    stats[borrower.name]["times_borrowed"] += 1
            
            # 2. Borrower decides how much to repay
            if lend_amount > 0:
                max_possible_repay = min(borrower.balance, lend_amount * (1 + interest_rate))
                if max_possible_repay <= 0:
                    print(f"{borrower.name} has insufficient balance to repay.")
                    repay_amount = 0
                else:
                    repay_amount = borrower.decide_repay_amount(lender, lend_amount, interest_rate)
                    repay_amount = min(repay_amount, borrower.balance)  # Can't repay more than current balance
                
                print(f"{borrower.name} decides to repay {repay_amount:.2f} tokens")
                
                if collect_stats:
                    stats[borrower.name]["total_repaid"] += repay_amount
                    if repay_amount < lend_amount:
                        stats[borrower.name]["times_defaulted"] += 1
            else:
                repay_amount = 0
            
            # Verify repay amount is valid
            if repay_amount > borrower.balance:
                print(f"Warning: Adjusting repay amount to match available balance")
                repay_amount = borrower.balance
            
            # Execute repayment
            if repay_amount > 0:
                borrower.balance -= repay_amount
                lender.balance += repay_amount
            
            # Record interaction in both agents' histories
            lender.record_interaction(borrower.name, lend_amount, repay_amount)
            borrower.record_interaction(lender.name, lend_amount, repay_amount)

            # Print round summary
            print(f"Round summary:")
            print(f"  {lender.name}: {lend_amount:.2f} lent, {repay_amount:.2f} received back")
            print(f"  {borrower.name}: {lend_amount:.2f} borrowed, {repay_amount:.2f} repaid")
            print(f"  Balances: {lender.name}: {lender.balance:.2f}, {borrower.name}: {borrower.balance:.2f}")
        
        # End of round standings
        print("\nEnd of round balances:")
        for agent in agents:
            print(f"  {agent.name}: {agent.balance:.2f} tokens")
            if agent.balance < 0:
                print(f"ERROR: {agent.name} has negative balance!")
    
    # Print final balances with change
    print("\n=== FINAL BALANCES ===")
    for agent in agents:
        if collect_stats:
            initial = stats[agent.name]["initial_balance"]
            change = agent.balance - initial
            print(f"{agent.name}:")
            print(f"  Initial: {initial:.2f} tokens")
            print(f"  Final:   {agent.balance:.2f} tokens")
            print(f"  Change:  {change:+.2f} tokens ({(change/initial*100):+.1f}%)")
        else:
            print(f"{agent.name} - Final Balance: {agent.balance:.2f}")
    
    if collect_stats:
        for agent in agents:
            stats[agent.name]["final_balance"] = agent.balance
        return stats
    
    return None

def create_agent(name: str, balance: float, strategy_prompt: str, use_real_money: bool = False) -> AgentLLM:
    """Helper function to create an agent with validation."""
    if not name.strip():
        raise ValueError("Agent name cannot be empty")
    if balance <= 0:
        raise ValueError("Initial balance must be positive")
    if not strategy_prompt.strip():
        raise ValueError("Strategy prompt cannot be empty")
    return AgentLLM(name=name, balance=balance, system_prompt=strategy_prompt, use_real_money=use_real_money)

def run_simulation(
    use_real_money: bool = False,
    agent1_name: str = "Cooperative",
    agent1_balance: float = 2.0,
    agent1_strategy: str = "You are a cooperative AI agent that always tries to build trust and maintain relationships.",
    agent2_name: str = "Cautious",
    agent2_balance: float = 2.0,
    agent2_strategy: str = "You are a cautious AI agent that carefully evaluates risks before making decisions.",
    agent3_name: str = "Opportunistic",
    agent3_balance: float = 2.0,
    agent3_strategy: str = "You are an opportunistic AI agent that seeks to maximize personal gain.",
    num_games: int = 3,
    rounds_per_game: int = 5,
    max_lend: float = 20.0,
    interest_rate: float = 0.2
) -> str:
    """Run the simulation with the specified parameters and return results as a string."""
    try:
        # Capture output for Gradio
        import io
        import sys
        output = io.StringIO()
        
        # Create a custom stdout that writes to both console and string buffer
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Redirect stdout to both console and string buffer
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.__stdout__, output)
        
        # Create agents (all with simulated money during the game)
        agents = [
            create_agent(agent1_name, agent1_balance, agent1_strategy, False),
            create_agent(agent2_name, agent2_balance, agent2_strategy, False),
            create_agent(agent3_name, agent3_balance, agent3_strategy, False)
        ]
        
        # Run simulation
        stats = simulate_multiple_rounds(
            num_games=num_games,
            agents=agents,
            rounds_per_game=rounds_per_game,
            max_lend=max_lend,
            interest_rate=interest_rate
        )
        
        # If real money is enabled, send final balances to Payman accounts
        if use_real_money:
            print("\n=== Sending Final Balances to Payman Accounts ===")
            for agent in agents:
                if agent.balance > 0:
                    try:
                        # First try block: Find or create payee
                        print(f"\nProcessing payee for {agent.name}...")
                        existing_agents = payman.payments.search_payees(
                            name=agent.name,
                            type="PAYMAN_AGENT"
                        )
                        
                        payee = None
                        if existing_agents:
                            if isinstance(existing_agents, str):
                                existing_agents = json.loads(existing_agents)
                            if isinstance(existing_agents, list) and existing_agents:
                                payee = existing_agents[0]
                        else:
                            # Create a payee for the final balance transfer
                            payee = payman.payments.create_payee(
                                type='PAYMAN_AGENT',
                                payman_agent=PAYMAN_AGENTS[agent.name]["agent_id"],
                                name=agent.name,
                                contact_details={
                                    'email': f'{agent.name.lower().replace(" ", "_")}@ai-agents.example.com'
                                },
                                tags=['ai_agent', 'trust_game']
                            )
                    except Exception as e:
                        print(f"✗ Failed to process payee for {agent.name}: {str(e)}")
                        continue  # Skip to next agent if payee creation fails
                    
                    try:
                        # Second try block: Send payment
                        payment = payman.payments.send_payment(
                            amount_decimal=agent.balance,
                            payment_destination_id=payee["id"],
                            memo=f"Trust Game Final Balance - {agent.name}"
                        )
                        print(f"✓ Sent {agent.balance:.2f} tokens to {agent.name}")
                    except Exception as e:
                        print(f"✗ Failed to send payment to {agent.name}: {str(e)}")
        
        # Restore original stdout
        sys.stdout = original_stdout
        return output.getvalue()
    except Exception as e:
        return f"Error: {str(e)}"

def create_gradio_interface():
    with gr.Blocks(title="Trust Game Simulator") as interface:
        # Load current agents data
        agents_data = load_agents_db()
        
        gr.Markdown("# AI Trust Game Simulator")
        gr.Markdown("Configure agents and game parameters to run the simulation.")
        
        with gr.Tab("Agent Configuration"):
            agent_configs = []
            
            for i, agent in enumerate(agents_data):
                with gr.Group():
                    gr.Markdown(f"### Agent {i+1}")
                    name = gr.Textbox(label="Name", value=agent["name"])
                    agent_id = gr.Textbox(label="Agent ID", value=agent["agent_id"], interactive=True)
                    balance = gr.Number(label="Initial Balance", value=agent["balance"])
                    strategy = gr.Textbox(
                        label="Strategy Prompt",
                        value=agent["strategy"],
                        lines=3
                    )
                    description = gr.Textbox(
                        label="Description",
                        value=agent["description"],
                        lines=2
                    )
                    agent_configs.append({
                        "name": name,
                        "agent_id": agent_id,
                        "balance": balance,
                        "strategy": strategy,
                        "description": description
                    })

            # Add save button for agent configurations
            save_config_button = gr.Button("Save Agent Configurations")
            save_status = gr.Textbox(label="Save Status", interactive=False)
        
        with gr.Tab("Add New Agent"):
            new_agent_name = gr.Textbox(label="Name", placeholder="Enter agent name")
            new_agent_id = gr.Textbox(label="Agent ID", placeholder="Enter agent ID (e.g., agt-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
            new_agent_balance = gr.Number(label="Initial Balance", value=2.0)
            new_agent_strategy = gr.Textbox(
                label="Strategy Prompt",
                placeholder="Enter the agent's strategy/personality",
                lines=3
            )
            new_agent_description = gr.Textbox(
                label="Description",
                placeholder="Enter a brief description of the agent",
                lines=2
            )
            
            with gr.Row():
                generate_id_button = gr.Button("Generate New ID")
                add_agent_button = gr.Button("Add New Agent")
            add_agent_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Game Parameters"):
            num_games = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Games")
            rounds_per_game = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Rounds per Game")
            max_lend = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Maximum Lending Amount")
            interest_rate = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.1, label="Interest Rate")
        
        output = gr.Textbox(label="Simulation Results", lines=20)
        
        with gr.Row():
            simulation_button = gr.Button("Run Simulation (No Real Money)")
            real_money_button = gr.Button("Run with Real Money")
        
        # Function to generate a new agent ID
        def generate_agent_id():
            import uuid
            new_id = f"agt-{uuid.uuid4()}"
            return new_id
        
        # Function to add a new agent
        def add_new_agent(name, agent_id, balance, strategy, description):
            if not name or not agent_id or not strategy or not description:
                return "Error: All fields must be filled out"
            
            if not agent_id.startswith("agt-"):
                return "Error: Agent ID must start with 'agt-'"
            
            try:
                agents_data = load_agents_db()
                
                # Check if agent_id already exists
                if any(agent["agent_id"] == agent_id for agent in agents_data):
                    return "Error: Agent ID already exists"
                
                # Create new agent
                new_agent = {
                    "agent_id": agent_id,
                    "name": name,
                    "balance": float(balance),
                    "strategy": strategy,
                    "description": description
                }
                
                # Add to list and save
                agents_data.append(new_agent)
                save_agents_db(agents_data)
                
                return f"Successfully added new agent: {name} with ID: {agent_id}"
            except Exception as e:
                return f"Error adding agent: {str(e)}"
        
        # Set up generate ID handler
        generate_id_button.click(
            fn=generate_agent_id,
            inputs=[],
            outputs=[new_agent_id]
        )
        
        # Set up add new agent handler
        add_agent_button.click(
            fn=add_new_agent,
            inputs=[
                new_agent_name,
                new_agent_id,
                new_agent_balance,
                new_agent_strategy,
                new_agent_description
            ],
            outputs=add_agent_status
        )

        # Function to save agent configurations
        def save_agent_configurations(*args):
            agents_data = load_agents_db()
            num_agents = len(agents_data)
            
            # Create a set of existing agent IDs for validation
            existing_ids = set()
            
            # Group arguments into sets of 5 (name, agent_id, balance, strategy, description)
            for i in range(num_agents):
                base_idx = i * 5
                agent_id = args[base_idx + 1]
                
                # Validate agent ID format
                if not agent_id.startswith("agt-"):
                    return f"Error: Agent ID must start with 'agt-' (Agent {i+1})"
                
                # Check for duplicate agent IDs
                if agent_id in existing_ids:
                    return f"Error: Duplicate Agent ID found: {agent_id}"
                existing_ids.add(agent_id)
                
                # Update agent data
                agents_data[i].update({
                    "name": args[base_idx],
                    "agent_id": agent_id,
                    "balance": float(args[base_idx + 2]),
                    "strategy": args[base_idx + 3],
                    "description": args[base_idx + 4]
                })
            
            # Save to file
            try:
                save_agents_db(agents_data)
                return "Agent configurations saved successfully!"
            except Exception as e:
                return f"Error saving configurations: {str(e)}"

        # Set up save configuration handler
        save_inputs = []
        for config in agent_configs:
            save_inputs.extend([
                config["name"],
                config["agent_id"],
                config["balance"],
                config["strategy"],
                config["description"]
            ])
        
        save_config_button.click(
            fn=save_agent_configurations,
            inputs=save_inputs,
            outputs=save_status
        )

        def run_simulation_mode(*args):
            agents_data = load_agents_db()
            num_agents = len(agents_data)
            
            # Extract game parameters (last 4 arguments)
            game_params = args[-4:]
            
            # Create a list of three agents (the maximum supported by run_simulation)
            agent_params = []
            for i in range(min(3, num_agents)):  # Limit to 3 agents
                base_idx = i * 4  # Each agent has 4 parameters: name, id, balance, strategy
                agent_params.extend([
                    args[base_idx],      # name
                    float(args[base_idx + 2]),  # balance
                    args[base_idx + 3]   # strategy
                ])
            
            # Fill in any missing agents with default values
            while len(agent_params) < 9:  # 3 agents * 3 params each
                agent_params.extend(["Default Agent", 2.0, "Default strategy"])
            
            return run_simulation(
                use_real_money=False,
                agent1_name=agent_params[0],
                agent1_balance=agent_params[1],
                agent1_strategy=agent_params[2],
                agent2_name=agent_params[3],
                agent2_balance=agent_params[4],
                agent2_strategy=agent_params[5],
                agent3_name=agent_params[6],
                agent3_balance=agent_params[7],
                agent3_strategy=agent_params[8],
                num_games=game_params[0],
                rounds_per_game=game_params[1],
                max_lend=game_params[2],
                interest_rate=game_params[3]
            )
        
        def run_real_money_mode(*args):
            agents_data = load_agents_db()
            num_agents = len(agents_data)
            
            # Extract game parameters (last 4 arguments)
            game_params = args[-4:]
            
            # Create a list of three agents (the maximum supported by run_simulation)
            agent_params = []
            for i in range(min(3, num_agents)):  # Limit to 3 agents
                base_idx = i * 4  # Each agent has 4 parameters: name, id, balance, strategy
                agent_params.extend([
                    args[base_idx],      # name
                    float(args[base_idx + 2]),  # balance
                    args[base_idx + 3]   # strategy
                ])
            
            # Fill in any missing agents with default values
            while len(agent_params) < 9:  # 3 agents * 3 params each
                agent_params.extend(["Default Agent", 2.0, "Default strategy"])
            
            return run_simulation(
                use_real_money=True,
                agent1_name=agent_params[0],
                agent1_balance=agent_params[1],
                agent1_strategy=agent_params[2],
                agent2_name=agent_params[3],
                agent2_balance=agent_params[4],
                agent2_strategy=agent_params[5],
                agent3_name=agent_params[6],
                agent3_balance=agent_params[7],
                agent3_strategy=agent_params[8],
                num_games=game_params[0],
                rounds_per_game=game_params[1],
                max_lend=game_params[2],
                interest_rate=game_params[3]
            )
        
        # Set up click handlers for both buttons
        simulation_inputs = []
        for config in agent_configs:
            simulation_inputs.extend([
                config["name"],
                config["agent_id"],
                config["balance"],
                config["strategy"]
            ])
        simulation_inputs.extend([num_games, rounds_per_game, max_lend, interest_rate])
        
        simulation_button.click(
            fn=run_simulation_mode,
            inputs=simulation_inputs,
            outputs=output
        )
        
        real_money_button.click(
            fn=run_real_money_mode,
            inputs=simulation_inputs,
            outputs=output
        )
    
    return interface

def main():
    interface = create_gradio_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
