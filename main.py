# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import os
import importlib.util
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='backtesting_bot.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# File structure:
# - app/
#   - main.py
#   - core/
#       - backtesting_engine.py
#       - kpi_calculator.py
#       - visualizer.py
#   - strategies/
#       - example_strategy.py
#   - data/
#       - example_data.csv
#   - logs/
#   - output/

# Workflow:
# 1. Parse arguments (strategy file, data file, configuration parameters).
# 2. Load and validate data from CSV.
# 3. Import and validate the strategy.
# 4. Run the backtesting engine.
# 5. Calculate KPIs.
# 6. Generate visualizations and summary report.

# Backtesting Engine Class
class BacktestingEngine:
    def __init__(self, data, strategy, initial_capital=100000, config=None):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.config = config or {}
        self.portfolio = pd.DataFrame(index=self.data.index)
        self.portfolio['cash'] = float(initial_capital)
        self.portfolio['positions'] = 0.0
        self.portfolio['total'] = float(initial_capital)
        
        # Ensure float dtype for all numeric columns
        self.portfolio = self.portfolio.astype({
            'cash': 'float64',
            'positions': 'float64',
            'total': 'float64'
        })

    def run(self):
        try:
            signals = self.strategy.generate_signals(self.data)
            for date, signal in signals.iterrows():
                self._execute_trade(date, signal)
            self._finalize_portfolio()
        except Exception as e:
            logging.error(f"Error during backtesting: {e}")
            raise

    def _execute_trade(self, date, signal):
        try:
            # Example trade execution logic
            position_change = signal['position'] * self.data.loc[date, 'price']
            self.portfolio.loc[date, 'positions'] += position_change
            self.portfolio.loc[date, 'cash'] -= position_change
            self.portfolio.loc[date, 'total'] = (
                self.portfolio.loc[date, 'positions'] + self.portfolio.loc[date, 'cash']
            )
        except KeyError as e:
            logging.warning(f"Missing data for date {date}: {e}")
        except Exception as e:
            logging.error(f"Trade execution error: {e}")

    def _finalize_portfolio(self):
        self.portfolio['equity_curve'] = self.portfolio['total'].cumsum()

    def calculate_kpis(self):
        returns = self.portfolio['total'].pct_change().dropna()
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        max_drawdown = (self.portfolio['total'] / self.portfolio['total'].cummax() - 1).min()
        turnover = returns.abs().sum()
        return {
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Turnover': turnover,
            'Total Return': self.portfolio['total'].iloc[-1] - self.initial_capital
        }

    def visualize_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio['equity_curve'], label='Equity Curve')
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()

# CSV Parsing and Validation with Unix Timestamp
def load_data(file_path):
    try:
        # Read the CSV file and convert the 'time' column (Unix timestamp) to datetime
        data = pd.read_csv(file_path)

        # Convert the Unix timestamp (in 'time' column) to datetime
        data['time'] = pd.to_datetime(data['time'], unit='s')

        # Set the 'time' column as the index
        data.set_index('time', inplace=True)

        # Add a 'price' column (use 'close' price here)
        data['price'] = data['close']

        # Check if the necessary columns exist
        required_columns = ['open', 'high', 'low', 'close', 'Volume', 'Volume MA']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV must include columns: {', '.join(required_columns)}")

        return data
    except Exception as e:
        logging.error(f"Error loading CSV data: {e}")
        raise



# Strategy Import and Validation
def load_strategy(strategy_path):
    try:
        spec = importlib.util.spec_from_file_location("strategy", strategy_path)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        if not hasattr(strategy_module, 'generate_signals'):
            raise ValueError("Strategy file must implement a 'generate_signals' function.")
        return strategy_module
    except Exception as e:
        logging.error(f"Error loading strategy: {e}")
        raise

# Main Function
def main():
    parser = argparse.ArgumentParser(description='Backtesting Bot')
    parser.add_argument('--data', required=True, help='Path to the CSV data file.')
    parser.add_argument('--strategy', required=True, help='Path to the strategy Python file.')
    parser.add_argument('--output', default='output', help='Output directory for results.')
    parser.add_argument('--initial_capital', type=float, default=100000, help='Initial capital for the portfolio.')
    args = parser.parse_args()

    try:
        os.makedirs(args.output, exist_ok=True)

        # Debug: Check if data file is accessible
        print(f"Loading data from: {args.data}")
        data = load_data(args.data)
        print(f"Data loaded successfully with {len(data)} rows.")
        print(data.head())  # Show first few rows of the data

        # Debug: Check if strategy file is accessible
        print(f"Loading strategy from: {args.strategy}")
        strategy = load_strategy(args.strategy)
        print("Strategy loaded successfully.")

        # Debug: Check if the strategy generates valid signals
        test_signals = strategy.generate_signals(data)
        if 'position' in test_signals.columns:
            print("Strategy generated signals successfully.")
            print(test_signals.head())  # Show first few signal rows
        else:
            print("Error: Strategy did not generate 'position' signals.")
            return

        # Configure and run backtesting
        engine = BacktestingEngine(data, strategy, initial_capital=args.initial_capital)
        print("Running backtesting engine...")
        engine.run()
        print("Backtesting completed.")

        # Calculate KPIs
        kpis = engine.calculate_kpis()
        print("Key Performance Indicators:")
        for key, value in kpis.items():
            print(f"{key}: {value}")

        # Visualize results
        print("Visualizing results...")
        engine.visualize_results()

        # Save summary report
        summary_report_path = os.path.join(args.output, 'summary_report.txt')
        print(f"Saving summary report to {summary_report_path}...")
        with open(summary_report_path, 'w') as f:
            for key, value in kpis.items():
                f.write(f"{key}: {value}\n")
        print("Summary report saved successfully.")

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
