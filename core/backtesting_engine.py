import pandas as pd
import logging

class BacktestingEngine:
    def __init__(self, data, strategy, initial_capital=100000):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.portfolio = pd.DataFrame(index=data.index)

    def run(self):
        self.portfolio['signal'] = self.strategy.generate_signals(self.data)
        self.portfolio['cash'] = self.cash
        self.portfolio['position'] = self.position
        self.portfolio['portfolio_value'] = self.cash + self.position

        for i, row in self.portfolio.iterrows():
            signal = row['signal']
            price = self.data.loc[i, 'price']

            if signal > 0:  # Buy signal
                self.position += self.cash / price
                self.cash = 0
            elif signal < 0:  # Sell signal
                self.cash += self.position * price
                self.position = 0

            self.portfolio.loc[i, 'cash'] = self.cash
            self.portfolio.loc[i, 'position'] = self.position * price
            self.portfolio.loc[i, 'portfolio_value'] = self.cash + self.position * price

        return self.portfolio
