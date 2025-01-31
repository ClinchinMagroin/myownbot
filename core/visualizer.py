import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_performance(portfolio):
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio.index, portfolio['portfolio_value'], label='Portfolio Value')
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_drawdown(portfolio):
        drawdown = portfolio['portfolio_value'] / portfolio['portfolio_value'].cummax() - 1
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio.index, drawdown, label='Drawdown', color='red')
        plt.title('Drawdown Over Time')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.show()
