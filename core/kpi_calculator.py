import numpy as np

class KPICalculator:
    @staticmethod
    def calculate_kpis(portfolio):
        returns = portfolio['portfolio_value'].pct_change().dropna()
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_drawdown = (portfolio['portfolio_value'] / portfolio['portfolio_value'].cummax() - 1).min()

        return {
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Total Return': portfolio['portfolio_value'].iloc[-1] - portfolio['portfolio_value'].iloc[0]
        }
