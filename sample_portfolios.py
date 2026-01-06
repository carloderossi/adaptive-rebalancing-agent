import random
import pandas as pd
from models import Portfolio, PortfolioPosition


# --- Asset universe (simplified but realistic) ---
EQUITY_TICKERS = [
    ("AAPL", "equity"), ("MSFT", "equity"), ("NVDA", "equity"),
    ("AMZN", "equity"), ("META", "equity"), ("TSLA", "equity"),
    ("GOOGL", "equity"), ("JPM", "equity"), ("V", "equity"),
]

BOND_TICKERS = [
    ("BND", "bonds"), ("AGG", "bonds"), ("TLT", "bonds"),
    ("IEF", "bonds"), ("LQD", "bonds"),
]

CASH_TICKERS = [
    ("CASH", "cash"),
]


def _random_price(base=100, volatility=0.2):
    """Generate a realistic price."""
    return round(base * (1 + random.uniform(-volatility, volatility)), 2)


def _random_cost_basis(price):
    """Cost basis slightly below or above current price."""
    return round(price * (1 + random.uniform(-0.15, 0.15)), 2)


def _generate_positions(tickers, total_value, min_positions=3, max_positions=6):
    """Generate random positions for a given asset class."""
    n = random.randint(min_positions, max_positions)
    chosen = random.sample(tickers, n)

    positions = []
    for ticker, asset_class in chosen:
        price = _random_price()
        weight = random.random()
        value = total_value * weight
        quantity = round(value / price, 2)
        cost_basis = _random_cost_basis(price)

        positions.append(
            PortfolioPosition(
                ticker=ticker,
                quantity=quantity,
                price=price,
                cost_basis=cost_basis,
                asset_class=asset_class,
            )
        )

    return positions


# --- Public API: generate full portfolios ---


def generate_random_portfolio(client_id="sample-client", total_value=100_000):
    """Fully random but realistic portfolio."""
    equity_value = total_value * random.uniform(0.4, 0.8)
    bond_value = total_value * random.uniform(0.1, 0.4)
    cash_value = total_value - equity_value - bond_value

    positions = []
    positions += _generate_positions(EQUITY_TICKERS, equity_value)
    positions += _generate_positions(BOND_TICKERS, bond_value)
    positions += _generate_positions(CASH_TICKERS, cash_value, min_positions=1, max_positions=1)

    return Portfolio(client_id=client_id, positions=positions)


def generate_conservative_portfolio(client_id="sample-client", total_value=100_000):
    """60% bonds, 30% equity, 10% cash."""
    positions = []
    positions += _generate_positions(EQUITY_TICKERS, total_value * 0.30)
    positions += _generate_positions(BOND_TICKERS, total_value * 0.60)
    positions += _generate_positions(CASH_TICKERS, total_value * 0.10, min_positions=1, max_positions=1)
    return Portfolio(client_id=client_id, positions=positions)


def generate_balanced_portfolio(client_id="sample-client", total_value=100_000):
    """60% equity, 30% bonds, 10% cash."""
    positions = []
    positions += _generate_positions(EQUITY_TICKERS, total_value * 0.60)
    positions += _generate_positions(BOND_TICKERS, total_value * 0.30)
    positions += _generate_positions(CASH_TICKERS, total_value * 0.10, min_positions=1, max_positions=1)
    return Portfolio(client_id=client_id, positions=positions)


def generate_aggressive_portfolio(client_id="sample-client", total_value=100_000):
    """80% equity, 15% bonds, 5% cash."""
    positions = []
    positions += _generate_positions(EQUITY_TICKERS, total_value * 0.80)
    positions += _generate_positions(BOND_TICKERS, total_value * 0.15)
    positions += _generate_positions(CASH_TICKERS, total_value * 0.05, min_positions=1, max_positions=1)
    return Portfolio(client_id=client_id, positions=positions)

def save_portfolio_to_csv(portfolio: Portfolio, path: str):
    df = portfolio.to_dataframe()
    df.to_csv(path, index=False)