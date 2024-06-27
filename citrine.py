import numpy as np


def getMyPosition(prices):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)

    # Example strategy: Simple moving average crossover
    short_window = 10
    long_window = 30

    # Calculate moving averages
    short_ma = np.mean(prices[:, -short_window:], axis=1)
    long_ma = np.mean(prices[:, -long_window:], axis=1)

    # Determine positions based on moving average crossovers
    for i in range(nInst):
        if short_ma[i] > long_ma[i]:  # Bullish signal
            positions[i] = min(
                10000 // prices[i, -1], 100
            )  # Buy, but do not exceed $10k limit
        elif short_ma[i] < long_ma[i]:  # Bearish signal
            positions[i] = max(
                -10000 // prices[i, -1], -100
            )  # Sell, but do not exceed $10k limit

    return positions.astype(int)
