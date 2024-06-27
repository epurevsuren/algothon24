import numpy as np


def getMyPosition(prices):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)

    # Calculate momentum
    momentum = prices[:, -1] - prices[:, -5]

    # Calculate moving averages for trend
    short_ma = np.mean(prices[:, -5:], axis=1)
    long_ma = np.mean(prices[:, -20:], axis=1)

    # Calculate volatility
    volatility = np.std(prices[:, -20:], axis=1)

    for i in range(nInst):
        # Trade only if the momentum is strong
        if abs(momentum[i]) > volatility[i] * 0.5:
            if short_ma[i] > long_ma[i] and prices[i, -1] < short_ma[i]:
                # Calculate position size, not exceeding $10k limit
                position_size = int(
                    min(10000 / prices[i, -1], 10000 / prices[i, -1] * 0.1)
                )
                positions[i] = position_size
            elif short_ma[i] < long_ma[i] and prices[i, -1] > long_ma[i]:
                position_size = int(
                    max(-10000 / prices[i, -1], -10000 / prices[i, -1] * 0.1)
                )
                positions[i] = position_size

    return positions.astype(int)
