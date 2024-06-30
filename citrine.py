import numpy as np


def getMyPosition(prices):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)

    # Exponential Moving Averages for MACD
    exp1 = prices[:, -12:].mean(axis=1)
    exp2 = prices[:, -26:].mean(axis=1)
    macd = exp1 - exp2
    signal = prices[:, -9:].mean(axis=1)

    # Moving averages for trend and Bollinger Bands
    ma20 = np.mean(prices[:, -20:], axis=1)
    std20 = np.std(prices[:, -20:], axis=1)
    upper_band = ma20 + (std20 * 2)
    lower_band = ma20 - (std20 * 2)

    # EMA for short and long term
    short_ema = np.mean(prices[:, -5:], axis=1)
    long_ema = np.mean(prices[:, -30:], axis=1)

    # Simple MA crossover
    short_ma = np.mean(prices[:, -10:], axis=1)
    long_ma = np.mean(prices[:, -50:], axis=1)

    # RSI calculation
    delta = np.diff(prices[:, -15:], axis=1)
    gain = np.where(delta > 0, delta, 0).mean(axis=1)
    loss = np.where(delta < 0, -delta, 0).mean(axis=1)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    for i in range(nInst):
        # Conditions to refine the signals
        is_bullish = (
            macd[i] > signal[i]
            and short_ema[i] > long_ema[i]
            and short_ma[i] > long_ma[i]
        )
        is_bearish = (
            macd[i] < signal[i]
            and short_ema[i] < long_ema[i]
            and short_ma[i] < long_ma[i]
        )

        # Adjust position based on Bollinger Bands and RSI
        if is_bullish and prices[i, -1] < lower_band[i] and rsi[i] < 30:
            position_size = int(min(10000 / prices[i, -1], 10000 / prices[i, -1] * 0.1))
            positions[i] = position_size
        elif is_bearish and prices[i, -1] > upper_band[i] and rsi[i] > 70:
            position_size = int(
                max(-10000 / prices[i, -1], -10000 / prices[i, -1] * 0.1)
            )
            positions[i] = position_size

    return positions.astype(int)
