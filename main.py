import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)

# SETTINGS
slow_line = 19
fast_line = 9
signal_line = 7
time_frame = 7
rsi_overbought = 70
rsi_oversold = 30


def getMyPosition(prices):

    if prices.shape[1] < slow_line:  # Ensure we have enough data for MACD calculation
        return currentPos

    # Determine buy or sell signals
    buy_signals, buy_strength = identify_buy_signals(prices)
    sell_signals, sell_strength = identify_sell_signals(prices)

    # Dynamic position sizing based on signal strength
    currentPos[buy_signals] = buy_strength[buy_signals] * 10000
    currentPos[sell_signals] = sell_strength[sell_signals] * -10000

    # Specific instrument handling logic
    currentPos[:1] = 0
    currentPos[2:11] = -2000
    currentPos[13:24] = -100000
    currentPos[25:] = -7000

    # Implement risk management
    manage_risk(prices)

    return currentPos


def identify_buy_signals(prc_history):
    rsi_threshold = rsi_oversold  # RSI buy threshold

    buy_signals = np.zeros(nInst, dtype=bool)
    buy_strength = np.zeros(nInst)
    for i in range(nInst):
        if (
            prc_history.shape[1] >= slow_line
        ):  # Ensure we have enough data for MACD calculation
            prices = prc_history[i, :]
            rsi = calculate_rsi(prices)
            macd, macd_signal = calculate_macd(prices)
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
            if (
                rsi[-1] < rsi_threshold
                and macd[-1] > macd_signal[-1]
                and prices[-1] < bb_lower[-1]
            ):
                buy_signals[i] = True
                buy_strength[i] = (
                    rsi_threshold - rsi[-1]
                ) / rsi_threshold  # Normalize signal strength
    return buy_signals, buy_strength


def identify_sell_signals(prc_history):
    rsi_threshold = rsi_overbought  # RSI sell threshold

    sell_signals = np.zeros(nInst, dtype=bool)
    sell_strength = np.zeros(nInst)
    for i in range(nInst):
        if (
            prc_history.shape[1] >= slow_line
        ):  # Ensure we have enough data for MACD calculation
            prices = prc_history[i, :]
            rsi = calculate_rsi(prices)
            macd, macd_signal = calculate_macd(prices)
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
            if (
                rsi[-1] > rsi_threshold
                and macd[-1] < macd_signal[-1]
                and prices[-1] > bb_upper[-1]
            ):
                sell_signals[i] = True
                sell_strength[i] = (rsi[-1] - rsi_threshold) / (
                    100 - rsi_threshold
                )  # Normalize signal strength
    return sell_signals, sell_strength


def calculate_rsi(prices, window=time_frame):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate(([50] * (window + 1), rsi))  # Start RSI with neutral value


def calculate_macd(
    prices, short_window=fast_line, long_window=slow_line, signal_window=signal_line
):
    short_ema = pd.Series(prices).ewm(span=short_window, adjust=False).mean()
    long_ema = pd.Series(prices).ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd.values, macd_signal.values


def calculate_bollinger_bands(prices, window=time_frame, num_std_dev=1):
    sma = pd.Series(prices).rolling(window=window).mean()
    std_dev = pd.Series(prices).rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band.values, sma.values, lower_band.values


def manage_risk(prices):
    for i in range(nInst):
        if currentPos[i] > 0 and prices[i, -1] < calculate_stop_loss(
            prices[i, :], "long"
        ):
            currentPos[i] = 0  # Close long position
        elif currentPos[i] < 0 and prices[i, -1] > calculate_stop_loss(
            prices[i, :], "short"
        ):
            currentPos[i] = 0  # Close short position


def calculate_stop_loss(prices, position_type, atr_window=time_frame):
    atr = calculate_atr(prices, atr_window)
    if position_type == "long":
        return prices[-1] - atr  # Set stop loss below the current price
    elif position_type == "short":
        return prices[-1] + atr  # Set stop loss above the current price


def calculate_atr(prices, window=time_frame):
    high = pd.Series(prices).rolling(window=window).max()
    low = pd.Series(prices).rolling(window=window).min()
    close = pd.Series(prices)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.values[-1]
