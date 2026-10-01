import time
import numpy as np
import pandas as pd
import ccxt

from sklearn.ensemble import RandomForestClassifier


CONFIG = {
    "START_CASH": 750.0,
    "POLL_SECONDS": 300,
    "TIMEFRAME": "1h",
    "CANDLE_LIMIT": 500,
    "FEE": 0.001,

    "RISK_PER_TRADE": 0.02,
    "MAX_EXPOSURE": 0.60,

    "STOP_LOSS": 0.025,
    "TAKE_PROFIT": 0.05,

    "BUY_THRESHOLD": 0.62,
    "SELL_THRESHOLD": 0.45,

    "SYMBOLS": [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "QNT/USDT",
    ],
}


class PaperEngine:
    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot"
            }
        })

        self.cash = CONFIG["START_CASH"]
        self.start_cash = CONFIG["START_CASH"]

        self.positions = {}
        self.trades = []
        self.equity_history = []

        self.running = False
        self.last_error = ""
        self.last_update = None

        self.settings = {
            "risk_per_trade": CONFIG["RISK_PER_TRADE"],
            "max_exposure": CONFIG["MAX_EXPOSURE"],
            "stop_loss": CONFIG["STOP_LOSS"],
            "take_profit": CONFIG["TAKE_PROFIT"],
            "buy_threshold": CONFIG["BUY_THRESHOLD"],
            "sell_threshold": CONFIG["SELL_THRESHOLD"],
        }

    def fetch_df(self, symbol):
        ohlcv = self.exchange.fetch_ohlcv(
            symbol,
            timeframe=CONFIG["TIMEFRAME"],
            limit=CONFIG["CANDLE_LIMIT"]
        )

        df = pd.DataFrame(
            ohlcv,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms"
        )

        return df

    def indicators(self, df):
        df = df.copy()

        df["return_1"] = df["close"].pct_change()

        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_30"] = df["close"].rolling(30).mean()

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(
            span=9,
            adjust=False
        ).mean()

        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        df["rsi"] = 100 - (
            100 / (1 + rs)
        )

        high_low = df["high"] - df["low"]
        high_close = (
            df["high"] - df["close"].shift()
        ).abs()
        low_close = (
            df["low"] - df["close"].shift()
        ).abs()

        true_range = pd.concat(
            [
                high_low,
                high_close,
                low_close,
            ],
            axis=1
        ).max(axis=1)

        df["atr"] = true_range.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        volume_mean = df["volume"].rolling(30).mean()
        volume_std = df["volume"].rolling(30).std()

        df["volume_z"] = (
            (df["volume"] - volume_mean)
            / volume_std.replace(0, np.nan)
        )

        return df

    def model_probability(self, df):
        data = self.indicators(df)

        features = [
            "return_1",
            "sma_10",
            "sma_30",
            "macd",
            "macd_signal",
            "rsi",
            "atr_pct",
            "volume_z",
        ]

        data["target"] = (
            data["close"].shift(-6)
            > data["close"]
        ).astype(int)

        train = data.dropna().copy()

        if len(train) < 100:
            return 0.5

        X = train[features]
        y = train["target"]

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X, y)

        latest = data[features].iloc[[-1]]

        if latest.isna().any().any():
            return 0.5

        probability = model.predict_proba(
            latest
        )[0]

        classes = list(model.classes_)

        if 1 in classes:
            return float(
                probability[classes.index(1)]
            )

        return 0.5

    def price(self, symbol):
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def total_equity(self, prices=None):
        if prices is None:
            prices = {}

            for symbol in self.positions:
                try:
                    prices[symbol] = self.price(symbol)
                except Exception:
                    prices[symbol] = self.positions[
                        symbol
                    ]["entry_price"]

        value = self.cash

        for symbol, position in self.positions.items():
            current_price = prices.get(
                symbol,
                position["entry_price"]
            )

            value += (
                position["amount"]
                * current_price
            )

        return float(value)

    def update_settings(self, data):
        mapping = {
            "risk_per_trade": (
                "risk_per_trade",
                0.001,
                0.20,
            ),
            "max_exposure": (
                "max_exposure",
                0.05,
                1.00,
            ),
            "stop_loss": (
                "stop_loss",
                0.005,
                0.20,
            ),
            "take_profit": (
                "take_profit",
                0.005,
                0.50,
            ),
            "buy_threshold": (
                "buy_threshold",
                0.50,
                0.95,
            ),
            "sell_threshold": (
                "sell_threshold",
                0.05,
                0.50,
            ),
        }

        for key, info in mapping.items():
            if key not in data:
                continue

            try:
                value = float(data[key])
            except Exception:
                continue

            _, minimum, maximum = info

            value = max(
                minimum,
                min(maximum, value)
            )

            self.settings[key] = value

    def open_position(
        self,
        symbol,
        price,
        probability
    ):
        equity = self.total_equity()

        risk_budget = (
            equity
            * self.settings["risk_per_trade"]
        )

        stop_distance = (
            price
            * self.settings["stop_loss"]
        )

        if stop_distance <= 0:
            return

        amount = risk_budget / stop_distance

        maximum_position_value = (
            equity
            * self.settings["max_exposure"]
        )

        current_exposure = sum(
            p["amount"] * p["entry_price"]
            for p in self.positions.values()
        )

        available_exposure = (
            maximum_position_value
            - current_exposure
        )

        if available_exposure <= 0:
            return

        position_value = min(
            amount * price,
            available_exposure,
            self.cash / (1 + CONFIG["FEE"])
        )

        if position_value <= 5:
            return

        amount = position_value / price

        cost = (
            position_value
            * (1 + CONFIG["FEE"])
        )

        self.cash -= cost

        self.positions[symbol] = {
            "amount": amount,
            "entry_price": price,
            "entry_probability": probability,
            "opened_at": time.time(),
        }

        self.trades.append({
            "time": time.time(),
            "symbol": symbol,
            "side": "BUY",
            "price": price,
            "amount": amount,
            "reason": "AI signal",
            "probability": probability,
        })

    def close_position(
        self,
        symbol,
        price,
        reason
    ):
        position = self.positions.get(symbol)

        if not position:
            return

        gross = (
            position["amount"]
            * price
        )

        fee = gross * CONFIG["FEE"]

        proceeds = gross - fee

        entry_value = (
            position["amount"]
            * position["entry_price"]
        )

        pnl = proceeds - entry_value

        self.cash += proceeds

        self.trades.append({
            "time": time.time(),
            "symbol": symbol,
            "side": "SELL",
            "price": price,
            "amount": position["amount"],
            "reason": reason,
            "pnl": pnl,
        })

        del self.positions[symbol]

    def step(self):
        self.last_error = ""

        prices = {}

        for symbol in CONFIG["SYMBOLS"]:
            try:
                df = self.fetch_df(symbol)

                if len(df) < 100:
                    continue

                price = float(df["close"].iloc[-1])

                prices[symbol] = price

                probability = self.model_probability(df)

                position = self.positions.get(symbol)

                if position:
                    entry = position["entry_price"]

                    stop_price = (
                        entry
                        * (
                            1
                            - self.settings[
                                "stop_loss"
                            ]
                        )
                    )

                    take_profit_price = (
                        entry
                        * (
                            1
                            + self.settings[
                                "take_profit"
                            ]
                        )
                    )

                    if price <= stop_price:
                        self.close_position(
                            symbol,
                            price,
                            "stop loss"
                        )

                    elif price >= take_profit_price:
                        self.close_position(
                            symbol,
                            price,
                            "take profit"
                        )

                    elif (
                        probability
                        < self.settings[
                            "sell_threshold"
                        ]
                    ):
                        self.close_position(
                            symbol,
                            price,
                            "AI exit"
                        )

                else:
                    if (
                        probability
                        >= self.settings[
                            "buy_threshold"
                        ]
                    ):
                        self.open_position(
                            symbol,
                            price,
                            probability
                        )

            except Exception as exc:
                self.last_error = (
                    f"{symbol}: {exc}"
                )

        equity = self.total_equity(prices)

        self.equity_history.append({
            "time": time.time(),
            "equity": equity,
        })

        if len(self.equity_history) > 300:
            self.equity_history = (
                self.equity_history[-300:]
            )

        self.last_update = time.time()

        return self.snapshot()

    def snapshot(self):
        prices = {}

        for symbol, position in self.positions.items():
            try:
                prices[symbol] = self.price(symbol)
            except Exception:
                prices[symbol] = position["entry_price"]

        equity = self.total_equity(prices)

        unrealized = 0.0

        for symbol, position in self.positions.items():
            current = prices.get(
                symbol,
                position["entry_price"]
            )

            unrealized += (
                (
                    current
                    - position["entry_price"]
                )
                * position["amount"]
            )

        realized = sum(
            trade.get("pnl", 0)
            for trade in self.trades
            if trade["side"] == "SELL"
        )

        pnl = equity - self.start_cash

        market_data = []

        for symbol in CONFIG["SYMBOLS"]:
            try:
                df = self.fetch_df(symbol)

                price = float(
                    df["close"].iloc[-1]
                )

                probability = self.model_probability(
                    df
                )

                market_data.append({
                    "symbol": symbol,
                    "price": price,
                    "probability": probability,
                })

            except Exception as exc:
                market_data.append({
                    "symbol": symbol,
                    "price": None,
                    "probability": None,
                    "error": str(exc),
                })

        return {
            "running": self.running,
            "paper_only": True,
            "cash": self.cash,
            "equity": equity,
            "pnl": pnl,
            "realized": realized,
            "unrealized": unrealized,
            "positions": self.positions,
            "trades": self.trades[-30:],
            "equity_history": self.equity_history,
            "markets": market_data,
            "settings": self.settings,
            "last_error": self.last_error,
            "last_update": self.last_update,
        }

    def reset(self):
        self.cash = CONFIG["START_CASH"]
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.running = False
        self.last_error = ""
        self.last_update = None
