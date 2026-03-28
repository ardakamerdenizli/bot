import json
import math
import os
import subprocess
import sys
import time
import warnings
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# Pandas future warning spamini bastir (kodda asil fixler de uygulandi).
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
CONFIG_PATH = ROOT / "config.json"

# config.json üzerine yazılsa bile kod içindeki değerler geçerli olsun (manuel giriş yok).
FORCED_CONFIG_KEYS = (
    "mt5_login",
    "mt5_password",
    "mt5_server",
    "telegram_token",
    "telegram_chat_id",
    "max_open_positions",
    "max_daily_trades",
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "bot_name": "XM AI Rebuild",
    "symbol": "GOLD",
    "fallback_symbols": ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD."],
    "entry_timeframe": "M1",
    "mid_timeframe": "M5",
    "trend_timeframe": "M15",
    "history_bars": 800,
    "loop_seconds": 0.5,
    "paper_trade": False,
    "live_trading_enabled": True,
    "mt5_path": "",
    "mt5_login": 336059530,
    "mt5_password": "ARDAkAMER!?2814",
    "mt5_server": "XMGlobal-MT5 9",
    "magic_number": 26832602,
    "deviation": 20,
    "allow_long": True,
    "allow_short": True,
    "score_threshold": 1.2,
    "confidence_threshold": 0.5,
    "max_open_positions": 5,
    "risk_per_trade_pct": 0.5,
    "sl_atr_mult": 1.6,
    "tp_atr_mult": 2.2,
    "trailing_atr_mult": 1.0,
    "break_even_rr": 1.0,
    "partial_close_rr": 1.6,
    "partial_close_ratio": 0.4,
    "max_spread_points": 120,
    "max_daily_loss_pct": 4.0,
    "max_daily_trades": 999,
    "order_cooldown_seconds": 10,
    "max_consecutive_losses": 6,
    "context_weight": 1.0,
    "memory_weight": 0.8,
    "fake_signal_penalty": 1.3,
    "manual_blackout_hours": [],
    "telegram_token": "8369134789:AAG-ffAZl99AR28Lpm0CRkoIRYdzfh7aSv4",
    "telegram_chat_id": "1861806582",
    "preferred_filling_mode": "auto",
    "filling_mode_fallbacks": ["IOC", "RETURN", "FOK"],
    "near_tp_factor": 0.88,
    "tp_extension_atr_mult": 0.9,
    "tp_extension_min_confidence": 0.62,
    "early_take_profit_progress": 0.65,
    "tp_extension_max_steps": 3,
    "break_even_progress": 0.5,
    "trailing_start_progress": 0.6,
    "trailing_atr_mult": 0.8,
    "partial_close_enabled": True,
    "news_filter_enabled": True,
    "news_confidence_penalty": 0.15,
    "high_impact_keywords": ["NFP", "CPI", "FOMC", "Powell", "rate decision", "inflation"],
    "news_blackout_minutes": 25,
    "adaptive_memory_decay": 0.98,
    "ensemble_enabled": True,
    "online_learning_enabled": True,
    "retrain_every_seconds": 300,
    "order_log_verbose": False,
    "walkforward_splits": 5,
    "min_train_rows": 220,
    "regime_adx_threshold": 22,
    "max_trade_duration_minutes": 180,
    "volatility_exit_atr_mult": 2.6,
    "panel_host": "127.0.0.1",
    "panel_port": 5000,
    "auto_start_panel": True,
    "auto_open_panel_browser": True,
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    defaults = {
        "trade_journal.json": [],
        "trade_memory.json": {"setup_stats": {}, "last_update": None},
        "performance.json": {
            "equity_peak": 0.0,
            "today_date": datetime.now().strftime("%Y-%m-%d"),
            "today_pnl": 0.0,
            "today_trades": 0,
            "consecutive_losses": 0,
        },
        "runtime_state.json": {},
        "control.json": {"enabled": True, "running": True, "mode_override": None},
        "news_events.json": [],
        "backtest_report.json": {},
        "walkforward_report.json": {},
        "improvement_suggestions.json": {},
        "rl_state.json": {"q_table": {}, "alpha": 0.2, "gamma": 0.9, "epsilon": 0.15},
    }
    for name, value in defaults.items():
        path = DATA_DIR / name
        if not path.exists():
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


def load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_runtime_config() -> Dict[str, Any]:
    file_cfg = load_json(CONFIG_PATH, {})
    merged = {**DEFAULT_CONFIG, **file_cfg}
    for k in FORCED_CONFIG_KEYS:
        merged[k] = DEFAULT_CONFIG[k]
    return merged


def log_line(name: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    with (LOG_DIR / f"{name}.log").open("a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())


def tf_to_mt5(tf: str) -> int:
    mapping = {
        "M1": mt5.TIMEFRAME_M1 if mt5 else 1,
        "M5": mt5.TIMEFRAME_M5 if mt5 else 5,
        "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
        "M30": mt5.TIMEFRAME_M30 if mt5 else 30,
        "H1": mt5.TIMEFRAME_H1 if mt5 else 16385,
        "H4": mt5.TIMEFRAME_H4 if mt5 else 16388,
    }
    return mapping.get(tf.upper(), mapping["M5"])


@dataclass
class Decision:
    action: str
    score: float
    confidence: float
    reason: str
    trend: str
    regime: str
    pattern: str


class Telegram:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.token = str(config.get("telegram_token", "")).strip()
        self.chat_id = str(config.get("telegram_chat_id", "")).strip()

    def send(self, msg: str) -> None:
        if not self.token or not self.chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": msg},
                timeout=5,
            )
        except Exception:
            pass


class BrokerMT5:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.connected = False
        self.active_symbol = str(config.get("symbol", "XAUUSD"))

    def connect(self) -> bool:
        if mt5 is None:
            log_line("error", "MetaTrader5 modülü yok.")
            return False
        kwargs = {}
        if self.config.get("mt5_path"):
            kwargs["path"] = self.config["mt5_path"]
        if not mt5.initialize(**kwargs):
            log_line("error", f"MT5 initialize hata: {mt5.last_error()}")
            return False
        login = int(self.config.get("mt5_login", 0) or 0)
        pwd = str(self.config.get("mt5_password", ""))
        server = str(self.config.get("mt5_server", ""))
        if login and pwd and server and not mt5.login(login, password=pwd, server=server):
            log_line("error", f"MT5 login hata: {mt5.last_error()}")
            return False
        self.connected = True
        self.active_symbol = self.detect_symbol()
        return True

    def ensure_connection(self) -> bool:
        info = mt5.account_info() if mt5 else None
        if info is None:
            self.connected = False
        if not self.connected:
            return self.connect()
        return True

    def detect_symbol(self) -> str:
        forced = str(self.config.get("symbol", "GOLD")).strip() or "GOLD"
        symbols = [forced] + [s for s in list(self.config.get("fallback_symbols", [])) if str(s).strip() and str(s).strip() != forced]
        for sym in symbols:
            if mt5.symbol_select(sym, True):
                tick = mt5.symbol_info_tick(sym)
                if tick:
                    log_line("system", f"Aktif sembol: {sym}")
                    return sym
        # XM gibi brokerlarda GOLD.suffix formatlarını otomatik bul.
        all_symbols = mt5.symbols_get() or []
        gold_like: List[str] = []
        for s in all_symbols:
            name = str(getattr(s, "name", "")).upper()
            if "GOLD" in name or "XAUUSD" in name:
                gold_like.append(str(getattr(s, "name", "")))
        for sym in gold_like:
            if mt5.symbol_select(sym, True):
                tick = mt5.symbol_info_tick(sym)
                if tick:
                    log_line("system", f"Oto GOLD eşleşti: {sym}")
                    return sym
        log_line("error", "Hiçbir sembol seçilemedi, ilk sembol kullanılıyor.")
        return str(symbols[0])

    def symbol_info(self) -> Dict[str, Any]:
        info = mt5.symbol_info(self.active_symbol)
        return info._asdict() if info else {}

    def tick(self) -> Dict[str, float]:
        t = mt5.symbol_info_tick(self.active_symbol)
        if not t:
            return {}
        d = t._asdict()
        return {"bid": float(d.get("bid", 0.0)), "ask": float(d.get("ask", 0.0))}

    def candles(self, timeframe: str, bars: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.active_symbol, tf_to_mt5(timeframe), 0, bars)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def positions(self) -> List[Any]:
        p = mt5.positions_get(symbol=self.active_symbol)
        return list(p) if p else []

    def can_trade_now(self) -> Tuple[bool, str]:
        """Sembol ve terminal işlem açık mı (piyasa kapalıysa genelde tick yok veya emir reddedilir)."""
        if mt5 is None:
            return False, "no_mt5"
        if not mt5.symbol_select(self.active_symbol, True):
            return False, "symbol_select_fail"
        info = mt5.symbol_info(self.active_symbol)
        if not info:
            return False, "no_symbol_info"
        tm = int(getattr(info, "trade_mode", 0))
        disabled = int(getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0))
        if tm == disabled:
            return False, "symbol_trade_disabled"
        tick = mt5.symbol_info_tick(self.active_symbol)
        if not tick:
            return False, "no_tick_or_market_closed"
        term = mt5.terminal_info()
        if term is not None and not bool(getattr(term, "trade_allowed", True)):
            return False, "terminal_trade_blocked"
        return True, "ok"

    def close_position(self, pos: Any, reason: str) -> bool:
        tick = self.tick()
        side = mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL
        price = tick.get("ask") if side == mt5.ORDER_TYPE_BUY else tick.get("bid")
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.active_symbol,
            "position": int(pos.ticket),
            "volume": float(pos.volume),
            "type": side,
            "price": float(price),
            "deviation": int(self.config.get("deviation", 20)),
            "magic": int(self.config.get("magic_number", 0)),
            "comment": reason[:28],
        }
        ok, data = self._send_with_filling(req)
        if not ok:
            log_line("error", f"Pozisyon kapanamadı: {data}")
        return ok

    def partial_close_position(self, pos: Any, ratio: float, reason: str) -> bool:
        ratio = max(0.05, min(0.95, float(ratio)))
        vol = float(getattr(pos, "volume", 0.0))
        if vol <= 0:
            return False
        info = self.symbol_info()
        step = float(info.get("volume_step", 0.01) or 0.01)
        vmin = float(info.get("volume_min", 0.01) or 0.01)
        close_vol = max(vmin, math.floor((vol * ratio) / step) * step)
        if close_vol >= vol:
            close_vol = max(vmin, vol - step)
        if close_vol <= 0:
            return False
        tick = self.tick()
        side = mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL
        price = tick.get("ask") if side == mt5.ORDER_TYPE_BUY else tick.get("bid")
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.active_symbol,
            "position": int(pos.ticket),
            "volume": float(close_vol),
            "type": side,
            "price": float(price),
            "deviation": int(self.config.get("deviation", 20)),
            "magic": int(self.config.get("magic_number", 0)),
            "comment": reason[:28],
        }
        ok, data = self._send_with_filling(req)
        if not ok:
            log_line("error", f"Partial close fail: {data}")
        return ok

    def modify_position_sltp(self, pos: Any, sl: float, tp: float) -> bool:
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.active_symbol,
            "position": int(pos.ticket),
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(self.config.get("magic_number", 0)),
        }
        res = mt5.order_send(req)
        if res is None:
            log_line("error", f"SLTP modify fail: {mt5.last_error()}")
            return False
        d = res._asdict()
        ok = int(d.get("retcode", -1)) in (10008, 10009)
        if not ok:
            log_line("error", f"SLTP modify retcode fail: {d}")
        return ok

    def _fill_const(self, name: str) -> Optional[int]:
        n = name.upper()
        if n == "IOC":
            return getattr(mt5, "ORDER_FILLING_IOC", None)
        if n == "RETURN":
            return getattr(mt5, "ORDER_FILLING_RETURN", None)
        if n == "FOK":
            return getattr(mt5, "ORDER_FILLING_FOK", None)
        return None

    def _filling_candidates(self) -> List[int]:
        out: List[int] = []
        pref = str(self.config.get("preferred_filling_mode", "auto")).upper()
        cfg = list(self.config.get("filling_mode_fallbacks", ["IOC", "RETURN", "FOK"]))
        if pref != "AUTO":
            c = self._fill_const(pref)
            if c is not None:
                out.append(c)
        info = mt5.symbol_info(self.active_symbol)
        if info and getattr(info, "filling_mode", None) is not None:
            out.append(int(info.filling_mode))
        for n in cfg:
            c = self._fill_const(str(n))
            if c is not None:
                out.append(c)
        return list(dict.fromkeys(out))

    def _send_with_filling(self, req: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        attempts: List[Dict[str, Any]] = []
        for f in self._filling_candidates():
            # 1) normal istek
            # 2) type_time ekleyerek istek
            # 3) SL/TP olmadan açıp sonra modify (XM'de invalid stops için)
            variants: List[Dict[str, Any]] = []
            base = dict(req)
            base["type_filling"] = f
            variants.append(base)
            with_time = dict(base)
            with_time["type_time"] = getattr(mt5, "ORDER_TIME_GTC", 0)
            variants.append(with_time)
            no_stops = dict(with_time)
            no_stops.pop("sl", None)
            no_stops.pop("tp", None)
            variants.append(no_stops)

            for cur in variants:
                chk = mt5.order_check(cur)
                if chk and getattr(chk, "retcode", 0) not in (0, 10009, 10008):
                    attempts.append({"filling": f, "check_retcode": int(chk.retcode), "variant": list(cur.keys())})
                    continue
                res = mt5.order_send(cur)
                if res is None:
                    attempts.append({"filling": f, "error": str(mt5.last_error()), "variant": list(cur.keys())})
                    continue
                d = res._asdict()
                d["type_filling_used"] = f
                if int(d.get("retcode", -1)) in (10008, 10009):
                    # no_stops varyantı ile açıldıysa, SL/TP sonradan set etmeyi dene.
                    if "sl" not in cur and ("sl" in req or "tp" in req):
                        pos_ticket = int(d.get("order") or d.get("deal") or 0)
                        if pos_ticket > 0:
                            self._set_stops_after_open(pos_ticket, float(req.get("sl", 0.0)), float(req.get("tp", 0.0)))
                    return True, d
                attempts.append(d)
        return False, {"attempts": attempts}

    def _set_stops_after_open(self, ticket: int, sl: float, tp: float) -> None:
        if ticket <= 0:
            return
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.active_symbol,
            "position": ticket,
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(self.config.get("magic_number", 0)),
        }
        res = mt5.order_send(req)
        if res is None:
            log_line("error", f"SLTP set fail: {mt5.last_error()}")
            return
        d = res._asdict()
        if int(d.get("retcode", -1)) not in (10008, 10009):
            log_line("error", f"SLTP retcode fail: {d}")

    def open_market(self, side: str, volume: float, sl: float, tp: float, comment: str) -> Tuple[bool, Dict[str, Any]]:
        tick = self.tick()
        if not tick:
            return False, {"error": "tick yok"}
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick["ask"] if side == "BUY" else tick["bid"]
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.active_symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(self.config.get("deviation", 20)),
            "magic": int(self.config.get("magic_number", 0)),
            "comment": comment[:28],
        }
        return self._send_with_filling(req)


class FeatureEngine:
    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        close, high, low, vol = out["close"], out["high"], out["low"], out["tick_volume"]
        out["ema_20"] = close.ewm(span=20, adjust=False).mean()
        out["ema_50"] = close.ewm(span=50, adjust=False).mean()
        out["sma_20"] = close.rolling(20).mean()
        delta = close.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        rs = pd.Series(up).rolling(14).mean() / (pd.Series(down).rolling(14).mean() + 1e-9)
        out["rsi"] = 100 - (100 / (1 + rs))
        tr = np.maximum((high - low).values, np.maximum((high - close.shift(1)).abs().values, (low - close.shift(1)).abs().values))
        out["atr"] = pd.Series(tr, index=out.index).rolling(14).mean()
        out["bb_mid"] = close.rolling(20).mean()
        out["bb_std"] = close.rolling(20).std()
        out["bb_u"] = out["bb_mid"] + 2 * out["bb_std"]
        out["bb_l"] = out["bb_mid"] - 2 * out["bb_std"]
        out["momentum"] = close - close.shift(10)
        out["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["stoch_k"] = ((close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min() + 1e-9)) * 100
        out["stoch_d"] = out["stoch_k"].rolling(3).mean()
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr_raw = np.maximum((high - low).values, np.maximum((high - close.shift(1)).abs().values, (low - close.shift(1)).abs().values))
        tr_s = pd.Series(tr_raw, index=out.index).rolling(14).sum() + 1e-9
        plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(14).sum() / tr_s
        minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(14).sum() / tr_s
        out["adx"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)).rolling(14).mean()
        out["vwap"] = ((close * vol).cumsum() / (vol.cumsum() + 1e-9))
        out["obv"] = (np.sign(close.diff().fillna(0)) * vol).cumsum()
        tp_price = (high + low + close) / 3.0
        out["cci"] = (tp_price - tp_price.rolling(20).mean()) / (0.015 * (tp_price.rolling(20).std() + 1e-9))
        # Ichimoku (kismi)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        out["ichimoku_tenkan"] = tenkan
        out["ichimoku_kijun"] = kijun
        out["ichimoku_a"] = ((tenkan + kijun) / 2).shift(26)
        out["ichimoku_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        # Supertrend (kismi)
        hl2 = (high + low) / 2
        atr10 = out["atr"].rolling(10).mean().fillna(out["atr"])
        upper = hl2 + 2.0 * atr10
        lower = hl2 - 2.0 * atr10
        out["supertrend_dir"] = np.where(close > upper.shift(1), 1, np.where(close < lower.shift(1), -1, 0))
        # PSAR benzeri hizli proxy
        out["psar_proxy"] = close.shift(1) + (close.shift(1) - close.shift(2)).fillna(0) * 0.2
        # Volatility & range
        out["volatility_pct"] = out["atr"] / (close + 1e-9)
        out["range_score"] = (high.rolling(20).max() - low.rolling(20).min()) / (out["atr"] + 1e-9)
        out["dist_support"] = close - low.rolling(20).min()
        out["dist_resistance"] = high.rolling(20).max() - close
        out["body"] = (out["close"] - out["open"]).abs()
        out["wick_u"] = out["high"] - out[["close", "open"]].max(axis=1)
        out["wick_l"] = out[["close", "open"]].min(axis=1) - out["low"]
        out["doji"] = (out["body"] / (out["atr"] + 1e-9)) < 0.1
        prev_open = out["open"].shift(1)
        prev_close = out["close"].shift(1)
        out["bull_engulf"] = (out["close"] > out["open"]) & (prev_close < prev_open) & (out["close"] >= prev_open) & (out["open"] <= prev_close)
        out["bear_engulf"] = (out["close"] < out["open"]) & (prev_close > prev_open) & (out["open"] >= prev_close) & (out["close"] <= prev_open)
        out["inside_bar"] = (out["high"] < out["high"].shift(1)) & (out["low"] > out["low"].shift(1))
        out["breakout_up"] = out["close"] > out["high"].rolling(20).max().shift(1)
        out["breakout_down"] = out["close"] < out["low"].rolling(20).min().shift(1)
        out["candle_strength"] = (out["body"] / (out["atr"] + 1e-9)).clip(lower=0, upper=3)
        # BOS / CHoCH ve order-block benzeri işaretler (pratik proxy)
        hh = out["high"].rolling(10).max().shift(1)
        ll = out["low"].rolling(10).min().shift(1)
        out["bos_up"] = out["close"] > hh
        out["bos_down"] = out["close"] < ll
        out["swing_hi"] = out["high"] == out["high"].rolling(5, center=True).max()
        out["swing_lo"] = out["low"] == out["low"].rolling(5, center=True).min()
        out["choch_up"] = out["bos_up"] & out["swing_lo"].shift(1, fill_value=False).astype(bool)
        out["choch_down"] = out["bos_down"] & out["swing_hi"].shift(1, fill_value=False).astype(bool)
        out["order_block_bull"] = (out["close"] > out["open"]) & (out["low"] <= out["low"].rolling(8).min().shift(1))
        out["order_block_bear"] = (out["close"] < out["open"]) & (out["high"] >= out["high"].rolling(8).max().shift(1))
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
        out[num_cols] = out[num_cols].ffill().bfill()
        return out.dropna().reset_index(drop=True)


class Memory:
    def __init__(self) -> None:
        self.path = DATA_DIR / "trade_memory.json"
        self.data = load_json(self.path, {"setup_stats": {}, "last_update": None})

    def score_bias(self, key: str) -> float:
        row = self.data.get("setup_stats", {}).get(key, {"win": 0, "loss": 0})
        total = row["win"] + row["loss"]
        if total < 3:
            return 0.0
        wr = row["win"] / total
        return (wr - 0.5) * 2.0

    def decay(self, factor: float) -> None:
        factor = max(0.90, min(0.999, float(factor)))
        stats = self.data.get("setup_stats", {})
        for _, row in stats.items():
            row["win"] = float(row.get("win", 0.0)) * factor
            row["loss"] = float(row.get("loss", 0.0)) * factor
        self.data["last_update"] = datetime.now().isoformat()
        save_json(self.path, self.data)

    def learn(self, key: str, won: bool) -> None:
        stats = self.data.setdefault("setup_stats", {})
        row = stats.setdefault(key, {"win": 0, "loss": 0})
        row["win" if won else "loss"] += 1
        self.data["last_update"] = datetime.now().isoformat()
        save_json(self.path, self.data)


class NewsSentiment:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.path = DATA_DIR / "news_events.json"

    def evaluate(self, now: datetime) -> Dict[str, Any]:
        if not self.config.get("news_filter_enabled", True):
            return {"blocked": False, "sentiment_bias": 0.0, "label": "news_off"}
        events = load_json(self.path, [])
        blackout = int(self.config.get("news_blackout_minutes", 25))
        keywords = [str(k).lower() for k in self.config.get("high_impact_keywords", [])]
        blocked = False
        score = 0.0
        label = "clear"
        clusters: Dict[str, int] = {"inflation": 0, "rates": 0, "risk": 0, "labor": 0, "other": 0}
        for ev in events:
            try:
                ts = datetime.fromisoformat(str(ev.get("time")))
            except Exception:
                continue
            delta_min = abs((ts - now).total_seconds()) / 60.0
            title = str(ev.get("title", "")).lower()
            impact = str(ev.get("impact", "low")).lower()
            if "inflation" in title or "cpi" in title or "ppi" in title:
                clusters["inflation"] += 1
            elif "rate" in title or "fomc" in title or "powell" in title:
                clusters["rates"] += 1
            elif "risk" in title or "war" in title or "geopolitic" in title:
                clusters["risk"] += 1
            elif "nfp" in title or "unemployment" in title or "payroll" in title:
                clusters["labor"] += 1
            else:
                clusters["other"] += 1
            if delta_min <= blackout and any(k in title for k in keywords):
                blocked = True
                label = "high_impact_news"
            if impact == "high":
                score -= 0.10
            elif impact == "medium":
                score -= 0.04
            # Basit NLP/keyword sentiment
            positive_words = ["dovish", "soft landing", "cooling inflation", "risk-on", "growth"]
            negative_words = ["hawkish", "hot inflation", "risk-off", "recession", "tightening"]
            if any(w in title for w in positive_words):
                score += 0.08
            if any(w in title for w in negative_words):
                score -= 0.08
        dominant_cluster = max(clusters, key=clusters.get) if clusters else "other"
        return {"blocked": blocked, "sentiment_bias": float(score), "label": label, "cluster": dominant_cluster, "clusters": clusters}


class EnsembleAI:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.models = {
            "rf": RandomForestClassifier(n_estimators=120, random_state=42, max_depth=7),
            "gb": GradientBoostingClassifier(random_state=42),
            "lr": LogisticRegression(max_iter=500),
        }
        self.weights = {"rf": 0.45, "gb": 0.35, "lr": 0.20}
        self.is_ready = False
        self.feature_cols: List[str] = []
        self.feature_fill_values: Dict[str, float] = {}

    def _build_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        work = df.copy()
        work["y"] = np.where(work["close"].shift(-3) > work["close"], 1, 0)
        cols = [
            "rsi",
            "macd",
            "macd_signal",
            "adx",
            "stoch_k",
            "stoch_d",
            "volatility_pct",
            "candle_strength",
            "momentum",
            "dist_support",
            "dist_resistance",
        ]
        cols = [c for c in cols if c in work.columns]
        base = work[cols + ["y"]].copy()
        base[cols] = base[cols].replace([np.inf, -np.inf], np.nan)
        # Zaman serisinde veri kaybini azaltmak icin once doldur, sonra kalanlari at.
        base[cols] = base[cols].ffill().bfill()
        clean = base.dropna()
        return clean[cols], clean["y"]

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.config.get("ensemble_enabled", True):
            return {"trained": False, "reason": "disabled"}
        X, y = self._build_xy(df)
        if len(X) < int(self.config.get("min_train_rows", 220)):
            return {"trained": False, "reason": "not_enough_rows"}
        self.feature_cols = list(X.columns)
        self.feature_fill_values = {c: float(X[c].median()) for c in self.feature_cols}
        X_safe = X.copy()
        for c in self.feature_cols:
            X_safe[c] = X_safe[c].replace([np.inf, -np.inf], np.nan).fillna(self.feature_fill_values[c])
        Xs = self.scaler.fit_transform(X_safe)
        for m in self.models.values():
            m.fit(Xs, y)
        self.is_ready = True
        return {"trained": True, "rows": int(len(X))}

    def predict(self, row: pd.Series) -> Tuple[int, float]:
        if not self.is_ready or not self.feature_cols:
            return 1, 0.55
        values: List[float] = []
        for c in self.feature_cols:
            v = row.get(c, self.feature_fill_values.get(c, 0.0))
            try:
                fv = float(v)
            except Exception:
                fv = self.feature_fill_values.get(c, 0.0)
            if not np.isfinite(fv):
                fv = self.feature_fill_values.get(c, 0.0)
            values.append(float(fv))
        row_df = pd.DataFrame([values], columns=self.feature_cols, dtype=float)
        xs = self.scaler.transform(row_df)
        prob = 0.0
        for name, model in self.models.items():
            p = float(model.predict_proba(xs)[0][1])
            prob += p * float(self.weights.get(name, 0.0))
        pred = 1 if prob >= 0.5 else 0
        conf = abs(prob - 0.5) * 2.0
        return pred, max(0.5, min(0.99, conf))

    def walkforward(self, df: pd.DataFrame) -> Dict[str, Any]:
        X, y = self._build_xy(df)
        if len(X) < 200:
            return {"ok": False, "reason": "not_enough_rows"}
        tscv = TimeSeriesSplit(n_splits=max(2, int(self.config.get("walkforward_splits", 5))))
        scores: List[float] = []
        for tr, te in tscv.split(X):
            xtr, xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            if len(xtr) < 80 or len(xte) < 20:
                continue
            xtr = xtr.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            xte = xte.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            xs_tr = self.scaler.fit_transform(xtr)
            self.models["rf"].fit(xs_tr, ytr)
            xs_te = self.scaler.transform(xte)
            pred = self.models["rf"].predict(xs_te)
            acc = float((pred == yte.values).mean())
            scores.append(acc)
        return {"ok": True, "splits": len(scores), "mean_acc": float(np.mean(scores)) if scores else 0.0}


class ReinforcementPolicy:
    def __init__(self) -> None:
        self.path = DATA_DIR / "rl_state.json"
        self.state = load_json(self.path, {"q_table": {}, "alpha": 0.2, "gamma": 0.9, "epsilon": 0.15})

    def _key(self, regime: str, trend: str, pattern: str) -> str:
        return f"{regime}|{trend}|{pattern}"

    def choose_bias(self, regime: str, trend: str, pattern: str) -> float:
        key = self._key(regime, trend, pattern)
        q = self.state.get("q_table", {}).get(key, {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0})
        if np.random.rand() < float(self.state.get("epsilon", 0.15)):
            act = np.random.choice(["BUY", "SELL", "HOLD"])
        else:
            act = max(q, key=q.get)
        if act == "BUY":
            return 0.22
        if act == "SELL":
            return -0.22
        return 0.0

    def update(self, regime: str, trend: str, pattern: str, action: str, reward: float) -> None:
        key = self._key(regime, trend, pattern)
        table = self.state.setdefault("q_table", {})
        row = table.setdefault(key, {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0})
        alpha = float(self.state.get("alpha", 0.2))
        row[action] = row.get(action, 0.0) + alpha * (reward - row.get(action, 0.0))
        save_json(self.path, self.state)


class StrategyEvolution:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.path = DATA_DIR / "improvement_suggestions.json"

    def evolve(self, perf: Dict[str, Any]) -> Dict[str, Any]:
        suggestions: Dict[str, Any] = {"time": datetime.now().isoformat(), "changes": []}
        pf = float(perf.get("profit_factor", 0.0))
        losses = int(perf.get("consecutive_losses", 0))
        wr = int(perf.get("wins", 0)) / max(1, int(perf.get("wins", 0)) + int(perf.get("losses", 0)))
        if pf < 1.05 or losses >= 4:
            suggestions["changes"].append({"param": "score_threshold", "action": "increase", "delta": 0.1})
            suggestions["changes"].append({"param": "risk_per_trade_pct", "action": "decrease", "delta": 0.05})
        if wr > 0.58 and pf > 1.2:
            suggestions["changes"].append({"param": "tp_atr_mult", "action": "increase", "delta": 0.1})
        if wr < 0.45:
            suggestions["changes"].append({"param": "fake_signal_penalty", "action": "increase", "delta": 0.1})
        save_json(self.path, suggestions)
        return suggestions


class Bot:
    def __init__(self) -> None:
        ensure_dirs()
        self.config = build_runtime_config()
        self.broker = BrokerMT5(self.config)
        self.telegram = Telegram(self.config)
        self.memory = Memory()
        self.news = NewsSentiment(self.config)
        self.ai = EnsembleAI(self.config)
        self.rl = ReinforcementPolicy()
        self.evolution = StrategyEvolution(self.config)
        self.last_status = datetime.min
        self.tp_extended_tickets: set[int] = set()
        self.tp_extension_steps: Dict[int, int] = {}
        self.partial_closed_tickets: set[int] = set()
        self.data_cache: Dict[str, Any] = {"ts": datetime.min, "entry": pd.DataFrame(), "mid": pd.DataFrame(), "trend": pd.DataFrame()}
        self.last_train_at: datetime = datetime.min
        self.last_order_attempt_ts: float = 0.0
        self.last_order_fail_log_ts: float = 0.0
        self.last_trade_block_log_ts: float = 0.0
        self._panel_started: bool = False

    def reload_config(self) -> None:
        self.config = build_runtime_config()
        self.broker.config = self.config
        self.telegram = Telegram(self.config)
        self.news = NewsSentiment(self.config)
        self.ai.config = self.config

    def market_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if datetime.now() - self.data_cache["ts"] <= timedelta(seconds=1):
            return self.data_cache["entry"], self.data_cache["mid"], self.data_cache["trend"]
        bars = int(self.config.get("history_bars", 800))
        e = FeatureEngine.compute(self.broker.candles(self.config["entry_timeframe"], bars))
        m = FeatureEngine.compute(self.broker.candles(self.config["mid_timeframe"], bars))
        t = FeatureEngine.compute(self.broker.candles(self.config["trend_timeframe"], bars))
        self.data_cache = {"ts": datetime.now(), "entry": e, "mid": m, "trend": t}
        return e, m, t

    def trend(self, tdf: pd.DataFrame) -> str:
        if tdf.empty:
            return "flat"
        row = tdf.iloc[-1]
        if row["ema_20"] > row["ema_50"] and row["rsi"] > 55:
            return "up"
        if row["ema_20"] < row["ema_50"] and row["rsi"] < 45:
            return "down"
        return "flat"

    def regime(self, mdf: pd.DataFrame) -> str:
        if mdf.empty:
            return "unknown"
        row = mdf.iloc[-1]
        width = (row["bb_u"] - row["bb_l"]) / max(row["close"], 1e-9)
        if width < 0.003:
            return "range"
        if abs(row["momentum"]) > row["atr"] * 0.8 and float(row.get("adx", 0.0)) > float(self.config.get("regime_adx_threshold", 22)):
            return "trend"
        return "mixed"

    def pattern(self, row: pd.Series) -> str:
        body = float(row.get("body", 0.0))
        wu = float(row.get("wick_u", 0.0))
        wl = float(row.get("wick_l", 0.0))
        if bool(row.get("bull_engulf", False)):
            return "bull_engulf"
        if bool(row.get("bear_engulf", False)):
            return "bear_engulf"
        if bool(row.get("doji", False)):
            return "doji"
        if bool(row.get("inside_bar", False)):
            return "inside_bar"
        if body > 0 and wl > body * 1.8 and wu < body * 0.7:
            return "bull_pinbar"
        if body > 0 and wu > body * 1.8 and wl < body * 0.7:
            return "bear_pinbar"
        return "none"

    def pattern_confidence(self, row: pd.Series, pattern: str) -> float:
        strength = float(row.get("candle_strength", 0.0))
        conf = 0.5 + min(0.4, strength / 8.0)
        if pattern in ("bull_engulf", "bear_engulf", "bull_pinbar", "bear_pinbar"):
            conf += 0.08
        if pattern == "doji":
            conf -= 0.05
        return max(0.35, min(0.95, conf))

    def context_score(self) -> float:
        # Basit context: DXY/EURUSD gibi semboller varsa kısa momentum skorunu toplar.
        score = 0.0
        ctx = self.config.get("context_symbols", {})
        for _, meta in ctx.items():
            sym = str(meta.get("symbol", "")).strip()
            if not sym:
                continue
            try:
                mt5.symbol_select(sym, True)
                df = pd.DataFrame(mt5.copy_rates_from_pos(sym, tf_to_mt5("M5"), 0, 40))
                if df.empty:
                    continue
                c = df["close"]
                mom = float(c.iloc[-1] - c.iloc[-8])
                rel = str(meta.get("relation", "direct")).lower()
                w = float(meta.get("weight", 1.0))
                sign = 1.0 if rel == "direct" else -1.0
                score += sign * (1.0 if mom > 0 else -1.0) * w
            except Exception:
                continue
        return score * float(self.config.get("context_weight", 1.0))

    def no_trade(self, row: pd.Series) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        tick = self.broker.tick()
        info = self.broker.symbol_info()
        point = max(float(info.get("point", 0.01)), 1e-9)
        spread = abs(float(tick.get("ask", 0.0)) - float(tick.get("bid", 0.0))) / point if tick else 0.0
        if spread > float(self.config.get("max_spread_points", 80)):
            reasons.append("yüksek_spread")
        atr = max(float(row.get("atr", 0.0)), 1e-9)
        if float(row.get("dist_support", 99999)) < atr * 0.2:
            reasons.append("support_yakin")
        if float(row.get("dist_resistance", 99999)) < atr * 0.2:
            reasons.append("resistance_yakin")
        now = datetime.now()
        for blk in self.config.get("manual_blackout_hours", []):
            try:
                s = datetime.strptime(blk["start"], "%H:%M").time()
                e = datetime.strptime(blk["end"], "%H:%M").time()
                if s <= now.time() <= e:
                    reasons.append("blackout")
                    break
            except Exception:
                pass
        perf = load_json(DATA_DIR / "performance.json", {})
        acct = mt5.account_info() if mt5 else None
        bal = float(getattr(acct, "balance", 0.0) or 0.0)
        max_loss_money = bal * abs(float(self.config.get("max_daily_loss_pct", 4.0))) / 100.0 if bal > 0 else 999999.0
        if float(perf.get("today_pnl", 0.0)) <= -max_loss_money:
            reasons.append("daily_loss_limit")
        if int(perf.get("consecutive_losses", 0)) >= int(self.config.get("max_consecutive_losses", 6)):
            reasons.append("cons_loss_limit")
        news_eval = self.news.evaluate(now)
        if news_eval.get("blocked", False):
            reasons.append("news_blackout")
        return len(reasons) > 0, reasons

    def decide(self, edf: pd.DataFrame, mdf: pd.DataFrame, tdf: pd.DataFrame) -> Optional[Decision]:
        if edf.empty or mdf.empty or tdf.empty or len(edf) < 3:
            return None
        row = edf.iloc[-2]
        trend = self.trend(tdf)
        regime = self.regime(mdf)
        pattern = self.pattern(row)
        p_conf = self.pattern_confidence(row, pattern)
        score = 0.0
        reasons: List[str] = []

        if trend == "up":
            score += 1.2
            reasons.append("trend_up")
        elif trend == "down":
            score += 1.2
            reasons.append("trend_down")
        else:
            score -= 0.4
            reasons.append("trend_flat")

        action = "HOLD"
        if row["ema_20"] > row["ema_50"] and row["rsi"] > 50:
            action = "BUY"
            score += 1.5
        elif row["ema_20"] < row["ema_50"] and row["rsi"] < 50:
            action = "SELL"
            score += 1.5
        else:
            score -= 0.6
            reasons.append("ema_rsi_kararsiz")

        if (pattern in ("bull_pinbar", "bull_engulf") and action == "BUY") or (pattern in ("bear_pinbar", "bear_engulf") and action == "SELL"):
            score += 0.7 * p_conf
            reasons.append(pattern)
        if action == "BUY" and row.get("macd", 0.0) > row.get("macd_signal", 0.0):
            score += 0.4
            reasons.append("macd_buy")
        if action == "SELL" and row.get("macd", 0.0) < row.get("macd_signal", 0.0):
            score += 0.4
            reasons.append("macd_sell")
        if float(row.get("adx", 0.0)) > 18:
            score += 0.25
            reasons.append("adx_support")

        cscore = self.context_score()
        if action == "BUY":
            score += cscore * 0.2
        elif action == "SELL":
            score -= cscore * 0.2
        reasons.append(f"context={cscore:.2f}")

        key = f"{action}_{trend}_{regime}_{pattern}"
        mbias = self.memory.score_bias(key) * float(self.config.get("memory_weight", 0.8))
        score += mbias
        reasons.append(f"mem={mbias:.2f}")
        score += self.rl.choose_bias(regime, trend, pattern)

        # AI ensemble oyu
        ai_pred, ai_conf = self.ai.predict(row)
        ai_side = "BUY" if ai_pred == 1 else "SELL"
        if action != "HOLD" and ai_side == action:
            score += 0.6
            reasons.append("ai_agree")
        elif action != "HOLD":
            score -= 0.35
            reasons.append("ai_disagree")

        # News sentiment etkisi
        news_eval = self.news.evaluate(datetime.now())
        n_bias = float(news_eval.get("sentiment_bias", 0.0))
        if action == "BUY":
            score += n_bias
        elif action == "SELL":
            score -= n_bias
        if news_eval.get("blocked", False):
            score -= 0.4
            reasons.append("news_penalty")

        # BOS/CHoCH/order block katkıları
        if action == "BUY":
            if bool(row.get("bos_up", False)) or bool(row.get("choch_up", False)):
                score += 0.25
                reasons.append("structure_up")
            if bool(row.get("order_block_bull", False)):
                score += 0.2
                reasons.append("ob_bull")
        if action == "SELL":
            if bool(row.get("bos_down", False)) or bool(row.get("choch_down", False)):
                score += 0.25
                reasons.append("structure_down")
            if bool(row.get("order_block_bear", False)):
                score += 0.2
                reasons.append("ob_bear")

        confidence = max(0.0, min(0.99, (0.5 + score / 8.0) * 0.55 + ai_conf * 0.45))

        bad, bad_reasons = self.no_trade(row)
        if bad:
            action = "HOLD"
            reasons.extend(bad_reasons)
        if confidence < float(self.config.get("confidence_threshold", 0.52)):
            action = "HOLD"
            reasons.append("low_conf")
        if score < float(self.config.get("score_threshold", 2.0)):
            action = "HOLD"
            reasons.append("low_score")
        if action == "BUY" and not self.config.get("allow_long", True):
            action = "HOLD"
        if action == "SELL" and not self.config.get("allow_short", True):
            action = "HOLD"

        return Decision(action=action, score=round(score, 3), confidence=round(confidence, 3), reason="|".join(reasons[:12]), trend=trend, regime=regime, pattern=pattern)

    def on_trade_closed(self, pos: Any, close_reason: str) -> None:
        profit = float(getattr(pos, "profit", 0.0))
        side = "BUY" if int(getattr(pos, "type", 1)) == 0 else "SELL"
        perf = load_json(DATA_DIR / "performance.json", {})
        today = datetime.now().strftime("%Y-%m-%d")
        if perf.get("today_date") != today:
            perf["today_date"] = today
            perf["today_pnl"] = 0.0
            perf["today_trades"] = 0
            perf["consecutive_losses"] = 0
        perf["today_pnl"] = float(perf.get("today_pnl", 0.0)) + profit
        perf["wins"] = int(perf.get("wins", 0)) + (1 if profit >= 0 else 0)
        perf["losses"] = int(perf.get("losses", 0)) + (1 if profit < 0 else 0)
        perf["gross_profit"] = float(perf.get("gross_profit", 0.0)) + (profit if profit > 0 else 0.0)
        perf["gross_loss"] = float(perf.get("gross_loss", 0.0)) + (abs(profit) if profit < 0 else 0.0)
        gp, gl = float(perf.get("gross_profit", 0.0)), float(perf.get("gross_loss", 0.0))
        perf["profit_factor"] = round(gp / gl, 3) if gl > 0 else round(gp, 3)
        if profit < 0:
            perf["consecutive_losses"] = int(perf.get("consecutive_losses", 0)) + 1
        else:
            perf["consecutive_losses"] = 0
        save_json(DATA_DIR / "performance.json", perf)

        setup_key = f"{side}_{close_reason}"
        self.memory.learn(setup_key, profit >= 0)
        # RL reward update
        regime = "trend" if abs(float(getattr(pos, "profit", 0.0))) > 0 else "mixed"
        trend = "up" if side == "BUY" else "down"
        self.rl.update(regime, trend, close_reason, side, profit)
        journal = load_json(DATA_DIR / "trade_journal.json", [])
        journal.append(
            {
                "time": datetime.now().isoformat(),
                "ticket": int(getattr(pos, "ticket", 0)),
                "side": side,
                "profit": round(profit, 3),
                "close_reason": close_reason,
                "symbol": self.broker.active_symbol,
            }
        )
        save_json(DATA_DIR / "trade_journal.json", journal[-5000:])

        # Strategy evolution suggestions
        perf_after = load_json(DATA_DIR / "performance.json", {})
        self.evolution.evolve(perf_after)
        wins = int(perf_after.get("wins", 0))
        losses = int(perf_after.get("losses", 0))
        wr = (wins / max(1, wins + losses)) * 100.0
        self.telegram.send(
            f"KAPANDI {side} #{int(getattr(pos, 'ticket', 0))}\n"
            f"Kar/Zarar: {profit:.2f}\n"
            f"Neden: {close_reason}\n"
            f"Winrate: {wr:.1f}% | PF: {perf_after.get('profit_factor', '-')}\n"
            f"Gunluk PnL: {perf_after.get('today_pnl', 0)}"
        )

    def _spawn_panel(self) -> None:
        if self._panel_started or not self.config.get("auto_start_panel", True):
            return
        self._panel_started = True
        host = str(self.config.get("panel_host", "127.0.0.1"))
        port = int(self.config.get("panel_port", 5000))
        env = os.environ.copy()
        env["PANEL_HOST"] = host
        env["PANEL_PORT"] = str(port)
        try:
            subprocess.Popen([sys.executable, str(ROOT / "panel.py")], cwd=str(ROOT), env=env)
            time.sleep(1.0)
            if self.config.get("auto_open_panel_browser", True):
                webbrowser.open(f"http://{host}:{port}")
        except Exception as ex:
            log_line("error", f"Panel baslatilamadi: {ex}")

    @staticmethod
    def _extract_order_retcode(data: Any) -> Optional[int]:
        if not isinstance(data, dict):
            return None
        at = data.get("attempts")
        if isinstance(at, list) and at:
            last = at[-1]
            if isinstance(last, dict):
                rc = last.get("retcode")
                if rc is not None:
                    return int(rc)
        return None

    def _log_order_fail_throttled(self, data: Any) -> None:
        now = time.time()
        rc = self._extract_order_retcode(data)
        # 10018 market closed, 10030 invalid request / session, 10017 trade disabled vb.
        quiet = rc in {10014, 10015, 10016, 10017, 10018, 10019, 10030}
        interval = 120.0 if quiet else 25.0
        if now - self.last_order_fail_log_ts < interval:
            return
        self.last_order_fail_log_ts = now
        if self.config.get("order_log_verbose", False):
            log_line("error", f"Order fail: {data}")
        else:
            attempts = data.get("attempts", []) if isinstance(data, dict) else []
            short = []
            for a in attempts[:3]:
                if not isinstance(a, dict):
                    continue
                r = a.get("retcode", a.get("check_retcode", "na"))
                fill = a.get("type_filling_used", a.get("filling", "na"))
                short.append(f"f={fill},rc={r}")
            extra = f" retcode={rc}" if rc is not None else ""
            if quiet:
                log_line(
                    "system",
                    f"Emir reddedildi (muhtemelen piyasa kapali veya oturum disi){extra}. Kisa: {'; '.join(short) if short else str(data)}",
                )
            else:
                log_line("error", f"Order fail (kisa): {'; '.join(short) if short else str(data)}{extra}")

    def calc_lot_sl_tp(self, side: str, row: pd.Series, confidence: float) -> Tuple[float, float, float]:
        info = self.broker.symbol_info()
        tick = self.broker.tick()
        entry = float(tick["ask"] if side == "BUY" else tick["bid"])
        atr = float(row.get("atr", 0.0))
        sl_dist = max(atr * float(self.config.get("sl_atr_mult", 1.6)), float(info.get("point", 0.01)) * 80)
        tp_dist = max(atr * float(self.config.get("tp_atr_mult", 2.2)), sl_dist * 1.2)
        if side == "BUY":
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            sl, tp = entry + sl_dist, entry - tp_dist
        acct = mt5.account_info()
        balance = float(acct.balance) if acct else 10000.0
        risk_pct = float(self.config.get("risk_per_trade_pct", 0.5))
        risk_amount = balance * risk_pct / 100.0
        tick_value = float(info.get("trade_tick_value", 0.1) or 0.1)
        tick_size = float(info.get("trade_tick_size", info.get("point", 0.01)) or 0.01)
        price_risk = abs(entry - sl)
        cost_per_lot = (price_risk / max(tick_size, 1e-9)) * max(tick_value, 1e-9)
        lot = max(risk_amount / max(cost_per_lot, 1e-9), float(info.get("volume_min", 0.01)))
        step = float(info.get("volume_step", 0.01) or 0.01)
        vmax = float(info.get("volume_max", 100.0) or 100.0)
        lot = math.floor(lot / step) * step
        lot = max(float(info.get("volume_min", 0.01)), min(vmax, round(lot, 2)))
        return lot, sl, tp

    def open_trade(self, dec: Decision, row: pd.Series) -> None:
        if dec.action not in ("BUY", "SELL"):
            return
        if len(self.broker.positions()) >= int(self.config.get("max_open_positions", 1)):
            return
        ok_sym, why = self.broker.can_trade_now()
        if not ok_sym:
            now_ts = time.time()
            if now_ts - self.last_trade_block_log_ts >= 90.0:
                log_line("system", f"Islem atlaniyor: {why}")
                self.last_trade_block_log_ts = now_ts
            return
        perf = load_json(DATA_DIR / "performance.json", {})
        today = datetime.now().strftime("%Y-%m-%d")
        if perf.get("today_date") != today:
            perf["today_date"] = today
            perf["today_trades"] = 0
        if int(perf.get("today_trades", 0)) >= int(self.config.get("max_daily_trades", 999)):
            return
        cd = float(self.config.get("order_cooldown_seconds", 0) or 0)
        if cd > 0 and (time.time() - self.last_order_attempt_ts) < cd:
            return
        lot, sl, tp = self.calc_lot_sl_tp(dec.action, row, dec.confidence)
        self.last_order_attempt_ts = time.time()
        ok, data = self.broker.open_market(dec.action, lot, sl, tp, f"{dec.action}_{dec.score:.2f}")
        if ok:
            log_line("trade", f"OPEN {dec.action} lot={lot} sl={sl:.3f} tp={tp:.3f} {data}")
            self.telegram.send(
                f"ACILDI {dec.action} {self.broker.active_symbol}\n"
                f"Lot: {lot} SL: {sl:.3f} TP: {tp:.3f}\n"
                f"Skor: {dec.score} | Guven: {dec.confidence}"
            )
            perf = load_json(DATA_DIR / "performance.json", {})
            if perf.get("today_date") != today:
                perf["today_date"] = today
                perf["today_trades"] = 0
            perf["today_trades"] = int(perf.get("today_trades", 0)) + 1
            save_json(DATA_DIR / "performance.json", perf)
        else:
            self._log_order_fail_throttled(data)

    def reevaluate_open_positions(self, edf: pd.DataFrame) -> None:
        positions = self.broker.positions()
        if not positions or edf.empty:
            return
        dec = self.decide(edf, edf, edf)
        row = edf.iloc[-2]
        atr = max(float(row.get("atr", 0.0)), 1e-9)
        tick = self.broker.tick()
        for pos in positions:
            pos_side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            cur_price = float(tick.get("bid", 0.0)) if pos_side == "BUY" else float(tick.get("ask", 0.0))
            open_price = float(getattr(pos, "price_open", 0.0))
            tp = float(getattr(pos, "tp", 0.0))
            sl = float(getattr(pos, "sl", 0.0))
            profit = float(getattr(pos, "profit", 0.0))
            ticket = int(getattr(pos, "ticket", 0))

            # TP'ye yaklaşınca ve momentum güçlü ise TP'yi dinamik uzat.
            if tp > 0 and open_price > 0 and cur_price > 0:
                total_target = abs(tp - open_price)
                moved = abs(cur_price - open_price)
                progress = moved / max(total_target, 1e-9)
                rr_live = moved / max(abs(open_price - sl), 1e-9) if sl > 0 else 0.0

                # Break-even: hedefin bir kısmı görüldüğünde SL'i entry'ye çek.
                if progress >= float(self.config.get("break_even_progress", 0.5)):
                    if pos_side == "BUY" and sl < open_price:
                        if self.broker.modify_position_sltp(pos, open_price, tp):
                            sl = open_price
                            log_line("trade", f"BREAKEVEN BUY ticket={ticket} sl->{open_price:.3f}")
                    elif pos_side == "SELL" and sl > open_price:
                        if self.broker.modify_position_sltp(pos, open_price, tp):
                            sl = open_price
                            log_line("trade", f"BREAKEVEN SELL ticket={ticket} sl->{open_price:.3f}")

                # Trailing: yeterli ilerleme sonrası SL'i ATR ile izlet.
                if progress >= float(self.config.get("trailing_start_progress", 0.6)):
                    trail_dist = atr * float(self.config.get("trailing_atr_mult", 0.8))
                    if pos_side == "BUY":
                        candidate_sl = cur_price - trail_dist
                        if candidate_sl > sl and candidate_sl < cur_price:
                            if self.broker.modify_position_sltp(pos, candidate_sl, tp):
                                sl = candidate_sl
                                log_line("trade", f"TRAIL BUY ticket={ticket} sl->{candidate_sl:.3f}")
                    else:
                        candidate_sl = cur_price + trail_dist
                        if (sl <= 0 or candidate_sl < sl) and candidate_sl > cur_price:
                            if self.broker.modify_position_sltp(pos, candidate_sl, tp):
                                sl = candidate_sl
                                log_line("trade", f"TRAIL SELL ticket={ticket} sl->{candidate_sl:.3f}")

                if (
                    progress >= float(self.config.get("near_tp_factor", 0.88))
                    and dec is not None
                    and dec.action == pos_side
                    and dec.confidence >= float(self.config.get("tp_extension_min_confidence", 0.62))
                    and rr_live >= 1.0
                ):
                    step = int(self.tp_extension_steps.get(ticket, 0))
                    max_steps = int(self.config.get("tp_extension_max_steps", 3))
                    if step >= max_steps:
                        pass
                    else:
                        ext = atr * float(self.config.get("tp_extension_atr_mult", 0.9))
                        # Her uzatmada katkıyı biraz azalt.
                        ext = ext * max(0.4, (1.0 - (step * 0.2)))
                        new_tp = tp + ext if pos_side == "BUY" else tp - ext
                        if self.broker.modify_position_sltp(pos, sl, new_tp):
                            self.tp_extended_tickets.add(ticket)
                            self.tp_extension_steps[ticket] = step + 1
                            log_line(
                                "trade",
                                f"TP EXTEND ticket={ticket} step={step+1}/{max_steps} old_tp={tp:.3f} new_tp={new_tp:.3f}",
                            )

                # Kârda ve momentum zayıfladıysa TP'ye varmadan karı al.
                if (
                    profit > 0
                    and dec is not None
                    and dec.action != pos_side
                    and progress >= float(self.config.get("early_take_profit_progress", 0.65))
                ):
                    if self.broker.close_position(pos, "early_take_profit"):
                        self.on_trade_closed(pos, "early_take_profit")
                    log_line("trade", f"CLOSE {pos.ticket} early_take_profit")
                    continue

                # Kademeli partial close
                if (
                    self.config.get("partial_close_enabled", True)
                    and ticket not in self.partial_closed_tickets
                    and rr_live >= float(self.config.get("partial_close_rr", 1.6))
                    and profit > 0
                ):
                    ratio = float(self.config.get("partial_close_ratio", 0.4))
                    if self.broker.partial_close_position(pos, ratio, "partial_rr"):
                        self.partial_closed_tickets.add(ticket)
                        log_line("trade", f"PARTIAL CLOSE ticket={ticket} ratio={ratio}")

            # Kullanıcının istediği mantık: "Şimdi sıfırdan açar mıydım?"
            if dec is None or dec.action != pos_side:
                if self.broker.close_position(pos, "re_eval_close"):
                    self.on_trade_closed(pos, "re_eval_close")
                log_line("trade", f"CLOSE {pos.ticket} re-eval mismatch")
                self.tp_extension_steps.pop(ticket, None)

    def write_runtime(self, decision: Optional[Decision]) -> None:
        tick = self.broker.tick()
        acct = mt5.account_info() if mt5 else None
        acct_dict = acct._asdict() if acct else {}
        pos_list = []
        for p in self.broker.positions():
            try:
                side_txt = "BUY" if int(p.type) == getattr(mt5, "POSITION_TYPE_BUY", 0) else "SELL"
                pos_list.append(
                    {
                        "ticket": int(p.ticket),
                        "type": int(p.type),
                        "action": side_txt,
                        "volume": float(p.volume),
                        "price_open": float(p.price_open),
                        "sl": float(p.sl),
                        "tp": float(p.tp),
                        "profit": float(getattr(p, "profit", 0.0)),
                    }
                )
            except Exception:
                continue
        perf = load_json(DATA_DIR / "performance.json", {})
        state = {
            "time": datetime.now().isoformat(),
            "symbol": self.broker.active_symbol,
            "tick": tick,
            "paper_trade": bool(self.config.get("paper_trade", False)),
            "positions": pos_list,
            "market": {
                "last_close": tick.get("bid"),
                "bid": tick.get("bid"),
                "ask": tick.get("ask"),
            },
            "trading_ok": self.broker.can_trade_now()[0] if mt5 else False,
            "account": {
                "balance": acct_dict.get("balance"),
                "equity": acct_dict.get("equity"),
                "currency": acct_dict.get("currency"),
            },
            "performance": perf,
            "analytics": {
                "winrate": round((int(perf.get("wins", 0)) / max(1, int(perf.get("wins", 0)) + int(perf.get("losses", 0)))) * 100.0, 2),
                "avg_win": round(float(perf.get("gross_profit", 0.0)) / max(1, int(perf.get("wins", 0))), 3),
                "avg_loss": round(float(perf.get("gross_loss", 0.0)) / max(1, int(perf.get("losses", 0))), 3),
                "expectancy": round(
                    (float(perf.get("gross_profit", 0.0)) - float(perf.get("gross_loss", 0.0)))
                    / max(1, int(perf.get("wins", 0)) + int(perf.get("losses", 0))),
                    4,
                ),
            },
            "model_status": {"models": ["rule_engine", "rf", "gb", "lr"], "last_train_time": datetime.now().isoformat(), "scores": load_json(DATA_DIR / "walkforward_report.json", {})},
            "news": self.news.evaluate(datetime.now()),
            "decision": asdict(decision) if decision else None,
            "config": {
                "loop_seconds": self.config.get("loop_seconds"),
                "score_threshold": self.config.get("score_threshold"),
                "confidence_threshold": self.config.get("confidence_threshold"),
            },
        }
        save_json(DATA_DIR / "runtime_state.json", state)

    def maybe_send_status(self) -> None:
        if (datetime.now() - self.last_status) < timedelta(minutes=30):
            return
        p = load_json(DATA_DIR / "performance.json", {})
        self.telegram.send(
            f"{self.config['bot_name']} aktif\n"
            f"Sembol: {self.broker.active_symbol}\n"
            f"Günlük işlem: {p.get('today_trades', 0)}\n"
            f"Günlük PnL: {p.get('today_pnl', 0.0)}"
        )
        self.last_status = datetime.now()

    def run_once(self) -> None:
        self.reload_config()
        ctrl = load_json(DATA_DIR / "control.json", {"enabled": True, "running": True})
        running = bool(ctrl.get("enabled", ctrl.get("running", True)))
        if not running:
            self.write_runtime(None)
            return
        if not self.broker.ensure_connection():
            self.write_runtime(None)
            return
        edf, mdf, tdf = self.market_data()
        if edf.empty:
            log_line("error", "Veri alınamadı.")
            self.write_runtime(None)
            return
        # Online learning / adaptive memory
        self.memory.decay(float(self.config.get("adaptive_memory_decay", 0.98)))
        if self.config.get("online_learning_enabled", True):
            now = datetime.now()
            if (now - self.last_train_at).total_seconds() >= float(self.config.get("retrain_every_seconds", 300)):
                train_result = self.ai.train(edf)
                wf = self.ai.walkforward(edf)
                save_json(DATA_DIR / "walkforward_report.json", {"train": train_result, "walkforward": wf, "time": now.isoformat()})
                self.last_train_at = now
        self.reevaluate_open_positions(edf)
        dec = self.decide(edf, mdf, tdf)
        if dec:
            self.open_trade(dec, edf.iloc[-2])
            log_line("ai", f"{dec.action} score={dec.score} conf={dec.confidence} {dec.reason}")
        self.write_runtime(dec)
        self.maybe_send_status()

    def run_forever(self) -> None:
        log_line("system", f"{self.config['bot_name']} başlatıldı.")
        self._spawn_panel()
        if self.broker.ensure_connection():
            self.telegram.send(
                f"BOT AKTIF\n"
                f"Hesap baglandi | Sembol: {self.broker.active_symbol}\n"
                f"Max pozisyon: {self.config.get('max_open_positions')} | Gunluk max islem: {self.config.get('max_daily_trades')}\n"
                f"Cooldown: {self.config.get('order_cooldown_seconds')} sn"
            )
        while True:
            started = time.time()
            try:
                self.run_once()
            except Exception as ex:
                log_line("error", f"Loop hata: {ex}")
            elapsed = time.time() - started
            sleep_s = max(0.05, float(self.config.get("loop_seconds", 0.5)) - elapsed)
            time.sleep(sleep_s)


if __name__ == "__main__":
    os.environ["PYTHONUTF8"] = "1"
    bot = Bot()
    bot.run_forever()
