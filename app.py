import os
import sqlite3
from datetime import datetime
import pytz
import joblib
import numpy as np
import yfinance as yf
from flask import Flask, render_template, jsonify, request
from ml_engine.features import fetch_and_engineer_features, fetch_oi_and_flow_analytics, INSTRUMENTS
from ml_engine.train_model import train_and_save_model
from ml_engine.options_math import build_strategy_recommendation

app = Flask(__name__)
MODEL_PATH = "ml_engine/model_bundle.pkl"
DB_PATH = "trades.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            spot_entry REAL,
            regime TEXT,
            strategy TEXT,
            legs TEXT,
            confidence REAL,
            max_risk REAL,
            status TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            lot_size INTEGER
        )
    """)
    conn.commit()
    conn.close()


def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return train_and_save_model(MODEL_PATH)
    return joblib.load(MODEL_PATH)


init_db()
bundle = load_bundle()


def get_market_status():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30):
        return "MARKET LIVE (NSE / BSE)"
    return "MARKET CLOSED (Last Close Signal)"


def get_trade_timing_plan(segment: str, regime: str, dte: int) -> dict:
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    hm = (now.hour, now.minute)
    is_weekday = now.weekday() < 5

    if segment == "INTRADAY":
        entry_window = "09:45 AM – 11:15 AM IST (Post-Opening Range Breakout)"
        exit_window = "03:10 PM IST Mandatory MIS Auto-Square-Off (or +2.0x ATR Target / -1.0x ATR SL)"
    elif segment == "DELIVERY":
        entry_window = "02:30 PM – 03:20 PM IST (End-of-Day Institutional Closing Confirmation)"
        exit_window = "Hold 2–5 Trading Days | Exit at +3.0x ATR Target or -1.5x ATR Trailing Stop"
    else:
        if "SIDEWAYS" in regime:
            entry_window = "09:30 AM – 10:30 AM IST (Capture Maximum Morning IV & Theta Decay)"
            exit_window = "03:15 PM IST Intraday or 1 Day Before Expiry (Exit at +50% Max Credit / -40% SL)"
        else:
            entry_window = "09:45 AM – 11:30 AM IST or 01:45 PM – 02:45 PM IST (Momentum Breakout Window)"
            exit_window = (
                "Same Day 03:15 PM IST (High Expiry Gamma — Avoid Overnight Decay)"
                if dte <= 1
                else "1–3 Trading Days | Exit at +60% Spread Profit or -40% Stop-Loss"
            )

    if not is_weekday or hm < (9, 15) or hm > (15, 30):
        live_action = "QUEUE LIMIT ORDER FOR 09:45 AM IST OPEN (Avoid 09:15–09:30 Opening Volatility)"
    elif (9, 15) <= hm < (9, 45):
        live_action = "WAIT FOR 09:45 AM IST — Let initial 30-min opening range settle before entry"
    elif (9, 45) <= hm <= (14, 45):
        live_action = "OPTIMAL ENTRY WINDOW ACTIVE NOW — Execute limit order at recommended strike"
    else:
        live_action = "EXIT / SQUARE-OFF WINDOW (02:45–03:30 PM IST) — Book profits or trail Stop-Loss tight"

    return {
        "entry_window": entry_window,
        "exit_window": exit_window,
        "live_action": live_action,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/instruments", methods=["GET"])
def get_instruments():
    items = [
        {"symbol": k, "name": v["name"], "is_custom": False}
        for k, v in INSTRUMENTS.items()
    ]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT symbol, name FROM watchlist ORDER BY symbol ASC")
    rows = c.fetchall()
    conn.close()

    existing = {i["symbol"] for i in items}
    for sym, name in rows:
        if sym not in existing:
            items.append({"symbol": sym, "name": f"{name} (NSE)", "is_custom": True})
    return jsonify(items)


@app.route("/api/instruments", methods=["POST"])
def add_instrument():
    data = request.json or {}
    symbol = data.get("symbol", "").strip().upper().replace(".NS", "")
    lot_size = int(data.get("lot_size", 250))

    if not symbol or not symbol.isalnum():
        return jsonify({"error": "Enter a valid NSE stock symbol (e.g., INFY, SBIN, TATAMOTORS)."}), 400

    check = yf.download(f"{symbol}.NS", period="5d", progress=False)
    if check is None or check.empty:
        return jsonify({"error": f"Symbol '{symbol}.NS' was not found on Yahoo Finance / NSE."}), 404

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO watchlist (symbol, name, lot_size) VALUES (?, ?, ?)",
        (symbol, symbol, lot_size),
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "added", "symbol": symbol})


@app.route("/api/signal", methods=["GET"])
def api_signal():
    capital = float(request.args.get("capital", 50000))
    instrument = request.args.get("instrument", "NIFTY").strip().upper().replace(".NS", "")
    segment = request.args.get("segment", "OPTIONS").strip().upper()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT lot_size FROM watchlist WHERE symbol = ?", (instrument,))
    row = c.fetchone()
    conn.close()
    custom_lot = int(row[0]) if row else None

    try:
        df = fetch_and_engineer_features(instrument_key=instrument, is_live=True)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch data for {instrument}: {exc}"}), 400

    if df is None or df.empty or len(df) < 2:
        return jsonify({"error": f"Insufficient historical candles for {instrument}."}), 400

    latest = df.iloc[-1:]
    X_latest = latest[bundle["feature_cols"]]

    spot = round(float(latest["Close"].iloc[0]), 2)
    prev_close = float(df["Close"].iloc[-2])
    spot_chg_pct = round(((spot - prev_close) / prev_close) * 100, 2)
    vix = round(float(latest["VIX"].iloc[0]), 2)
    iv_rank = round(float(latest["IV_Rank"].iloc[0]), 1)
    atr = float(latest["ATR_14"].iloc[0])
    dte = int(latest["Days_To_Expiry"].iloc[0])

    # 1. Fetch Live Open Interest (OI), PCR & Real-Time Overflow Analytics
    oi_info = fetch_oi_and_flow_analytics(instrument, spot, df)

    # 2. Always-Active Strategy Selection blended with Live OI/PCR Flow
    probs = bundle["model"].predict_proba(X_latest)[0]
    trade_probs = np.array(probs[1:], dtype=float)  # [Bullish, Bearish, Sideways]

    if oi_info["flow_tone"] == "bullish":
        trade_probs[0] *= 1.15
    elif oi_info["flow_tone"] == "bearish":
        trade_probs[1] *= 1.15
    else:
        trade_probs[2] *= 1.12

    total_trade_prob = float(np.sum(trade_probs))
    best_trade_class = int(np.argmax(trade_probs)) + 1
    effective_class = best_trade_class
    confidence = float(trade_probs[best_trade_class - 1] / total_trade_prob) if total_trade_prob > 0 else 0.50

    # 3. Build Greek-Optimized Options or Equity Trade Structure
    rec = build_strategy_recommendation(
        spot,
        vix,
        iv_rank,
        atr,
        effective_class,
        confidence,
        capital,
        instrument_key=instrument,
        custom_lot=custom_lot,
        days_to_expiry=dte,
        oi_data=oi_info,
        segment=segment,
    )

    # 4. SHAP Feature Attribution
    shap_vals = bundle["explainer"].shap_values(X_latest)
    target_idx = best_trade_class
    if isinstance(shap_vals, list):
        vals = shap_vals[target_idx][0]
    else:
        vals = shap_vals[0, :, target_idx]

    top_indices = np.argsort(np.abs(vals))[::-1][:6]
    shap_data = [
        {"feature": bundle["feature_cols"][i], "impact": round(float(vals[i]), 4)}
        for i in top_indices
    ]

    # 5. 60-Day Price & EMA Chart Data
    hist = df.tail(60)
    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
        "close": [round(float(x), 2) for x in hist["Close"]],
        "ema9": [round(float(x), 2) for x in hist["Close"].ewm(span=9).mean()],
        "ema21": [round(float(x), 2) for x in hist["Close"].ewm(span=21).mean()],
    }

    return jsonify({
        "instrument": instrument,
        "market_status": get_market_status(),
        "test_accuracy": bundle["test_accuracy"],
        "spot": spot,
        "spot_chg_pct": spot_chg_pct,
        "vix": vix,
        "iv_rank": iv_rank,
        "confidence": round(confidence * 100, 1),
        "oi_flow": oi_info,
        "timing": get_trade_timing_plan(segment, rec["regime"], dte),
        "recommendation": rec,
        "shap": shap_data,
        "chart": chart_data,
    })


@app.route("/api/paper-trade", methods=["POST"])
def log_paper_trade():
    data = request.json
    ist = pytz.timezone("Asia/Kolkata")
    ts = datetime.now(ist).strftime("%Y-%m-%d %H:%M IST")
    legs_str = (
        " | ".join([f"{l['action']} {l['instrument']} ({l.get('moneyness', '')})" for l in data.get("legs", [])])
        or "Cash / Hold"
    )

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO paper_trades (timestamp, spot_entry, regime, strategy, legs, confidence, max_risk, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ts,
            data["spot"],
            data["regime"],
            data["strategy"],
            legs_str,
            data["confidence"],
            data["max_risk"],
            "OPEN",
        ),
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "saved"})


@app.route("/api/paper-trades", methods=["GET"])
def get_paper_trades():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, timestamp, spot_entry, regime, strategy, legs, confidence, max_risk, status FROM paper_trades ORDER BY id DESC LIMIT 10"
    )
    rows = c.fetchall()
    conn.close()
    return jsonify([
        {
            "id": r[0],
            "timestamp": r[1],
            "spot_entry": r[2],
            "regime": r[3],
            "strategy": r[4],
            "legs": r[5],
            "confidence": r[6],
            "max_risk": r[7],
            "status": r[8],
        }
        for r in rows
    ])


@app.route("/api/retrain", methods=["POST"])
def retrain():
    global bundle
    bundle = train_and_save_model(MODEL_PATH)
    return jsonify({"status": "retrained", "accuracy": bundle["test_accuracy"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)