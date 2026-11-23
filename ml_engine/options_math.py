import math
import numpy as np
from scipy.stats import norm
from ml_engine.features import INSTRUMENTS, get_instrument_config


def resolve_lot_and_step(spot: float, instrument_key: str, custom_lot: int = None):
    key = instrument_key.strip().upper().replace(".NS", "")
    cfg = get_instrument_config(key)
    lot_size = custom_lot if custom_lot else cfg["lot_size"]

    if key in INSTRUMENTS:
        return lot_size, cfg["strike_step"]

    if spot < 500:
        step = 5
    elif spot < 1500:
        step = 10
    elif spot < 5000:
        step = 20
    else:
        step = 50
    return lot_size, step


def classify_moneyness(spot: float, strike: int, opt_type: str, step: int) -> str:
    diff = strike - spot
    if abs(diff) <= (step * 0.55):
        return "ATM"
    if opt_type == "CE":
        return "ITM" if strike < spot else "OTM"
    return "ITM" if strike > spot else "OTM"


def black_scholes_full_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "CE"):
    T = max(T, 1.0 / 365.0)
    sigma = max(sigma, 0.05)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = (S * norm.pdf(d1) * sqrt_T) / 100.0

    if opt_type == "CE":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0

    return (
        round(max(price, 0.5), 2),
        round(delta, 3),
        round(gamma, 5),
        round(theta, 2),
        round(vega, 2),
    )


def round_strike(price: float, step: int) -> int:
    return int(round(price / step) * step)


def build_equity_recommendation(
    spot: float,
    atr: float,
    regime_class: int,
    capital: float,
    instrument_key: str,
    segment: str = "INTRADAY",
) -> dict:
    is_intraday = (segment == "INTRADAY")
    sl_mult = 1.0 if is_intraday else 1.5
    tgt_mult = 2.0 if is_intraday else 3.0
    risk_budget = capital * (0.01 if is_intraday else 0.02)
    buying_power = capital * (5.0 if is_intraday else 1.0)

    atr_safe = max(atr, spot * 0.005)
    sl_points = round(atr_safe * sl_mult, 2)
    tgt_points = round(atr_safe * tgt_mult, 2)

    if regime_class == 1:
        regime_name = "STRONG BULLISH"
        action = "BUY"
        strategy_name = "Intraday Momentum Long (MIS)" if is_intraday else "Positional Swing Delivery (CNC)"
        stop_loss = round(spot - sl_points, 2)
        target = round(spot + tgt_points, 2)
    elif regime_class == 2:
        regime_name = "STRONG BEARISH"
        if is_intraday:
            action = "SELL"
            strategy_name = "Intraday Short Sell (MIS)"
            stop_loss = round(spot + sl_points, 2)
            target = round(spot - tgt_points, 2)
        else:
            action = "HOLD / AVOID"
            strategy_name = "Cash Preservation (CNC Delivery Long-Only)"
            stop_loss = round(spot - sl_points, 2)
            target = round(spot + tgt_points, 2)
    else:
        regime_name = "SIDEWAYS / RANGE"
        action = "BUY"
        strategy_name = "Intraday Mean-Reversion Scalp (MIS)" if is_intraday else "Accumulate Near Support (CNC)"
        sl_points = round(atr_safe * 0.8, 2)
        tgt_points = round(atr_safe * 1.6, 2)
        stop_loss = round(spot - sl_points, 2)
        target = round(spot + tgt_points, 2)

    if action == "HOLD / AVOID":
        qty = 0
        max_risk = 0.0
        max_reward = 0.0
        legs = []
    else:
        qty_by_risk = int(risk_budget // max(sl_points, 0.5))
        qty_by_cap = int(buying_power // spot)
        qty = max(1, min(qty_by_risk, qty_by_cap)) if qty_by_cap >= 1 else 0
        max_risk = round(qty * sl_points, 2)
        max_reward = round(qty * tgt_points, 2)
        product_tag = "MIS (Intraday)" if is_intraday else "CNC (Delivery)"
        legs = [
            {
                "action": action,
                "instrument": f"{instrument_key} EQ ({product_tag})",
                "strike": spot,
                "moneyness": "CASH",
                "type": "EQ",
                "premium": spot,
                "delta": 1.0 if action == "BUY" else -1.0,
                "gamma": 0.0,
                "vega": 0.0,
                "lots": qty,
            }
        ]

    price_grid = np.linspace(spot * 0.95, spot * 1.05, 60)
    payoff_grid = []
    for px in price_grid:
        if qty == 0:
            pnl = 0.0
        elif action == "BUY":
            clamped_px = min(max(px, stop_loss), target)
            pnl = (clamped_px - spot) * qty
        else:
            clamped_px = max(min(px, stop_loss), target)
            pnl = (spot - clamped_px) * qty
        payoff_grid.append(round(pnl, 2))

    insight = (
        f"Equity Cash Mode ({'5x MIS Intraday' if is_intraday else '1x CNC Delivery'}): "
        f"Entry ₹{spot} | Target ₹{target} | Stop-Loss ₹{stop_loss} (1:2 ATR Risk-Reward)."
    )

    return {
        "segment": segment,
        "regime": regime_name,
        "strategy": strategy_name,
        "legs": legs,
        "lots": qty,
        "lot_size": 1,
        "unit_label": "Shares",
        "atm_strike": spot,
        "entry_price": spot,
        "target_price": target,
        "stop_loss_price": stop_loss,
        "max_risk": max_risk,
        "max_reward": max_reward,
        "breakeven": spot,
        "net_delta": round(float(qty if action == "BUY" else -qty), 1),
        "net_gamma": 0.0,
        "net_theta": 0.0,
        "net_vega": 0.0,
        "greek_insight": insight,
        "payoff_prices": [round(float(x), 1) for x in price_grid],
        "payoff_pnl": payoff_grid,
    }


def build_strategy_recommendation(
    spot: float,
    vix: float,
    iv_rank: float,
    atr: float,
    regime_class: int,
    confidence: float,
    capital: float,
    instrument_key: str = "NIFTY",
    custom_lot: int = None,
    days_to_expiry: int = 4,
    oi_data: dict = None,
    segment: str = "OPTIONS",
) -> dict:
    segment = segment.upper()
    if segment in ("INTRADAY", "DELIVERY"):
        return build_equity_recommendation(spot, atr, regime_class, capital, instrument_key, segment=segment)

    lot_size, step = resolve_lot_and_step(spot, instrument_key, custom_lot)
    atm = round_strike(spot, step)
    dte = max(int(days_to_expiry), 1)
    T = dte / 365.0
    r = 0.07
    sigma = max(vix / 100.0, 0.12)
    max_allowed_risk = capital * 0.02

    spread_steps = 3 if iv_rank >= 60.0 else 2
    anchor_shift = -1 if (dte <= 2 and confidence >= 0.65) else 0

    legs = []
    if regime_class == 1:
        regime_name = "STRONG BULLISH"
        strategy_name = "Bull Call Spread (Delta+ / Vega-Hedged)"
        k_buy = atm + (anchor_shift * step)
        k_sell = k_buy + (spread_steps * step)

        p_buy, d_buy, g_buy, t_buy, v_buy = black_scholes_full_greeks(spot, k_buy, T, r, sigma, "CE")
        p_sell, d_sell, g_sell, t_sell, v_sell = black_scholes_full_greeks(spot, k_sell, T, r, sigma, "CE")

        net_debit = max(p_buy - p_sell, 1.0)
        spread_width = k_sell - k_buy
        risk_per_lot = net_debit * lot_size
        reward_per_lot = max((spread_width - net_debit) * lot_size, risk_per_lot * 0.5)

        lots = max(1, int(max_allowed_risk // risk_per_lot))
        max_risk = round(risk_per_lot * lots, 2)
        max_reward = round(reward_per_lot * lots, 2)
        breakeven = round(k_buy + net_debit, 2)

        net_delta = round((d_buy - d_sell) * lot_size * lots, 2)
        net_gamma = round((g_buy - g_sell) * lot_size * lots, 4)
        net_theta = round((t_buy - t_sell) * lot_size * lots, 2)
        net_vega = round((v_buy - v_sell) * lot_size * lots, 2)

        legs = [
            {"action": "BUY", "instrument": f"{instrument_key} {k_buy} CE", "strike": k_buy, "moneyness": classify_moneyness(spot, k_buy, "CE", step), "type": "CE", "premium": p_buy, "delta": d_buy, "gamma": g_buy, "vega": v_buy, "lots": lots},
            {"action": "SELL", "instrument": f"{instrument_key} {k_sell} CE", "strike": k_sell, "moneyness": classify_moneyness(spot, k_sell, "CE", step), "type": "CE", "premium": p_sell, "delta": d_sell, "gamma": g_sell, "vega": v_sell, "lots": lots},
        ]

    elif regime_class == 2:
        regime_name = "STRONG BEARISH"
        strategy_name = "Bear Put Spread (Delta- / Vega-Hedged)"
        k_buy = atm - (anchor_shift * step)
        k_sell = k_buy - (spread_steps * step)

        p_buy, d_buy, g_buy, t_buy, v_buy = black_scholes_full_greeks(spot, k_buy, T, r, sigma, "PE")
        p_sell, d_sell, g_sell, t_sell, v_sell = black_scholes_full_greeks(spot, k_sell, T, r, sigma, "PE")

        net_debit = max(p_buy - p_sell, 1.0)
        spread_width = k_buy - k_sell
        risk_per_lot = net_debit * lot_size
        reward_per_lot = max((spread_width - net_debit) * lot_size, risk_per_lot * 0.5)

        lots = max(1, int(max_allowed_risk // risk_per_lot))
        max_risk = round(risk_per_lot * lots, 2)
        max_reward = round(reward_per_lot * lots, 2)
        breakeven = round(k_buy - net_debit, 2)

        net_delta = round((d_buy - d_sell) * lot_size * lots, 2)
        net_gamma = round((g_buy - g_sell) * lot_size * lots, 4)
        net_theta = round((t_buy - t_sell) * lot_size * lots, 2)
        net_vega = round((v_buy - v_sell) * lot_size * lots, 2)

        legs = [
            {"action": "BUY", "instrument": f"{instrument_key} {k_buy} PE", "strike": k_buy, "moneyness": classify_moneyness(spot, k_buy, "PE", step), "type": "PE", "premium": p_buy, "delta": d_buy, "gamma": g_buy, "vega": v_buy, "lots": lots},
            {"action": "SELL", "instrument": f"{instrument_key} {k_sell} PE", "strike": k_sell, "moneyness": classify_moneyness(spot, k_sell, "PE", step), "type": "PE", "premium": p_sell, "delta": d_sell, "gamma": g_sell, "vega": v_sell, "lots": lots},
        ]

    else:
        regime_name = "SIDEWAYS / HIGH IV"
        strategy_name = "Short Iron Condor (Theta+ / Short Vega)"
        k_sell_ce = atm + (2 * step)
        k_buy_ce = k_sell_ce + (2 * step)
        k_sell_pe = atm - (2 * step)
        k_buy_pe = k_sell_pe - (2 * step)

        p_sce, d_sce, g_sce, t_sce, v_sce = black_scholes_full_greeks(spot, k_sell_ce, T, r, sigma, "CE")
        p_bce, d_bce, g_bce, t_bce, v_bce = black_scholes_full_greeks(spot, k_buy_ce, T, r, sigma, "CE")
        p_spe, d_spe, g_spe, t_spe, v_spe = black_scholes_full_greeks(spot, k_sell_pe, T, r, sigma, "PE")
        p_bpe, d_bpe, g_bpe, t_bpe, v_bpe = black_scholes_full_greeks(spot, k_buy_pe, T, r, sigma, "PE")

        net_credit = max((p_sce - p_bce) + (p_spe - p_bpe), 1.5)
        wing_width = 2 * step
        risk_per_lot = max((wing_width - net_credit) * lot_size, 500.0)
        reward_per_lot = net_credit * lot_size

        lots = max(1, int(max_allowed_risk // risk_per_lot))
        max_risk = round(risk_per_lot * lots, 2)
        max_reward = round(reward_per_lot * lots, 2)
        breakeven = f"{round(k_sell_pe - net_credit, 1)} – {round(k_sell_ce + net_credit, 1)}"

        net_delta = round((-d_sce + d_bce - d_spe + d_bpe) * lot_size * lots, 2)
        net_gamma = round((-g_sce + g_bce - g_spe + g_bpe) * lot_size * lots, 4)
        net_theta = round((-t_sce + t_bce - t_spe + t_bpe) * lot_size * lots, 2)
        net_vega = round((-v_sce + v_bce - v_spe + v_bpe) * lot_size * lots, 2)

        legs = [
            {"action": "SELL", "instrument": f"{instrument_key} {k_sell_ce} CE", "strike": k_sell_ce, "moneyness": classify_moneyness(spot, k_sell_ce, "CE", step), "type": "CE", "premium": p_sce, "delta": d_sce, "gamma": g_sce, "vega": v_sce, "lots": lots},
            {"action": "BUY", "instrument": f"{instrument_key} {k_buy_ce} CE", "strike": k_buy_ce, "moneyness": classify_moneyness(spot, k_buy_ce, "CE", step), "type": "CE", "premium": p_bce, "delta": d_bce, "gamma": g_bce, "vega": v_bce, "lots": lots},
            {"action": "SELL", "instrument": f"{instrument_key} {k_sell_pe} PE", "strike": k_sell_pe, "moneyness": classify_moneyness(spot, k_sell_pe, "PE", step), "type": "PE", "premium": p_spe, "delta": d_spe, "gamma": g_spe, "vega": v_spe, "lots": lots},
            {"action": "BUY", "instrument": f"{instrument_key} {k_buy_pe} PE", "strike": k_buy_pe, "moneyness": classify_moneyness(spot, k_buy_pe, "PE", step), "type": "PE", "premium": p_bpe, "delta": d_bpe, "gamma": g_bpe, "vega": v_bpe, "lots": lots},
        ]

    greek_insight = (
        f"Net Delta {net_delta:+.1f} | Net Gamma {net_gamma:+.4f} | "
        f"Net Vega ₹{net_vega:+.1f}/1% VIX ({'benefits from VIX drop' if net_vega < 0 else 'benefits from VIX rise'})."
    )

    price_grid = np.linspace(spot * 0.94, spot * 1.06, 60)
    payoff_grid = []
    for px in price_grid:
        pnl = 0.0
        for leg in legs:
            intrinsic = max(px - leg["strike"], 0.0) if leg["type"] == "CE" else max(leg["strike"] - px, 0.0)
            mult = 1.0 if leg["action"] == "BUY" else -1.0
            pnl += mult * (intrinsic - leg["premium"]) * lot_size * leg["lots"]
        payoff_grid.append(round(pnl, 2))

    return {
        "segment": "OPTIONS",
        "regime": regime_name,
        "strategy": strategy_name,
        "legs": legs,
        "lots": lots,
        "lot_size": lot_size,
        "unit_label": "Lots",
        "atm_strike": atm,
        "max_risk": max_risk,
        "max_reward": max_reward,
        "breakeven": breakeven,
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_theta": net_theta,
        "net_vega": net_vega,
        "greek_insight": greek_insight,
        "payoff_prices": [round(float(x), 1) for x in price_grid],
        "payoff_pnl": payoff_grid,
    }
