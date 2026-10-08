#!/usr/bin/env python3
"""
Crypto Arbitrage Scanner — Cross Exchange + Triangular + P2P
=====================================================
Exchanges: Binance, OKX, Bybit, Bitget, KuCoin, MEXC, BingX
Spot + USDT Futures/Swaps | Liquidity filter | D/W + network fees

Default filter: estimated profit >= 4.0%
NOTE: This is a scanner only; it does NOT place trades.

USAGE
-----
  streamlit run arbitrage_scanner.py
  python arbitrage_scanner.py
  python arbitrage_scanner.py --min-profit 4
  python arbitrage_scanner.py --loop 60

Install:
  pip install ccxt pandas streamlit streamlit-autorefresh
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import ccxt

# =============================================================================
# CONFIG
# =============================================================================
EXCHANGES = [
    "binance", "okx", "bybit", "bitget",
    "kucoin", "mexc", "bingx",
]

MIN_QUOTE_VOLUME_USD = 5_000  # Cross-exchange minimum liquidity
MIN_EST_PROFIT_PCT = 4.0
MAX_SPREAD_PCT = 15.0
MAX_WORKERS = 8
TIMEOUT = 20

# Rough taker-fee estimate per side.
# This is NOT exchange-specific and must be verified before trading.
FEE_PCT_PER_SIDE = 0.10


# =============================================================================
# DATA
# =============================================================================
@dataclass
class TickerInfo:
    exchange: str
    symbol: str
    market_type: str
    bid: float
    ask: float
    last: float
    quote_volume: float
    base_volume: float = 0.0


@dataclass
class CurrencyStatus:
    exchange: str
    code: str
    deposit: bool | None = None
    withdraw: bool | None = None
    networks: list[dict] = field(default_factory=list)


@dataclass
class ArbOpportunity:
    symbol: str
    market_type: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    buy_volume: float
    sell_volume: float
    est_profit_pct: float
    buy_currency: CurrencyStatus | None = None
    sell_currency: CurrencyStatus | None = None


# =============================================================================
# CORE ENGINE
# =============================================================================
def _make_exchange(ex_id: str, market_type: str = "spot") -> ccxt.Exchange:
    opts: dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": TIMEOUT * 1000,
        "options": {},
    }
    if market_type == "swap":
        opts["options"]["defaultType"] = "swap"
        if ex_id == "binance":
            opts["options"]["defaultType"] = "future"
    return getattr(ccxt, ex_id)(opts)


def _safe_float(v, default=0.0) -> float:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def fetch_tickers_one(ex_id: str, market_type: str) -> list[TickerInfo]:
    out: list[TickerInfo] = []
    try:
        ex = _make_exchange(ex_id, market_type)
        try:
            markets = ex.load_markets()
        except Exception:
            markets = {}

        tickers = ex.fetch_tickers()

        for symbol, t in tickers.items():
            if not symbol.endswith("/USDT") and not symbol.endswith(":USDT"):
                if "/USDT:" not in symbol and ":USDT" not in symbol:
                    continue

            base_symbol = symbol.split(":")[0] if ":" in symbol else symbol
            if not base_symbol.endswith("/USDT"):
                continue

            m = markets.get(symbol) or markets.get(base_symbol)
            if m:
                if market_type == "spot" and not m.get("spot", False) and m.get("type") not in (None, "spot"):
                    continue
                if market_type == "swap" and not (
                    m.get("swap") or m.get("future") or m.get("type") in ("swap", "future")
                ):
                    continue

            bid = _safe_float(t.get("bid"))
            ask = _safe_float(t.get("ask"))
            last = _safe_float(t.get("last"))

            if bid <= 0 and ask <= 0 and last <= 0:
                continue
            if bid <= 0:
                bid = last
            if ask <= 0:
                ask = last
            if bid <= 0 or ask <= 0:
                continue

            qv = _safe_float(t.get("quoteVolume"))
            bv = _safe_float(t.get("baseVolume"))
            if qv <= 0 and bv > 0 and last > 0:
                qv = bv * last
            if qv < MIN_QUOTE_VOLUME_USD:
                continue

            out.append(
                TickerInfo(
                    exchange=ex_id,
                    symbol=base_symbol,
                    market_type=market_type,
                    bid=bid,
                    ask=ask,
                    last=last,
                    quote_volume=qv,
                    base_volume=bv,
                )
            )

    except Exception as e:
        print(
            f"[warn] {ex_id} {market_type}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )

    return out


def fetch_all_tickers(market_types: list[str] | None = None) -> list[TickerInfo]:
    if market_types is None:
        market_types = ["spot", "swap"]

    tasks = [(ex, mt) for ex in EXCHANGES for mt in market_types]
    results: list[TickerInfo] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(fetch_tickers_one, ex, mt): (ex, mt)
            for ex, mt in tasks
        }

        for fut in as_completed(futs):
            try:
                results.extend(fut.result())
            except Exception as e:
                print(f"[warn] worker: {e}", file=sys.stderr)

    return results


def find_opportunities(
    tickers: list[TickerInfo],
    min_est_profit_pct: float = MIN_EST_PROFIT_PCT,
    max_spread_pct: float = MAX_SPREAD_PCT,
) -> list[ArbOpportunity]:

    groups: dict[tuple[str, str], list[TickerInfo]] = {}

    for t in tickers:
        groups.setdefault((t.symbol, t.market_type), []).append(t)

    opps: list[ArbOpportunity] = []

    for (symbol, mtype), rows in groups.items():
        if len(rows) < 2:
            continue

        for buy in rows:
            for sell in rows:
                if buy.exchange == sell.exchange:
                    continue

                if buy.ask <= 0 or sell.bid <= 0 or sell.bid <= buy.ask:
                    continue

                spread = (sell.bid - buy.ask) / buy.ask * 100

                if spread > max_spread_pct:
                    continue

                est_profit = spread - (FEE_PCT_PER_SIDE * 2)

                # MAIN FILTER: estimated profit must be >= user's threshold.
                if est_profit < min_est_profit_pct:
                    continue

                # Extra sanity check against extreme ticker mismatches.
                if sell.bid / buy.ask > 1.0 + max_spread_pct / 100.0:
                    continue

                opps.append(
                    ArbOpportunity(
                        symbol=symbol,
                        market_type=mtype,
                        buy_exchange=buy.exchange,
                        sell_exchange=sell.exchange,
                        buy_price=buy.ask,
                        sell_price=sell.bid,
                        spread_pct=spread,
                        buy_volume=buy.quote_volume,
                        sell_volume=sell.quote_volume,
                        est_profit_pct=est_profit,
                    )
                )

    opps.sort(key=lambda x: x.est_profit_pct, reverse=True)

    seen = set()
    unique = []

    for o in opps:
        k = (o.symbol, o.market_type, o.buy_exchange, o.sell_exchange)
        if k not in seen:
            seen.add(k)
            unique.append(o)

    return unique


def fetch_currency_status(ex_id: str, codes: set[str]) -> dict[str, CurrencyStatus]:
    result: dict[str, CurrencyStatus] = {}

    try:
        ex = _make_exchange(ex_id, "spot")

        if not ex.has.get("fetchCurrencies"):
            return result

        currencies = ex.fetch_currencies()

        for code in codes:
            c = currencies.get(code)
            if not c:
                continue

            st = CurrencyStatus(
                exchange=ex_id,
                code=code,
                deposit=c.get("deposit"),
                withdraw=c.get("withdraw"),
            )

            nets = c.get("networks") or {}

            for net_name, net in nets.items():
                if not isinstance(net, dict):
                    continue

                fee = net.get("fee")

                if fee is None and isinstance(net.get("withdraw"), dict):
                    fee = net["withdraw"].get("fee")

                st.networks.append(
                    {
                        "network": net_name,
                        "fee": fee,
                        "deposit": net.get("deposit", c.get("deposit")),
                        "withdraw": (
                            net.get("withdraw", c.get("withdraw"))
                            if not isinstance(net.get("withdraw"), dict)
                            else net.get("active")
                        ),
                        "min": (
                            net.get("limits", {})
                            .get("withdraw", {})
                            .get("min")
                            if isinstance(net.get("limits"), dict)
                            else net.get("withdrawMin")
                        ),
                    }
                )

            if not st.networks and c.get("fee") is not None:
                st.networks.append(
                    {
                        "network": "default",
                        "fee": c.get("fee"),
                        "deposit": c.get("deposit"),
                        "withdraw": c.get("withdraw"),
                        "min": None,
                    }
                )

            result[code] = st

    except Exception as e:
        print(
            f"[warn] currencies {ex_id}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )

    return result


def enrich_with_currency_status(
    opps: list[ArbOpportunity],
    max_symbols: int = 30,
) -> list[ArbOpportunity]:

    top = opps[:max_symbols]
    need: dict[str, set[str]] = {}

    for o in top:
        base = o.symbol.split("/")[0]
        need.setdefault(o.buy_exchange, set()).add(base)
        need.setdefault(o.sell_exchange, set()).add(base)

    cache: dict[str, dict[str, CurrencyStatus]] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(fetch_currency_status, ex, codes): ex
            for ex, codes in need.items()
        }

        for fut in as_completed(futs):
            ex = futs[fut]
            try:
                cache[ex] = fut.result()
            except Exception:
                cache[ex] = {}

    for o in top:
        base = o.symbol.split("/")[0]
        o.buy_currency = cache.get(o.buy_exchange, {}).get(base)
        o.sell_currency = cache.get(o.sell_exchange, {}).get(base)

    return opps


def format_network_fees(st: CurrencyStatus | None) -> str:
    if not st:
        return "N/A"

    dep = "Y" if st.deposit else ("N" if st.deposit is False else "?")
    wd = "Y" if st.withdraw else ("N" if st.withdraw is False else "?")

    parts = [f"D:{dep}/W:{wd}"]

    for n in (st.networks or [])[:3]:
        fee = n.get("fee")
        parts.append(
            f"{n.get('network', '?')}:fee={fee if fee is not None else '?'}"
        )

    return " | ".join(parts)


def run_scan(
    market_types: list[str] | None = None,
    min_est_profit_pct: float = MIN_EST_PROFIT_PCT,
    max_spread_pct: float = MAX_SPREAD_PCT,
    min_volume: float = MIN_QUOTE_VOLUME_USD,
    enrich: bool = True,
    top_n: int = 40,
) -> tuple[list[ArbOpportunity], list[TickerInfo]]:

    global MIN_QUOTE_VOLUME_USD

    old = MIN_QUOTE_VOLUME_USD
    MIN_QUOTE_VOLUME_USD = min_volume

    try:
        tickers = fetch_all_tickers(market_types)

        opps = find_opportunities(
            tickers,
            min_est_profit_pct=min_est_profit_pct,
            max_spread_pct=max_spread_pct,
        )

        if enrich and opps:
            enrich_with_currency_status(
                opps,
                max_symbols=min(top_n, 40),
            )

        return opps[:top_n], tickers

    finally:
        MIN_QUOTE_VOLUME_USD = old


# =============================================================================
# CLI MODE
# =============================================================================
def run_cli():
    p = argparse.ArgumentParser(
        description="Cross-exchange crypto arbitrage scanner"
    )

    p.add_argument(
        "--min-profit",
        type=float,
        default=4.0,
        help="Minimum estimated profit percentage",
    )
    p.add_argument("--max-spread", type=float, default=15.0)
    p.add_argument("--min-vol", type=float, default=5000)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--spot-only", action="store_true")
    p.add_argument("--swap-only", action="store_true")
    p.add_argument("--no-enrich", action="store_true")
    p.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Rescan every N seconds (0=once)",
    )

    args = p.parse_args()

    if args.spot_only:
        mtypes = ["spot"]
    elif args.swap_only:
        mtypes = ["swap"]
    else:
        mtypes = ["spot", "swap"]

    while True:
        t0 = time.time()

        print("\n" + "=" * 100)
        print(
            f" SCAN {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        print(f" Exchanges: {', '.join(EXCHANGES)}")
        print(
            f" Markets: {mtypes} | min vol ${args.min_vol:,.0f} | "
            f"MIN EST PROFIT {args.min_profit:.2f}% | "
            f"MAX SPREAD {args.max_spread:.2f}%"
        )
        print("=" * 100)

        opps, tickers = run_scan(
            market_types=mtypes,
            min_est_profit_pct=args.min_profit,
            max_spread_pct=args.max_spread,
            min_volume=args.min_vol,
            enrich=not args.no_enrich,
            top_n=args.top,
        )

        print(f" Tickers: {len(tickers)} | Opportunities: {len(opps)}")
        print("-" * 100)

        for i, o in enumerate(opps, 1):
            print(
                f"{i:>3} {o.market_type:<5} {o.symbol:<14} "
                f"BUY {o.buy_exchange:<9} {o.buy_price:>12.6g} | "
                f"SELL {o.sell_exchange:<9} {o.sell_price:>12.6g} | "
                f"Spread {o.spread_pct:>6.2f}% | "
                f"Est {o.est_profit_pct:>6.2f}%"
            )

        if not opps:
            print(" No opportunities with estimated profit >= threshold.")

        print("-" * 100)
        print(f" Done in {time.time() - t0:.1f}s")
        print(
            " WARNING: Est% uses a rough 0.10% fee per side. "
            "Real fees, slippage, funding and network costs can reduce profit."
        )

        if args.loop <= 0:
            break

        print(f" Sleeping {args.loop}s...")
        time.sleep(args.loop)


# =============================================================================
# TRIANGULAR ARBITRAGE (single exchange, spot, starts/ends in USDT)
# =============================================================================
def fetch_spot_book(ex_id: str) -> dict[str, tuple]:
    ex = _make_exchange(ex_id, "spot")
    ex.load_markets()
    out = {}
    for sym, t in ex.fetch_tickers().items():
        m = ex.markets.get(sym)
        if not m or not m.get("spot") or m.get("active") is False:
            continue
        bid, ask = _safe_float(t.get("bid")), _safe_float(t.get("ask"))
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        out[sym] = (m["base"], m["quote"], bid, ask, _safe_float(t.get("quoteVolume")))
    return out


def find_triangles(ex_id, book, fee_pct, min_profit, min_vol, max_profit=30.0):
    usdt = {v[0]: v for v in book.values() if v[1] == "USDT"}
    f = 1 - fee_pct / 100
    res = []
    for A, B, bid, ask, qv in book.values():
        if B == "USDT" or A not in usdt or B not in usdt:
            continue
        a, b = usdt[A], usdt[B]
        vol = min(a[4], b[4], qv * b[2])
        if vol < min_vol:
            continue
        p1 = (1 / b[3]) * f * (1 / ask) * f * a[2] * f   # USDT->B->A->USDT
        p2 = (1 / a[3]) * f * bid * f * b[2] * f          # USDT->A->B->USDT
        for path, val in ((f"USDT → {B} → {A} → USDT", p1),
                          (f"USDT → {A} → {B} → USDT", p2)):
            pct = (val - 1) * 100
            if min_profit <= pct <= max_profit:
                res.append({"Exchange": ex_id, "Path": path,
                            "Net profit %": round(pct, 3),
                            "Min leg vol $": round(vol)})
    return res


def run_triangular(exchanges, fee_pct, min_profit, min_vol, top_n):
    rows, errs = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(fetch_spot_book, e): e for e in exchanges}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                rows += find_triangles(e, fut.result(), fee_pct, min_profit, min_vol)
            except Exception as ex:
                errs.append(f"{e}: {type(ex).__name__}")
    rows.sort(key=lambda r: r["Net profit %"], reverse=True)
    return rows[:top_n], errs


# =============================================================================
# P2P (Binance P2P public ads)
# =============================================================================
P2P_URL = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"


def fetch_p2p(fiat, asset, trade_type, pay_types=None, rows=20):
    import json
    import urllib.request
    body = {"fiat": fiat, "page": 1, "rows": rows, "tradeType": trade_type,
            "asset": asset, "countries": [], "proMerchantAds": False,
            "shieldMerchantAds": False, "publisherType": None,
            "payTypes": pay_types or [], "classifies": ["mass", "profession"]}
    req = urllib.request.Request(
        P2P_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)
    ads = []
    for d in data.get("data") or []:
        adv, usr = d.get("adv", {}), d.get("advertiser", {})
        ads.append({
            "Price": _safe_float(adv.get("price")),
            "Min": _safe_float(adv.get("minSingleTransAmount")),
            "Max": _safe_float(adv.get("dynamicMaxSingleTransAmount")
                               or adv.get("maxSingleTransAmount")),
            "Available": _safe_float(adv.get("tradableQuantity")),
            "Merchant": usr.get("nickName", "?"),
            "Orders/month": int(_safe_float(usr.get("monthOrderCount"))),
            "Completion %": round(_safe_float(usr.get("monthFinishRate")) * 100, 1),
            "Payment": ", ".join(m.get("tradeMethodName") or m.get("identifier", "?")
                                 for m in adv.get("tradeMethods", [])),
        })
    return ads


def p2p_filter(ads, min_orders, min_rate, max_min_limit):
    return [a for a in ads if a["Orders/month"] >= min_orders
            and a["Completion %"] >= min_rate and a["Min"] <= max_min_limit]


# =============================================================================
# STREAMLIT DASHBOARD (3 tabs)
# =============================================================================
def run_streamlit():
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Arbitrage Scanner", page_icon="⚡", layout="wide")
    ss = st.session_state

    # Automatic reruns every 60 seconds. Install streamlit-autorefresh for the timer.
    try:
        from streamlit_autorefresh import st_autorefresh
        auto_supported = True
    except ImportError:
        auto_supported = False
        st_autorefresh = None

    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg, rgba(16,24,48,.04), rgba(0,180,160,.07));}
    .hero {padding: 1.1rem 1.4rem; border-radius: 16px;
           background: linear-gradient(110deg,#172554,#075985,#0f766e);
           color: white; margin-bottom: .7rem;}
    .hero h1 {color:white; margin:0;}
    div[data-testid="stMetric"] {background:rgba(14,165,233,.08);
      border:1px solid rgba(14,165,233,.25); padding:12px; border-radius:12px;}
    div.stButton > button[kind="primary"] {background:linear-gradient(90deg,#2563eb,#7c3aed);
      border:0; color:white; border-radius:10px;}
    </style>
    <div class="hero"><h1>⚡ Crypto Arbitrage Radar</h1>
    <p>Cross-exchange • Triangular routes • P2P ads | Live opportunity monitor</p></div>
    """, unsafe_allow_html=True)
    st.caption("Read-only scanner — it does not place trades. Prices and ads can change quickly; verify fees, liquidity, and payment details before acting.")

    h1, h2, h3 = st.columns([1.2, 1, 2])
    auto_scan = h1.toggle("Auto-scan every 60 sec", value=True, key="auto_scan")
    h2.metric("Refresh interval", "60 seconds" if auto_scan else "Manual")
    if auto_scan and auto_supported:
        tick_count = st_autorefresh(interval=60_000, key="arb_autorefresh")
        auto_due = ss.get("_last_auto_count") != tick_count
        ss["_last_auto_count"] = tick_count
    else:
        auto_due = False
        if auto_scan and not auto_supported:
            st.warning("For automatic 60-second refresh, install: pip install streamlit-autorefresh")
    if "last_scan_time" in ss:
        h3.caption(f"Last completed scan: {ss['last_scan_time']}")
    else:
        h3.caption("No scan completed yet. The first scan starts automatically when auto-scan is enabled.")

    t_tri, t_cross, t_p2p = st.tabs(["🔺 Triangular (priority)", "🌐 Cross Exchange", "🤝 P2P Ads"])

    # ---------------- Triangular ----------------
    with t_tri:
        st.subheader("🔺 Triangular arbitrage — same-exchange routes")
        st.caption("Looks for USDT → coin → coin → USDT routes. Results are estimates using best bid/ask, not guaranteed fills.")
        exs = st.multiselect("Exchanges", EXCHANGES, default=EXCHANGES[:4], key="te")
        c1, c2, c3 = st.columns(3)
        fee = c1.number_input("Fee % per leg", 0.0, 1.0, 0.10, 0.01, key="tf")
        tmin = c2.number_input("Min net profit %", 0.0, 20.0, 0.3, 0.1, key="tp")
        tvol = c3.number_input("Min volume $", 1000, value=20000, step=1000, key="tv")
        tri_auto_key = (tick_count if auto_scan and auto_supported else None)
        if st.button("🔺 Scan triangular now", type="primary", key="tb") or (auto_due and auto_scan):
            with st.spinner("Scanning triangular routes across selected exchanges..."):
                ss["tri"] = run_triangular(exs, float(fee), float(tmin), float(tvol), 50)
                ss["last_scan_time"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        if "tri" in ss:
            rows, errs = ss["tri"]
            for e in errs:
                st.caption(f"⚠️ {e}")
            if rows:
                df_tri = pd.DataFrame(rows)
                st.dataframe(df_tri.style.format({"Net profit %": "{:.3f}%"}), use_container_width=True, hide_index=True)
            else:
                st.info("No triangular opportunities matched the filters on the latest scan. This is common.")
        st.caption("Important: order-book depth, slippage, exchange-specific fees, and minimum order sizes can change real results.")

    # ---------------- Cross Exchange ----------------
    with t_cross:
        st.subheader("🌐 Cross-exchange arbitrage")
        c1, c2, c3 = st.columns(3)
        min_vol = c1.number_input("Min volume $", 1000, value=5000, step=1000, key="cv")
        min_profit = c2.number_input("Min est. profit %", 0.1, 20.0, 4.0, 0.1, key="cp")
        max_spread = c3.number_input("Max raw spread %", 5.0, 50.0, 15.0, 1.0, key="cs")
        market = st.radio("Market", ["Both", "Spot only", "Futures/Swap only"], horizontal=True, key="cm")
        enrich = st.checkbox("Check deposit/withdraw + network fees", True, key="ce")
        top_n = st.slider("Max rows", 10, 100, 30, key="ct")
        if st.button("🌐 Scan cross-exchange now", type="primary", key="cb") or (auto_due and auto_scan):
            mt = {"Both": ["spot", "swap"], "Spot only": ["spot"]}.get(market, ["swap"])
            with st.spinner("Fetching tickers from 7 exchanges..."):
                t0 = time.time()
                opps, tk = run_scan(mt, float(min_profit), float(max_spread), float(min_vol), enrich, top_n)
            ss["cross"] = (opps, len(tk), time.time() - t0)
            ss["last_scan_time"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        if "cross" in ss:
            opps, n, el = ss["cross"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Tickers checked", f"{n:,}")
            m2.metric("Matching opportunities", f"{len(opps)}")
            m3.metric("Scan duration", f"{el:.1f}s")
            if opps:
                df_cross = pd.DataFrame([{
                    "Market": o.market_type, "Pair": o.symbol,
                    "Buy exchange": o.buy_exchange.upper(), "Sell exchange": o.sell_exchange.upper(),
                    "Buy price": o.buy_price, "Sell price": o.sell_price,
                    "Spread %": round(o.spread_pct, 3), "Est. net %": round(o.est_profit_pct, 3),
                    "Buy volume $": round(o.buy_volume), "Sell volume $": round(o.sell_volume),
                    "Buy D/W + network": format_network_fees(o.buy_currency),
                    "Sell D/W + network": format_network_fees(o.sell_currency),
                } for o in opps])
                st.dataframe(df_cross.style.background_gradient(subset=["Est. net %"], cmap="YlGn"),
                             use_container_width=True, hide_index=True)
            else:
                st.warning("No opportunities match the current filters. Try a lower profit threshold, but account for real fees.")
        st.caption("Estimated fees currently use a rough 0.10% per side; verify each exchange's actual trading fees.")

    # ---------------- P2P ----------------
    with t_p2p:
        st.subheader("🤝 P2P ads and fiat spread")
        st.info("P2P ad source: Binance P2P public ads. The exchange is shown with every ad; this version does not claim to scan P2P ads on other exchanges.")
        c1, c2 = st.columns(2)
        fiat = c1.text_input("Fiat currency", "PKR", key="pf").upper().strip()
        assets = c2.multiselect("Assets", ["USDT", "USDC", "BTC", "ETH", "BNB"], default=["USDT"], key="pa")
        pay = st.text_input("Payment method (optional, e.g. JazzCash, EasyPaisa)", "", key="pp")
        c1, c2, c3 = st.columns(3)
        mo = c1.number_input("Min merchant orders/month", 0, value=100, step=50, key="po")
        mr = c2.number_input("Min completion %", 0.0, 100.0, 95.0, 1.0, key="pr")
        ml = c3.number_input("Max minimum-order limit", 0, value=1_000_000, step=10000, key="pl")
        if st.button("🤝 Scan P2P ads now", type="primary", key="pb") or (auto_due and auto_scan):
            out = {}
            with st.spinner("Fetching Binance P2P ads..."):
                for a in assets:
                    try:
                        pt = [pay.strip()] if pay.strip() else None
                        buy = p2p_filter(fetch_p2p(fiat, a, "BUY", pt), mo, mr, ml)
                        sell = p2p_filter(fetch_p2p(fiat, a, "SELL", pt), mo, mr, ml)
                        out[a] = (buy, sell, None)
                    except Exception as e:
                        out[a] = ([], [], f"{type(e).__name__}: {e}")
            ss["p2p"] = (fiat, out)
            ss["last_scan_time"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        if "p2p" in ss:
            fiat_s, out = ss["p2p"]
            for a, (buy, sell, err) in out.items():
                st.markdown(f"### 🟣 Binance P2P — {a}/{fiat_s}")
                if err:
                    st.error(err)
                    continue
                if not buy or not sell:
                    st.warning("No ads match your filters.")
                    continue
                bp = min(x["Price"] for x in buy)
                sp = max(x["Price"] for x in sell)
                pct = (sp - bp) / bp * 100 if bp else 0
                m1, m2, m3 = st.columns(3)
                m1.metric("Best BUY ad", f"{bp:,.2f} {fiat_s}")
                m2.metric("Best SELL ad", f"{sp:,.2f} {fiat_s}")
                m3.metric("Indicative spread", f"{pct:.2f}%")
                if pct > 0:
                    st.success("Positive ad spread detected — compare payment methods, limits, fees, and transfer risks before acting.")
                else:
                    st.info("No positive ad spread after this comparison.")
                buy_df = pd.DataFrame(sorted(buy, key=lambda x: x["Price"]))
                sell_df = pd.DataFrame(sorted(sell, key=lambda x: -x["Price"]))
                buy_df.insert(0, "Exchange", "Binance")
                sell_df.insert(0, "Exchange", "Binance")
                with st.expander("🟢 Ads to BUY from (cheapest first)", expanded=True):
                    st.dataframe(buy_df, use_container_width=True, hide_index=True)
                with st.expander("🔴 Ads to SELL to (highest first)", expanded=True):
                    st.dataframe(sell_df, use_container_width=True, hide_index=True)


# =============================================================================
# ENTRY
# =============================================================================
def _is_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _is_streamlit() or any("streamlit" in a for a in sys.argv):
        run_streamlit()
    else:
        run_cli()
