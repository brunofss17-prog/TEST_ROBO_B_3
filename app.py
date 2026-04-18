from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def to_list(series):
    """Converte pd.Series para lista Python (NaN → None)."""
    return [round(float(v), 4) if pd.notna(v) else None for v in series]

def last(s):
    c = s.dropna()
    return float(c.iloc[-1]) if len(c) else None

def prev(s):
    c = s.dropna()
    return float(c.iloc[-2]) if len(c) > 1 else None

# ─────────────────────────────────────────────────────────────────
#  INDICADORES
# ─────────────────────────────────────────────────────────────────

def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_sma(s, n):
    return s.rolling(n).mean()

def calc_rsi(s, n):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_macd(s, f=12, sl=26, sig=9):
    ml = calc_ema(s, f) - calc_ema(s, sl)
    sl_ = calc_ema(ml, sig)
    return ml, sl_, ml - sl_

def calc_bollinger(s, n=20, k=2):
    m = calc_sma(s, n)
    std = s.rolling(n).std()
    return m + k * std, m, m - k * std

def calc_medias(s):
    return calc_ema(s, 9), calc_ema(s, 21), calc_sma(s, 50), calc_sma(s, 200)

# ─────────────────────────────────────────────────────────────────
#  TENDÊNCIA (médias móveis)
# ─────────────────────────────────────────────────────────────────

def analisar_tendencia(closes, mme9, mme21, mms50, mms200):
    p   = float(closes.iloc[-1])
    v9  = last(mme9);  v21 = last(mme21)
    v50 = last(mms50); v200 = last(mms200)

    pts = 0
    razoes = []

    if v9 and v21:
        if v9 > v21:
            pts += 1
            razoes.append({"txt": "MME9 acima da MME21 — momentum de curto prazo positivo", "tipo": "alta"})
        else:
            pts -= 1
            razoes.append({"txt": "MME9 abaixo da MME21 — momentum de curto prazo negativo", "tipo": "baixa"})
        if p > v21:
            pts += 1
            razoes.append({"txt": f"Preço (R${p:.2f}) acima da MME21 (R${v21:.2f}) — tendência intermediária de alta", "tipo": "alta"})
        else:
            pts -= 1
            razoes.append({"txt": f"Preço (R${p:.2f}) abaixo da MME21 (R${v21:.2f}) — tendência intermediária de baixa", "tipo": "baixa"})

    if v50:
        if p > v50:
            pts += 1
            razoes.append({"txt": f"Preço acima da MMS50 (R${v50:.2f}) — tendência de médio prazo positiva", "tipo": "alta"})
        else:
            pts -= 1
            razoes.append({"txt": f"Preço abaixo da MMS50 (R${v50:.2f}) — pressão vendedora de médio prazo", "tipo": "baixa"})

    if v200:
        if p > v200:
            pts += 2
            razoes.append({"txt": f"Preço acima da MMS200 (R${v200:.2f}) — regime de bull market confirmado", "tipo": "alta"})
        else:
            pts -= 2
            razoes.append({"txt": f"Preço abaixo da MMS200 (R${v200:.2f}) — regime de bear market, cautela máxima", "tipo": "baixa"})

    if v9 and v21 and v50 and v200:
        if v9 > v21 > v50 > v200:
            pts += 2
            razoes.append({"txt": "Médias em ordem perfeita crescente — tendência de alta forte e consistente", "tipo": "alta"})
        elif v9 < v21 < v50 < v200:
            pts -= 2
            razoes.append({"txt": "Médias em ordem perfeita decrescente — tendência de baixa forte e consistente", "tipo": "baixa"})

    mx = 7 if v200 else 4
    forca = round(abs(pts) / mx * 100)

    if   pts >= 5: label, cor = "ALTA FORTE",  "alta-forte"
    elif pts >= 3: label, cor = "ALTA",         "alta"
    elif pts >= 1: label, cor = "LEVE ALTA",    "leve-alta"
    elif pts <= -5:label, cor = "BAIXA FORTE",  "baixa-forte"
    elif pts <= -3:label, cor = "BAIXA",        "baixa"
    elif pts <= -1:label, cor = "LEVE BAIXA",   "leve-baixa"
    else:          label, cor = "LATERAL",      "lateral"

    return {
        "label": label, "cor": cor, "pts": pts, "forca": forca,
        "razoes": razoes,
        "mme9":  round(v9, 2)  if v9  else None,
        "mme21": round(v21, 2) if v21 else None,
        "mms50": round(v50, 2) if v50 else None,
        "mms200":round(v200,2) if v200 else None,
    }

# ─────────────────────────────────────────────────────────────────
#  ESTRATÉGIAS DE SINAL
# ─────────────────────────────────────────────────────────────────

def _decisao(score):
    if score >= 3:  return "COMPRA", "compra"
    if score <= -3: return "VENDA",  "venda"
    return "NEUTRO", "neutro"

def sinal_est1(closes_d, volume_d=None):
    """
    Estratégia 1 — RSI5 diário + MACD diário + Bollinger diário
    """
    rsi5   = calc_rsi(closes_d, 5)
    ml, sl, ht = calc_macd(closes_d)
    bsup, bmed, binf = calc_bollinger(closes_d)

    p      = float(closes_d.iloc[-1])
    r5     = last(rsi5)
    h      = last(ht);   h_ant = prev(ht)
    sup    = last(bsup); inf   = last(binf); med = last(bmed)

    sc = 0; detalhes = []

    # RSI5
    if r5 is not None:
        if r5 < 30:
            sc += 2; detalhes.append({"l": f"RSI5 {r5:.1f} — sobrevendido", "t": "compra"})
        elif r5 > 70:
            sc -= 2; detalhes.append({"l": f"RSI5 {r5:.1f} — sobrecomprado", "t": "venda"})
        else:
            detalhes.append({"l": f"RSI5 {r5:.1f} — neutro", "t": "neutro"})

    # MACD
    if h is not None and h_ant is not None:
        if h > 0 and h_ant <= 0:
            sc += 2; detalhes.append({"l": "MACD cruzamento altista ↑", "t": "compra"})
        elif h < 0 and h_ant >= 0:
            sc -= 2; detalhes.append({"l": "MACD cruzamento baixista ↓", "t": "venda"})
        elif h > 0:
            sc += 1; detalhes.append({"l": "MACD histograma positivo", "t": "compra"})
        else:
            sc -= 1; detalhes.append({"l": "MACD histograma negativo", "t": "venda"})

    # Bollinger
    if p and sup and inf and med:
        if p <= inf:
            sc += 2; detalhes.append({"l": "Preço tocou banda inferior (suporte)", "t": "compra"})
        elif p >= sup:
            sc -= 2; detalhes.append({"l": "Preço tocou banda superior (resistência)", "t": "venda"})
        elif p < med:
            sc += 0.5; detalhes.append({"l": "Preço abaixo da média Bollinger", "t": "neutro"})
        else:
            sc -= 0.5; detalhes.append({"l": "Preço acima da média Bollinger", "t": "neutro"})

    dec, cor = _decisao(sc)
    return {
        "dec": dec, "cor": cor, "score": round(sc, 2), "detalhes": detalhes,
        "rsi": round(r5, 2) if r5 else None,
        "series": {
            "rsi5d":  to_list(rsi5),
            "macd_line": to_list(ml), "signal_line": to_list(sl), "hist": to_list(ht),
            "bsup": to_list(bsup), "bmed": to_list(bmed), "binf": to_list(binf),
        }
    }

def sinal_est2(closes_w):
    """
    Estratégia 2 — RSI14 semanal + MACD semanal + Bollinger semanal
    """
    rsi14  = calc_rsi(closes_w, 14)
    ml, sl, ht = calc_macd(closes_w)
    bsup, bmed, binf = calc_bollinger(closes_w)

    p      = float(closes_w.iloc[-1])
    r14    = last(rsi14)
    h      = last(ht);   h_ant = prev(ht)
    sup    = last(bsup); inf   = last(binf); med = last(bmed)

    sc = 0; detalhes = []

    if r14 is not None:
        if r14 < 35:
            sc += 2; detalhes.append({"l": f"RSI14 {r14:.1f} — sobrevendido (semanal)", "t": "compra"})
        elif r14 > 65:
            sc -= 2; detalhes.append({"l": f"RSI14 {r14:.1f} — sobrecomprado (semanal)", "t": "venda"})
        else:
            detalhes.append({"l": f"RSI14 {r14:.1f} — neutro (semanal)", "t": "neutro"})

    if h is not None and h_ant is not None:
        if h > 0 and h_ant <= 0:
            sc += 2; detalhes.append({"l": "MACD semanal cruzamento altista ↑", "t": "compra"})
        elif h < 0 and h_ant >= 0:
            sc -= 2; detalhes.append({"l": "MACD semanal cruzamento baixista ↓", "t": "venda"})
        elif h > 0:
            sc += 1; detalhes.append({"l": "MACD semanal positivo", "t": "compra"})
        else:
            sc -= 1; detalhes.append({"l": "MACD semanal negativo", "t": "venda"})

    if p and sup and inf and med:
        if p <= inf:
            sc += 2; detalhes.append({"l": "Preço na banda inferior semanal", "t": "compra"})
        elif p >= sup:
            sc -= 2; detalhes.append({"l": "Preço na banda superior semanal", "t": "venda"})
        elif p < med:
            sc += 0.5; detalhes.append({"l": "Preço abaixo da média Bollinger semanal", "t": "neutro"})
        else:
            sc -= 0.5; detalhes.append({"l": "Preço acima da média Bollinger semanal", "t": "neutro"})

    dec, cor = _decisao(sc)
    return {
        "dec": dec, "cor": cor, "score": round(sc, 2), "detalhes": detalhes,
        "rsi": round(r14, 2) if r14 else None,
        "series": {
            "rsi14w": to_list(rsi14),
            "macd_line": to_list(ml), "signal_line": to_list(sl), "hist": to_list(ht),
            "bsup": to_list(bsup), "bmed": to_list(bmed), "binf": to_list(binf),
        }
    }

def sinal_est3(closes_d, volume_d):
    """
    Estratégia 3 — Médias Móveis + MACD + Volume
    """
    mme9, mme21, mms50, mms200 = calc_medias(closes_d)
    ml, sl, ht = calc_macd(closes_d)

    p    = float(closes_d.iloc[-1])
    v9   = last(mme9);  v21 = last(mme21)
    v50  = last(mms50); v200 = last(mms200)
    h    = last(ht);    h_ant = prev(ht)

    sc = 0; detalhes = []

    # Médias
    if v9 and v21:
        if v9 > v21:
            sc += 1; detalhes.append({"l": "MME9 > MME21 — momento positivo", "t": "compra"})
        else:
            sc -= 1; detalhes.append({"l": "MME9 < MME21 — momento negativo", "t": "baixa"})
    if v50 and p:
        if p > v50:
            sc += 1; detalhes.append({"l": "Preço acima da MMS50", "t": "compra"})
        else:
            sc -= 1; detalhes.append({"l": "Preço abaixo da MMS50", "t": "venda"})
    if v200 and p:
        if p > v200:
            sc += 2; detalhes.append({"l": "Preço acima da MMS200 (bull market)", "t": "compra"})
        else:
            sc -= 2; detalhes.append({"l": "Preço abaixo da MMS200 (bear market)", "t": "venda"})
    if v9 and v21 and v50 and v200:
        if v9 > v21 > v50 > v200:
            sc += 1; detalhes.append({"l": "Médias alinhadas em alta perfeita", "t": "compra"})
        elif v9 < v21 < v50 < v200:
            sc -= 1; detalhes.append({"l": "Médias alinhadas em baixa perfeita", "t": "venda"})

    # MACD
    if h is not None and h_ant is not None:
        if h > 0 and h_ant <= 0:
            sc += 2; detalhes.append({"l": "MACD cruzamento altista ↑", "t": "compra"})
        elif h < 0 and h_ant >= 0:
            sc -= 2; detalhes.append({"l": "MACD cruzamento baixista ↓", "t": "venda"})
        elif h > 0:
            sc += 1; detalhes.append({"l": "MACD histograma positivo", "t": "compra"})
        else:
            sc -= 1; detalhes.append({"l": "MACD histograma negativo", "t": "venda"})

    # Volume
    if volume_d is not None and len(volume_d.dropna()) > 20:
        vol_med = float(volume_d.rolling(20).mean().iloc[-1])
        vol_ult = float(volume_d.iloc[-1])
        if vol_ult > vol_med * 1.5:
            sc += 1; detalhes.append({"l": f"Volume {vol_ult/vol_med:.1f}x acima da média — confirmação", "t": "compra"})
        elif vol_ult < vol_med * 0.5:
            sc -= 0.5; detalhes.append({"l": "Volume muito baixo — movimento sem convicção", "t": "neutro"})

    dec, cor = _decisao(sc)
    return {
        "dec": dec, "cor": cor, "score": round(sc, 2), "detalhes": detalhes,
        "series": {
            "mme9": to_list(mme9), "mme21": to_list(mme21),
            "mms50": to_list(mms50), "mms200": to_list(mms200),
            "macd_line": to_list(ml), "signal_line": to_list(sl), "hist": to_list(ht),
            "volume": to_list(volume_d) if volume_d is not None else [],
        }
    }


def _rsi_puro(rsi_val, ob_high, ob_low):
    """
    Helper: converte valor de RSI em sinal.
    ob_high = nível sobrecomprado
    ob_low  = nível sobrevendido

    Escala de score:
      sobrevendido  (≤ob_low)          → +4  COMPRA forte
      saindo de sobrevendido (ob_low..45) → +2  COMPRA
      zona neutra baixa (45-50)        →  0  NEUTRO
      zona neutra alta  (50-55)        →  0  NEUTRO
      saindo de sobrecomprado (55..ob_high) → -2 VENDA
      sobrecomprado (≥ob_high)         → -4  VENDA forte

    _decisao usa ±3, então:
      ±4 → COMPRA/VENDA (extremo)
      ±2 → NEUTRO (zona intermediária não gera sinal sozinha)

    Para as estratégias puras de RSI usamos _decisao_rsi com limiar ±2,
    permitindo que a zona intermediária também gere sinal.
    """
    sc = 0; detalhes = []
    if rsi_val is None:
        return 0, [{"l": "RSI indisponível", "t": "neutro"}]

    if rsi_val <= ob_low:
        sc = 4
        detalhes.append({"l": f"RSI {rsi_val:.1f} — sobrevendido (≤{ob_low}) — sinal forte de compra", "t": "compra"})
    elif rsi_val >= ob_high:
        sc = -4
        detalhes.append({"l": f"RSI {rsi_val:.1f} — sobrecomprado (≥{ob_high}) — sinal forte de venda", "t": "venda"})
    elif rsi_val < 45:
        sc = 2
        detalhes.append({"l": f"RSI {rsi_val:.1f} — abaixo de 45, pressão compradora", "t": "compra"})
    elif rsi_val > 55:
        sc = -2
        detalhes.append({"l": f"RSI {rsi_val:.1f} — acima de 55, pressão vendedora", "t": "venda"})
    else:
        sc = 0
        detalhes.append({"l": f"RSI {rsi_val:.1f} — zona neutra (45–55)", "t": "neutro"})

    return sc, detalhes


def sinal_est4(closes_w):
    """
    Estratégia 4 — RSI 14 Semanal puro
    Sobrevendido <35, Sobrecomprado >65
    """
    rsi14w = calc_rsi(closes_w, 14)
    r = last(rsi14w)
    sc, detalhes = _rsi_puro(r, ob_high=65, ob_low=35)
    dec, cor = _decisao(sc)
    return {
        "dec": dec, "cor": cor, "score": round(sc, 2),
        "detalhes": detalhes,
        "rsi": round(r, 2) if r else None,
        "series": {"rsi": to_list(rsi14w)},
    }


def sinal_est5(closes_d):
    """
    Estratégia 5 — RSI 2 Diário puro
    RSI2 é extremamente sensível: sobrevendido <10, sobrecomprado >90
    Muito usado para swing trade de curtíssimo prazo.
    """
    rsi2 = calc_rsi(closes_d, 2)
    r = last(rsi2)
    sc, detalhes = _rsi_puro(r, ob_high=90, ob_low=10)
    dec, cor = _decisao(sc)
    return {
        "dec": dec, "cor": cor, "score": round(sc, 2),
        "detalhes": detalhes,
        "rsi": round(r, 2) if r else None,
        "series": {"rsi": to_list(rsi2)},
    }


def sinal_est6(closes_d):
    """
    Estratégia 6 — RSI 5 Diário puro
    Sobrevendido <30, Sobrecomprado >70
    """
    rsi5 = calc_rsi(closes_d, 5)
    r = last(rsi5)
    sc, detalhes = _rsi_puro(r, ob_high=70, ob_low=30)
    dec, cor = _decisao(sc)
    return {
        "dec": dec, "cor": cor, "score": round(sc, 2),
        "detalhes": detalhes,
        "rsi": round(r, 2) if r else None,
        "series": {"rsi": to_list(rsi5)},
    }


# ─────────────────────────────────────────────────────────────────
#  BUSCA DE DADOS + ANÁLISE COMPLETA
# ─────────────────────────────────────────────────────────────────

def buscar_dados(ticker):
    sym = ticker.upper() + (".SA" if not ticker.upper().endswith(".SA") else "")
    dd = yf.download(sym, period="5y",  interval="1d",  progress=False, auto_adjust=True)
    dw = yf.download(sym, period="10y", interval="1wk", progress=False, auto_adjust=True)
    return dd, dw

def analisar_ticker(ticker):
    try:
        dd, dw = buscar_dados(ticker)
        if dd.empty or len(dd) < 60:
            return None

        cd = dd["Close"].squeeze().dropna()
        vol_d = dd["Volume"].squeeze() if "Volume" in dd.columns else None
        cw = dw["Close"].squeeze().dropna() if not dw.empty else pd.Series(dtype=float)

        preco     = float(cd.iloc[-1])
        preco_ant = float(cd.iloc[-2])
        var       = preco - preco_ant
        var_pct   = var / preco_ant * 100

        mme9, mme21, mms50, mms200 = calc_medias(cd)
        tend = analisar_tendencia(cd, mme9, mme21, mms50, mms200)

        e1 = sinal_est1(cd)
        e2 = sinal_est2(cw) if len(cw) >= 30 else {"dec": "NEUTRO", "cor": "neutro", "score": 0, "detalhes": [], "rsi": None, "series": {}}
        e3 = sinal_est3(cd, vol_d)
        e4 = sinal_est4(cw) if len(cw) >= 16 else {"dec": "NEUTRO", "cor": "neutro", "score": 0, "detalhes": [], "rsi": None, "series": {}}
        e5 = sinal_est5(cd)
        e6 = sinal_est6(cd)

        datas_d = [d.strftime("%d/%m/%y") for d in cd.index]
        datas_w = [d.strftime("%d/%m/%y") for d in cw.index] if len(cw) else []

        # RSI5 diário e RSI14 semanal para gráficos
        rsi5d  = calc_rsi(cd, 5)
        rsi14w = calc_rsi(cw, 14) if len(cw) >= 16 else pd.Series(dtype=float)

        return {
            "ticker": ticker.upper(),
            "atualizado": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "preco": round(preco, 2),
            "var": round(var, 2),
            "var_pct": round(var_pct, 2),
            "tend": tend,
            "e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6,
            # Séries para gráficos
            "datas_d": datas_d,
            "datas_w": datas_w,
            "closes_d": to_list(cd),
            "closes_w": to_list(cw) if len(cw) else [],
            "rsi5d":   to_list(rsi5d),
            "rsi14w":  to_list(rsi14w) if len(rsi14w) else [],
            "mme9":  to_list(mme9),  "mme21": to_list(mme21),
            "mms50": to_list(mms50), "mms200":to_list(mms200),
            # MACD e Bollinger diários (para gráficos principais)
            **{k: e1["series"][k] for k in ["macd_line","signal_line","hist","bsup","bmed","binf"]},
            # Volume
            "volume_d": to_list(vol_d) if vol_d is not None else [],
        }
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────────
#  ROTAS
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analisar")
def rota_analisar():
    ticker = request.args.get("ticker", "PETR4").upper().strip()
    res = analisar_ticker(ticker)
    if res is None:
        return jsonify({"erro": f"Ticker '{ticker}' não encontrado ou dados insuficientes."})
    return jsonify(res)

@app.route("/scanner")
def rota_scanner():
    """
    Recebe lista de tickers separados por vírgula/quebra de linha
    e a estratégia desejada (1, 2 ou 3).
    """
    raw_tickers = request.args.get("tickers", "")
    estrategia  = int(request.args.get("estrategia", 1))

    tickers = [t.strip().upper() for t in raw_tickers.replace("\n", ",").split(",") if t.strip()]
    if not tickers:
        return jsonify({"erro": "Nenhum ticker informado."})

    compras = []; vendas = []; neutros = []

    def job(t):
        dd, dw = buscar_dados(t)
        if dd.empty or len(dd) < 60:
            return t, None
        cd  = dd["Close"].squeeze().dropna()
        vol_d = dd["Volume"].squeeze() if "Volume" in dd.columns else None
        cw  = dw["Close"].squeeze().dropna() if not dw.empty else pd.Series(dtype=float)

        preco     = float(cd.iloc[-1])
        preco_ant = float(cd.iloc[-2])
        var_pct   = (preco - preco_ant) / preco_ant * 100

        mme9, mme21, mms50, mms200 = calc_medias(cd)
        tend = analisar_tendencia(cd, mme9, mme21, mms50, mms200)

        if estrategia == 1:
            sig = sinal_est1(cd)
        elif estrategia == 2:
            sig = sinal_est2(cw) if len(cw) >= 30 else {"dec":"NEUTRO","cor":"neutro","score":0,"detalhes":[]}
        elif estrategia == 3:
            sig = sinal_est3(cd, vol_d)
        elif estrategia == 4:
            sig = sinal_est4(cw) if len(cw) >= 16 else {"dec":"NEUTRO","cor":"neutro","score":0,"detalhes":[]}
        elif estrategia == 5:
            sig = sinal_est5(cd)
        else:
            sig = sinal_est6(cd)

        return t, {
            "ticker": t,
            "preco": round(preco, 2),
            "var_pct": round(var_pct, 2),
            "score": sig["score"],
            "dec": sig["dec"],
            "tend": tend["label"],
            "tend_cor": tend["cor"],
            "detalhes": sig["detalhes"][:2],   # primeiros 2 motivos
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(job, t): t for t in tickers}
        for f in concurrent.futures.as_completed(futs):
            try:
                t, res = f.result(timeout=30)
                if res is None:
                    continue
                if res["dec"] == "COMPRA":
                    compras.append(res)
                elif res["dec"] == "VENDA":
                    vendas.append(res)
                else:
                    neutros.append(res)
            except Exception:
                pass

    compras.sort(key=lambda x: x["score"], reverse=True)
    vendas.sort(key=lambda x: x["score"])
    neutros.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "atualizado": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "estrategia": estrategia,
        "total": len(tickers),
        "compras": compras,
        "vendas": vendas,
        "neutros": neutros,
    })

# ─────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────

def gerar_sinais_serie(closes, volume, closes_w_full, estrategia):
    """
    Gera um sinal (COMPRA / VENDA / NEUTRO) para cada candle da série diária.
    Usa janela deslizante — mínimo de 60 candles de aquecimento.

    Para estratégias semanais (2 e 4), a série semanal é fatiada até
    a data correspondente ao candle diário atual, garantindo que o backtest
    não usa dados futuros.
    """
    sinais  = []
    # Preserva o índice original (datas) para cruzar com semanal
    closes_c = closes.copy()
    vol_s    = volume.copy() if volume is not None else None

    for i in range(len(closes_c)):
        if i < 59:
            sinais.append("NEUTRO")
            continue

        janela   = closes_c.iloc[:i+1]
        vol_jan  = vol_s.iloc[:i+1] if vol_s is not None else None

        if estrategia == 1:
            sig = sinal_est1(janela)

        elif estrategia == 2:
            # Fatia semanal até a data atual do candle diário
            if closes_w_full is not None and len(closes_w_full) >= 30:
                data_atual = closes_c.index[i]
                cw_ate = closes_w_full[closes_w_full.index <= data_atual]
                sig = sinal_est2(cw_ate) if len(cw_ate) >= 30 else {"dec": "NEUTRO"}
            else:
                sig = {"dec": "NEUTRO"}

        elif estrategia == 3:
            sig = sinal_est3(janela, vol_jan)

        elif estrategia == 4:
            # Fatia semanal até a data atual
            if closes_w_full is not None and len(closes_w_full) >= 16:
                data_atual = closes_c.index[i]
                cw_ate = closes_w_full[closes_w_full.index <= data_atual]
                sig = sinal_est4(cw_ate) if len(cw_ate) >= 16 else {"dec": "NEUTRO"}
            else:
                sig = {"dec": "NEUTRO"}

        elif estrategia == 5:
            sig = sinal_est5(janela)

        else:
            sig = sinal_est6(janela)

        sinais.append(sig["dec"])

    return sinais


def rodar_backtest(ticker, estrategia, modo, usar_alvo, alvo_gain, alvo_loss, alvo_dias, periodo):
    """
    Executa o backtest completo.

    modo        : 'comprado' ou 'vendido'
    usar_alvo   : bool — ativa take profit / stop loss por %
    alvo_gain   : % de ganho para fechar posição (ex: 20.0)
    alvo_loss   : % de perda para fechar posição (ex: 5.0) — None = sem stop
    periodo     : '1y','2y','5y','10y'
    """
    sym  = ticker.upper() + (".SA" if not ticker.upper().endswith(".SA") else "")

    dd   = yf.download(sym, period=periodo, interval="1d",  progress=False, auto_adjust=True)
    dw   = yf.download(sym, period=periodo, interval="1wk", progress=False, auto_adjust=True)

    if dd.empty or len(dd) < 80:
        return {"erro": "Dados insuficientes para o período selecionado."}

    closes   = dd["Close"].squeeze().dropna()
    # Alinha high/low/volume ao mesmo índice de closes para evitar desalinhamento
    highs    = dd["High"].squeeze().reindex(closes.index).fillna(closes)   if "High"   in dd.columns else closes.copy()
    lows     = dd["Low"].squeeze().reindex(closes.index).fillna(closes)    if "Low"    in dd.columns else closes.copy()
    volume   = dd["Volume"].squeeze().reindex(closes.index).fillna(0)      if "Volume" in dd.columns else None
    closes_w = dw["Close"].squeeze().dropna() if not dw.empty else None

    datas  = list(closes.index)
    precos = closes.values
    highs_ = highs.values
    lows_  = lows.values

    # Gera vetor de sinais
    sinais = gerar_sinais_serie(closes, volume, closes_w, estrategia)

    operacoes  = []
    em_posicao = False
    entrada_idx = None
    entrada_preco = None

    sinal_entrada = "COMPRA" if modo == "comprado" else "VENDA"
    sinais_saida  = ["VENDA", "NEUTRO"] if modo == "comprado" else ["COMPRA", "NEUTRO"]

    for i in range(1, len(sinais)):
        sig_hoje = sinais[i]
        sig_ant  = sinais[i-1]
        p_hoje   = float(precos[i])
        h_hoje   = float(highs_[i])
        l_hoje   = float(lows_[i])

        if not em_posicao:
            # Entrada: sinal muda para o sinal de entrada
            if sig_hoje == sinal_entrada and sig_ant != sinal_entrada:
                em_posicao    = True
                entrada_idx   = i
                # Usa o preço de abertura do próximo candle como entrada (simulação realista)
                # Como não temos open, usamos close do candle de sinal como proxy
                entrada_preco = p_hoje

        else:
            # Não verifica saída no mesmo candle de entrada
            if i == entrada_idx:
                continue

            saiu         = False
            motivo_saida = ""
            preco_saida  = p_hoje

            if usar_alvo:
                # ── Modo comprado: TP quando high atinge alvo, SL quando low cai ──
                if modo == "comprado":
                    var_high = (h_hoje - entrada_preco) / entrada_preco * 100
                    var_low  = (l_hoje - entrada_preco) / entrada_preco * 100

                    tp_ativo = alvo_gain is not None and var_high >= alvo_gain
                    sl_ativo = alvo_loss is not None and var_low  <= -alvo_loss

                    if tp_ativo and sl_ativo:
                        # Ambos atingidos no mesmo candle — assume SL primeiro (conservador)
                        preco_saida  = round(entrada_preco * (1 - alvo_loss / 100), 2)
                        motivo_saida = f"Stop loss -{alvo_loss:.1f}% (mesmo candle que TP)"
                        saiu = True
                    elif tp_ativo:
                        preco_saida  = round(entrada_preco * (1 + alvo_gain / 100), 2)
                        motivo_saida = f"Take profit +{alvo_gain:.1f}%"
                        saiu = True
                    elif sl_ativo:
                        preco_saida  = round(entrada_preco * (1 - alvo_loss / 100), 2)
                        motivo_saida = f"Stop loss -{alvo_loss:.1f}%"
                        saiu = True

                # ── Modo vendido: TP quando low cai, SL quando high sobe ──
                else:
                    var_low  = (entrada_preco - l_hoje) / entrada_preco * 100   # ganho no short
                    var_high = (entrada_preco - h_hoje) / entrada_preco * 100   # perda no short

                    tp_ativo = alvo_gain is not None and var_low  >= alvo_gain
                    sl_ativo = alvo_loss is not None and var_high <= -alvo_loss

                    if tp_ativo and sl_ativo:
                        preco_saida  = round(entrada_preco * (1 + alvo_loss / 100), 2)
                        motivo_saida = f"Stop loss -{alvo_loss:.1f}% (mesmo candle que TP)"
                        saiu = True
                    elif tp_ativo:
                        preco_saida  = round(entrada_preco * (1 - alvo_gain / 100), 2)
                        motivo_saida = f"Take profit +{alvo_gain:.1f}%"
                        saiu = True
                    elif sl_ativo:
                        preco_saida  = round(entrada_preco * (1 + alvo_loss / 100), 2)
                        motivo_saida = f"Stop loss -{alvo_loss:.1f}%"
                        saiu = True

                # Checagem de dias máximos (quando TP/SL não foram atingidos ainda)
                if not saiu and alvo_dias is not None:
                    dias_na_posicao = i - entrada_idx
                    if dias_na_posicao >= alvo_dias:
                        preco_saida  = p_hoje
                        motivo_saida = f"Máx. {alvo_dias}d atingido ({dias_na_posicao}d)"
                        saiu = True

                # Quando usar_alvo=True: NÃO sai por sinal — só por TP/SL/dias
                # (comportamento intencional: o usuário quer controle total)

            else:
                # Sem alvos: saída puramente por mudança de sinal
                if sig_hoje in sinais_saida and sig_ant not in sinais_saida:
                    preco_saida  = p_hoje
                    motivo_saida = f"Sinal {sig_hoje}"
                    saiu = True

            if saiu:
                if modo == "comprado":
                    resultado_pct = (preco_saida - entrada_preco) / entrada_preco * 100
                else:
                    resultado_pct = (entrada_preco - preco_saida) / entrada_preco * 100

                dias = i - entrada_idx
                operacoes.append({
                    "entrada_data":  datas[entrada_idx].strftime("%d/%m/%Y"),
                    "entrada_preco": round(entrada_preco, 2),
                    "saida_data":    datas[i].strftime("%d/%m/%Y"),
                    "saida_preco":   round(preco_saida, 2),
                    "resultado_pct": round(resultado_pct, 2),
                    "dias":          dias,
                    "motivo":        motivo_saida,
                    "ganho":         resultado_pct > 0,
                })
                em_posicao = False

    # Posição ainda aberta no final
    posicao_aberta = None
    if em_posicao:
        p_ult = float(precos[-1])
        if modo == "comprado":
            res_aberto = (p_ult - entrada_preco) / entrada_preco * 100
        else:
            res_aberto = (entrada_preco - p_ult) / entrada_preco * 100
        posicao_aberta = {
            "entrada_data":  datas[entrada_idx].strftime("%d/%m/%Y"),
            "entrada_preco": round(entrada_preco, 2),
            "resultado_pct": round(res_aberto, 2),
        }

    # ── Estatísticas ──
    total      = len(operacoes)
    ganhos     = [o for o in operacoes if o["ganho"]]
    perdas     = [o for o in operacoes if not o["ganho"]]
    pct_acerto = round(len(ganhos) / total * 100, 1) if total else 0

    retorno_total = sum(o["resultado_pct"] for o in operacoes)

    # Retorno composto (como se reinvestisse sempre 100%)
    fator = 1.0
    for o in operacoes:
        fator *= (1 + o["resultado_pct"] / 100)
    retorno_composto = round((fator - 1) * 100, 2)

    media_ganho  = round(sum(o["resultado_pct"] for o in ganhos) / len(ganhos), 2) if ganhos else 0
    media_perda  = round(sum(o["resultado_pct"] for o in perdas) / len(perdas), 2) if perdas else 0
    media_dias   = round(sum(o["dias"] for o in operacoes) / total, 1) if total else 0
    maior_ganho  = round(max((o["resultado_pct"] for o in operacoes), default=0), 2)
    maior_perda  = round(min((o["resultado_pct"] for o in operacoes), default=0), 2)

    fator_lucro  = round(
        abs(sum(o["resultado_pct"] for o in ganhos) /
            sum(o["resultado_pct"] for o in perdas))
        if perdas and ganhos else 0, 2
    )

    # Drawdown máximo (sequência de perdas acumuladas)
    pico = 1.0; fundo = 1.0; max_dd = 0.0; fator_dd = 1.0
    for o in operacoes:
        fator_dd *= (1 + o["resultado_pct"] / 100)
        if fator_dd > pico:
            pico = fator_dd
        dd_atual = (fator_dd - pico) / pico * 100
        if dd_atual < max_dd:
            max_dd = dd_atual

    return {
        "ticker":    ticker.upper(),
        "estrategia": estrategia,
        "modo":      modo,
        "periodo":   periodo,
        "usar_alvo": usar_alvo,
        "alvo_gain": alvo_gain,
        "alvo_loss": alvo_loss,
        "alvo_dias": alvo_dias,
        "operacoes": operacoes,
        "posicao_aberta": posicao_aberta,
        "stats": {
            "total":            total,
            "ganhos":           len(ganhos),
            "perdas":           len(perdas),
            "pct_acerto":       pct_acerto,
            "retorno_total":    round(retorno_total, 2),
            "retorno_composto": retorno_composto,
            "media_ganho":      media_ganho,
            "media_perda":      media_perda,
            "media_dias":       media_dias,
            "maior_ganho":      maior_ganho,
            "maior_perda":      maior_perda,
            "fator_lucro":      fator_lucro,
            "max_drawdown":     round(max_dd, 2),
        }
    }


@app.route("/backtest")
def rota_backtest():
    ticker     = request.args.get("ticker", "PETR4").upper().strip()
    estrategia = int(request.args.get("estrategia", 1))
    modo       = request.args.get("modo", "comprado")           # comprado | vendido
    usar_alvo  = request.args.get("usar_alvo", "false") == "true"
    alvo_gain  = request.args.get("alvo_gain", None)
    alvo_loss  = request.args.get("alvo_loss", None)
    periodo    = request.args.get("periodo", "5y")

    alvo_gain = float(alvo_gain) if alvo_gain and alvo_gain.strip() else None
    alvo_loss = float(alvo_loss) if alvo_loss and alvo_loss.strip() else None
    alvo_dias_raw = request.args.get("alvo_dias", None)
    alvo_dias = int(alvo_dias_raw) if alvo_dias_raw and alvo_dias_raw.strip() else None

    try:
        res = rodar_backtest(ticker, estrategia, modo, usar_alvo, alvo_gain, alvo_loss, alvo_dias, periodo)
        return jsonify(res)
    except Exception as e:
        return jsonify({"erro": str(e)})



@app.route("/configurar_telegram")
def configurar_telegram():
    token   = request.args.get("token", "")
    chat_id = request.args.get("chat_id", "")
    if not token or not chat_id:
        return jsonify({"erro": "Informe token e chat_id"})
    cfg = cfg_load()
    cfg["tg_token"]  = token
    cfg["tg_chat_id"] = chat_id
    cfg_save(cfg)
    ok = tg_send(token, chat_id, "🤖 <b>Robô B3 conectado!</b>\nTelegram configurado com sucesso!")
    return jsonify({"ok": ok, "salvo": True})

@app.route("/scan_manual")
def scan_manual():
    cfg = cfg_load()
    alertas = monitor_scan(cfg)
    return jsonify({"ok": True, "alertas": len(alertas), "ultimo_scan": cfg["ultimo_scan"]})

@app.route("/tg_debug")
def tg_debug():
    token   = os.environ.get("TG_TOKEN", "NAO_ENCONTRADO")
    chat_id = os.environ.get("TG_CHAT_ID", "NAO_ENCONTRADO")
    cfg = cfg_load()
    ok = tg_send(token, chat_id, "🤖 Teste debug Robô B3")
    return jsonify({
        "env_token":  token[:8] + "..." if len(token) > 8 else token,
        "env_chatid": chat_id,
        "cfg_token":  cfg.get("tg_token","")[:8] + "..." if cfg.get("tg_token") else "",
        "cfg_chatid": cfg.get("tg_chat_id",""),
        "tg_enviado": ok
    })

if __name__ == "__main__":
    print("\n🤖 Robô B3 — Análise Técnica")
    print("👉  Acesse: http://localhost:5000\n")
    app.run(debug=True, port=5000)


# ═════════════════════════════════════════════════════════════════
#  MONITOR + TELEGRAM + SCHEDULER
#  Toda config fica em memória (_STATE) — sem depender de /tmp
#  Todas as rotas aceitam GET — sem problemas de CORS/POST
# ═════════════════════════════════════════════════════════════════

import threading
import time
import json
import os
import urllib.request
from datetime import datetime

# ── Estado em memória (persiste enquanto o servidor estiver rodando) ──
_STATE = {
    "ativo":             True,
    "estrategia":        1,
    "tickers":           ["PETR4", "VALE3", "ITUB4", "WEGE3", "PRIO3"],
    "intervalo":         30,
    "hora_inicio":       "10:00",
    "hora_fim":          "18:00",
    "tg_token":          "",
    "tg_chat_id":        "",
    "alertar_compra":    True,
    "alertar_venda":     True,
    "alertar_neutro":    False,
    "ultimo_scan":       "",
    "sinais_anteriores": {},
}

def state_get():
    return _STATE

def state_set(**kwargs):
    for k, v in kwargs.items():
        if k in _STATE:
            _STATE[k] = v

# ── Telegram ──────────────────────────────────────────────────────

def tg_send(token, chat_id, texto):
    if not token or not chat_id:
        return False
    try:
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({
            "chat_id":    str(chat_id),
            "text":       texto,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[Telegram] Erro: {e}")
        return False

def tg_alerta(ticker, sinal, preco, var_pct, score, tendencia, estrategia):
    est_names = {
        1: "RSI5d+MACD+BB", 2: "RSI14sem+MACD+BB",
        3: "Médias+MACD+Vol", 4: "RSI14 Semanal",
        5: "RSI2 Diário",    6: "RSI5 Diário"
    }
    emoji   = "🟢" if sinal == "COMPRA" else "🔴" if sinal == "VENDA" else "⚪"
    var_str = f"+{var_pct:.2f}%" if var_pct >= 0 else f"{var_pct:.2f}%"
    sc_str  = f"+{score}" if score > 0 else str(score)
    msg = (
        f"{emoji} <b>{ticker} — {sinal}</b>\n"
        f"💰 Preço: <b>R$ {preco:.2f}</b> ({var_str})\n"
        f"📊 Score: {sc_str} | Tendência: {tendencia}\n"
        f"🔧 Estratégia: {est_names.get(estrategia, str(estrategia))}\n"
        f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    s = state_get()
    return tg_send(s["tg_token"], s["tg_chat_id"], msg)

# ── Scanner do monitor ────────────────────────────────────────────

def monitor_scan():
    s          = state_get()
    tickers    = s.get("tickers", [])
    estrategia = int(s.get("estrategia", 1))
    anteriores = s.get("sinais_anteriores", {})
    novos      = dict(anteriores)
    alertas    = []

    for ticker in tickers:
        try:
            res = analisar_ticker(ticker)
            if not res:
                continue
            key_map = {1:"e1",2:"e2",3:"e3",4:"e4",5:"e5",6:"e6"}
            sig_obj = res.get(key_map.get(estrategia,"e1"), {})
            sinal   = sig_obj.get("dec", "NEUTRO")
            score   = sig_obj.get("score", 0)
            preco   = res.get("preco", 0)
            var_pct = res.get("var_pct", 0)
            tend    = res.get("tend", {}).get("label", "—")

            sinal_ant = anteriores.get(ticker, "")
            if sinal != sinal_ant:
                deve = (
                    (sinal == "COMPRA" and s.get("alertar_compra", True)) or
                    (sinal == "VENDA"  and s.get("alertar_venda",  True)) or
                    (sinal == "NEUTRO" and s.get("alertar_neutro", False))
                )
                if deve:
                    ok = tg_alerta(ticker, sinal, preco, var_pct, score, tend, estrategia)
                    alertas.append({
                        "ticker": ticker, "sinal": sinal, "preco": preco,
                        "score": score, "tendencia": tend,
                        "ok": ok, "mudou_de": sinal_ant or "—"
                    })
                novos[ticker] = sinal
        except Exception as e:
            print(f"[monitor_scan] {ticker}: {e}")

    _STATE["sinais_anteriores"] = novos
    _STATE["ultimo_scan"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return alertas

# ── Scheduler ────────────────────────────────────────────────────

_sched_running = False

def _sched_loop():
    global _sched_running
    print("[Scheduler] Iniciado.")
    while _sched_running:
        s = state_get()
        if not s.get("ativo", False):
            time.sleep(60)
            continue
        agora = datetime.now()
        try:
            h_ini = datetime.strptime(s["hora_inicio"], "%H:%M").replace(
                year=agora.year, month=agora.month, day=agora.day)
            h_fim = datetime.strptime(s["hora_fim"], "%H:%M").replace(
                year=agora.year, month=agora.month, day=agora.day)
        except Exception:
            time.sleep(60)
            continue
        if agora.weekday() < 5 and h_ini <= agora <= h_fim:
            print(f"[Scheduler] Varrendo {len(s['tickers'])} ativos...")
            try:
                alertas = monitor_scan()
                if alertas:
                    print(f"[Scheduler] {len(alertas)} alerta(s) enviado(s).")
            except Exception as e:
                print(f"[Scheduler] Erro: {e}")
        time.sleep(int(s.get("intervalo", 30)) * 60)
    print("[Scheduler] Parado.")

def scheduler_start():
    global _sched_running
    if _sched_running:
        return
    _sched_running = True
    t = threading.Thread(target=_sched_loop, daemon=True)
    t.start()

scheduler_start()

# ── Rotas do monitor (todas GET — sem problemas de CORS/POST) ─────

@app.route("/monitor/status")
def monitor_status():
    s = state_get()
    return jsonify({
        "ativo":          s["ativo"],
        "estrategia":     s["estrategia"],
        "intervalo":      s["intervalo"],
        "hora_inicio":    s["hora_inicio"],
        "hora_fim":       s["hora_fim"],
        "tg_configurado": bool(s["tg_token"] and s["tg_chat_id"]),
        "tickers":        s["tickers"],
        "total_ativos":   len(s["tickers"]),
        "ultimo_scan":    s["ultimo_scan"],
        "sinais":         s["sinais_anteriores"],
        "alertar_compra": s["alertar_compra"],
        "alertar_venda":  s["alertar_venda"],
        "alertar_neutro": s["alertar_neutro"],
        "scheduler_on":   _sched_running,
    })

@app.route("/monitor/salvar")
def monitor_salvar():
    """Salva todas as configs via GET params."""
    s = state_get()

    # Token e Chat ID
    token   = request.args.get("tg_token",   "").strip()
    chat_id = request.args.get("tg_chat_id", "").strip()
    if token:   s["tg_token"]   = token
    if chat_id: s["tg_chat_id"] = chat_id

    # Configurações gerais
    if request.args.get("estrategia"):
        s["estrategia"] = int(request.args.get("estrategia"))
    if request.args.get("intervalo"):
        s["intervalo"] = int(request.args.get("intervalo"))
    if request.args.get("hora_inicio"):
        s["hora_inicio"] = request.args.get("hora_inicio")
    if request.args.get("hora_fim"):
        s["hora_fim"] = request.args.get("hora_fim")
    if request.args.get("ativo") is not None:
        s["ativo"] = request.args.get("ativo") == "true"
    if request.args.get("alertar_compra") is not None:
        s["alertar_compra"] = request.args.get("alertar_compra") == "true"
    if request.args.get("alertar_venda") is not None:
        s["alertar_venda"] = request.args.get("alertar_venda") == "true"
    if request.args.get("alertar_neutro") is not None:
        s["alertar_neutro"] = request.args.get("alertar_neutro") == "true"

    # Tickers
    tickers_raw = request.args.get("tickers", "")
    if tickers_raw:
        tickers = [t.strip().upper() for t in tickers_raw.replace("\n",",").split(",") if t.strip()]
        if tickers:
            s["tickers"] = tickers

    return jsonify({"ok": True, "estado": {
        "ativo": s["ativo"],
        "estrategia": s["estrategia"],
        "tickers": s["tickers"],
        "tg_configurado": bool(s["tg_token"] and s["tg_chat_id"]),
    }})

@app.route("/monitor/testar_telegram")
def monitor_testar_telegram():
    s = state_get()
    token   = request.args.get("tg_token",   s["tg_token"]).strip()
    chat_id = request.args.get("tg_chat_id", s["tg_chat_id"]).strip()
    # Salva na memória ao mesmo tempo
    if token:   s["tg_token"]   = token
    if chat_id: s["tg_chat_id"] = chat_id
    ok = tg_send(token, chat_id,
        "🤖 <b>Robô B3 conectado!</b>\n"
        "Você receberá alertas de compra e venda aqui.\n"
        f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    )
    return jsonify({
        "ok":  ok,
        "msg": "✅ Mensagem enviada! Verifique o Telegram." if ok else "❌ Falhou. Verifique o token e o Chat ID."
    })

@app.route("/monitor/scan_agora")
def monitor_scan_agora():
    alertas = monitor_scan()
    s = state_get()
    return jsonify({"ok": True, "alertas": alertas, "total": len(alertas), "ultimo_scan": s["ultimo_scan"]})

@app.route("/monitor/toggle_ativo")
def monitor_toggle_ativo():
    s = state_get()
    s["ativo"] = not s["ativo"]
    return jsonify({"ok": True, "ativo": s["ativo"]})
