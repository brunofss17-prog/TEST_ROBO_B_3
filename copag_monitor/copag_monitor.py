#!/usr/bin/env python3
"""
Robô monitor — Copag Loja × Pokémon TCG: Celebração de 30 Anos
──────────────────────────────────────────────────────────────
Fica rodando em loop até detectar que a Copag Loja (copagloja.com.br)
liberou a venda de produtos da coleção Pokémon TCG 30 anos.

Como funciona
  A Copag Loja roda na plataforma VTEX, que expõe uma API pública de
  catálogo:  /api/catalog_system/pub/products/search?ft=<termo>
  A cada ciclo o robô pesquisa vários termos, filtra os produtos da
  coleção 30 anos e verifica, SKU a SKU, se há estoque à venda
  (commertialOffer.AvailableQuantity > 0 / IsAvailable).

  Dispara alerta quando:
    • um produto da coleção fica DISPONÍVEL PARA COMPRA (alerta principal);
    • um produto novo da coleção aparece no site, mesmo indisponível
      (aviso de "pré-lançamento" — sinal de que a venda está próxima).

  Alertas: terminal + bipe, Telegram e/ou Discord (opcionais, por variáveis
  de ambiente). O estado fica salvo em JSON para não repetir alertas.

Uso
  python copag_monitor.py                 # roda até encontrar produto à venda
  python copag_monitor.py --intervalo 30  # checa a cada ~30 s
  python copag_monitor.py --continuar     # não para após o 1º alerta
  python copag_monitor.py --uma-vez       # faz só uma verificação
  python copag_monitor.py --testar-alerta # testa Telegram/Discord

Variáveis de ambiente (opcionais)
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID   → alertas no Telegram
  DISCORD_WEBHOOK_URL                → alertas no Discord
  COPAG_BASE_URL                     → padrão https://www.copagloja.com.br

Só usa a biblioteca padrão do Python (3.8+).
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timedelta, timezone

BASE_URL = os.environ.get("COPAG_BASE_URL", "https://www.copagloja.com.br").rstrip("/")

# Termos pesquisados na busca da loja (ft = full text). A busca da VTEX é
# fraca com frases, então buscamos o catálogo Pokémon inteiro e filtramos
# localmente; os demais termos são uma rede de segurança.
TERMOS_BUSCA = [
    "pokemon",
    "pokémon",
    "pikachu",
    "celebração",
    "30 anos",
]

# Um produto é da coleção se o texto (nome/descrição/categorias) bater com
# algum destes padrões (texto já normalizado: minúsculo e sem acentos)...
PADROES_COLECAO = [
    r"\b30\s*anos\b",
    r"\bcelebracao\b",
    r"\b30th\b",
    r"\bcelebration\b",
    r"\bsv\s*-?\s*30\b",
]
# ...e também mencionar Pokémon.
PADRAO_POKEMON = r"pokemon"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

TZ_BR = timezone(timedelta(hours=-3))


# ─────────────────────────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────────────────────────

def agora():
    return datetime.now(TZ_BR).strftime("%d/%m/%Y %H:%M:%S")


def log(msg):
    print(f"[{agora()}] {msg}", flush=True)


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.lower()


def http_get_json(url, timeout=20):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Cache-Control": "no-cache",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # VTEX responde 206 (Partial Content) em buscas paginadas.
        if resp.status not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


# ─────────────────────────────────────────────────────────────────
#  Catálogo VTEX
# ─────────────────────────────────────────────────────────────────

def buscar_produtos(termo, por_pagina=50, max_paginas=5):
    """Busca produtos na API pública de catálogo VTEX (paginando)."""
    produtos = []
    for pagina in range(max_paginas):
        # quote_via=quote: a VTEX rejeita (HTTP 400) espaço codificado como "+".
        qs = urllib.parse.urlencode({
            "ft": termo,
            "_from": pagina * por_pagina,
            "_to": (pagina + 1) * por_pagina - 1,
            "_": int(time.time()),  # evita cache de CDN
        }, quote_via=urllib.parse.quote)
        url = f"{BASE_URL}/api/catalog_system/pub/products/search?{qs}"
        dados = http_get_json(url)
        if not isinstance(dados, list) or not dados:
            break
        produtos.extend(dados)
        if len(dados) < por_pagina:
            break
    return produtos


def eh_da_colecao(produto):
    partes = [
        produto.get("productName", ""),
        produto.get("productTitle", ""),
        produto.get("description", ""),
        produto.get("brand", ""),
        " ".join(produto.get("categories", []) or []),
        produto.get("linkText", "").replace("-", " "),
    ]
    texto = normalizar(" ".join(p for p in partes if isinstance(p, str)))
    if not re.search(PADRAO_POKEMON, texto):
        return False
    return any(re.search(p, texto) for p in PADROES_COLECAO)


def resumir_produto(produto):
    """Extrai o essencial: disponibilidade e menor preço entre os SKUs."""
    disponivel = False
    qtd_total = 0
    precos = []
    for item in produto.get("items", []) or []:
        for seller in item.get("sellers", []) or []:
            oferta = seller.get("commertialOffer", {}) or {}
            qtd = oferta.get("AvailableQuantity", 0) or 0
            preco = oferta.get("Price", 0) or 0
            if qtd > 0 or oferta.get("IsAvailable") is True:
                disponivel = True
                qtd_total += qtd
                if preco:
                    precos.append(preco)

    link = produto.get("link") or ""
    if not link and produto.get("linkText"):
        link = f"{BASE_URL}/{produto['linkText']}/p"

    return {
        "id": str(produto.get("productId", "")),
        "nome": produto.get("productName", "(sem nome)"),
        "link": link,
        "disponivel": disponivel,
        "quantidade": qtd_total,
        "preco": min(precos) if precos else None,
    }


def verificar():
    """Retorna {id: resumo} dos produtos da coleção encontrados agora."""
    encontrados = {}
    total_pokemon = set()
    erros = 0
    for termo in TERMOS_BUSCA:
        try:
            for p in buscar_produtos(termo):
                if re.search(PADRAO_POKEMON, normalizar(p.get("productName", ""))):
                    total_pokemon.add(p.get("productId"))
                if eh_da_colecao(p):
                    r = resumir_produto(p)
                    if r["id"]:
                        encontrados[r["id"]] = r
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError,
                ValueError, TimeoutError) as e:
            erros += 1
            log(f"⚠  Falha ao buscar '{termo}': {e}")
        time.sleep(random.uniform(0.5, 1.5))  # gentil com o servidor
    if erros == len(TERMOS_BUSCA):
        raise RuntimeError("todas as buscas falharam")
    if not total_pokemon:
        log("⚠  Nenhum produto Pokémon retornado — a busca da loja pode ter mudado.")
    return encontrados, len(total_pokemon)


# ─────────────────────────────────────────────────────────────────
#  Alertas
# ─────────────────────────────────────────────────────────────────

def _post_json(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return 200 <= resp.status < 300


def enviar_telegram(texto):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        return _post_json(f"https://api.telegram.org/bot{token}/sendMessage", {
            "chat_id": chat_id, "text": texto, "disable_web_page_preview": False,
        })
    except Exception as e:
        log(f"[Telegram] Erro: {e}")
        return False


def enviar_discord(texto):
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return False
    try:
        return _post_json(webhook, {"content": texto[:1900]})
    except Exception as e:
        log(f"[Discord] Erro: {e}")
        return False


def tocar_som():
    if sys.platform == "win32":
        import winsound  # o "\a" não toca no terminal do VS Code
        for _ in range(5):
            winsound.Beep(1200, 400)
            time.sleep(0.15)
    else:
        for _ in range(5):
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.3)


def popup(titulo, texto):
    """Janela de aviso na tela (não bloqueia o robô)."""
    def _mostrar():
        try:
            if sys.platform == "win32":
                import ctypes
                # 0x40 = ícone de informação, 0x1000 = sempre por cima
                ctypes.windll.user32.MessageBoxW(0, texto, titulo, 0x40 | 0x1000)
            elif sys.platform == "darwin":
                subprocess.run(["osascript", "-e",
                                f'display notification {json.dumps(texto[:200])} '
                                f'with title {json.dumps(titulo)}'], check=False)
            else:
                subprocess.run(["notify-send", titulo, texto[:300]], check=False)
        except Exception:
            pass
    threading.Thread(target=_mostrar, daemon=True).start()


def alertar(titulo, produtos, bipe=True, abrir_navegador=False):
    linhas = [titulo, ""]
    for p in produtos:
        preco = f"R$ {p['preco']:.2f}".replace(".", ",") if p.get("preco") else "—"
        status = "✅ À VENDA" if p["disponivel"] else "⏳ indisponível"
        linhas.append(f"• {p['nome']}\n  {status} | {preco}\n  {p['link']}")
    texto = "\n".join(linhas)

    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70 + "\n", flush=True)
    popup(titulo, texto)
    if abrir_navegador:
        for p in produtos[:3]:
            if p.get("link"):
                webbrowser.open(p["link"])
    if bipe:
        tocar_som()
    enviar_telegram(texto)
    enviar_discord(texto)


# ─────────────────────────────────────────────────────────────────
#  Estado
# ─────────────────────────────────────────────────────────────────

def carregar_estado(caminho):
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {"vistos": {}, "a_venda": {}}


def salvar_estado(caminho, estado):
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)
    os.replace(tmp, caminho)


def processar(estado, encontrados):
    """Compara com o estado anterior. Retorna (novos, liberados)."""
    novos, liberados = [], []
    for pid, p in encontrados.items():
        if pid not in estado["vistos"]:
            novos.append(p)
        if p["disponivel"] and pid not in estado["a_venda"]:
            liberados.append(p)
        estado["vistos"][pid] = p
        if p["disponivel"]:
            estado["a_venda"][pid] = p
        else:
            # Esgotou de novo: permite alertar outra vez se voltar.
            estado["a_venda"].pop(pid, None)
    return novos, liberados


# ─────────────────────────────────────────────────────────────────
#  Loop principal
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intervalo", type=int, default=60,
                    help="segundos entre verificações (padrão 60, mínimo 15)")
    ap.add_argument("--continuar", action="store_true",
                    help="não encerra após detectar produto à venda")
    ap.add_argument("--uma-vez", action="store_true", help="faz uma verificação e sai")
    ap.add_argument("--estado", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "estado_copag.json"),
                    help="arquivo JSON de estado")
    ap.add_argument("--testar-alerta", action="store_true",
                    help="envia um alerta de teste e sai")
    args = ap.parse_args()
    intervalo = max(15, args.intervalo)

    if args.testar_alerta:
        alertar("🔔 Teste do robô Copag × Pokémon 30 anos", [{
            "nome": "Produto de teste", "link": BASE_URL, "disponivel": True,
            "preco": 99.9, "quantidade": 1,
        }])
        return

    estado = carregar_estado(args.estado)
    log(f"Robô iniciado — monitorando {BASE_URL} a cada ~{intervalo}s.")
    log(f"Telegram: {'ativo' if os.environ.get('TELEGRAM_TOKEN') else 'desligado'} | "
        f"Discord: {'ativo' if os.environ.get('DISCORD_WEBHOOK_URL') else 'desligado'}")

    falhas_seguidas = 0
    while True:
        try:
            encontrados, n_pokemon = verificar()
            falhas_seguidas = 0
            novos, liberados = processar(estado, encontrados)
            salvar_estado(args.estado, estado)

            a_venda = [p for p in encontrados.values() if p["disponivel"]]
            log(f"{n_pokemon} produto(s) Pokémon no catálogo | "
                f"{len(encontrados)} da coleção 30 anos | {len(a_venda)} à venda.")

            novos_indisp = [p for p in novos if not p["disponivel"]]
            if novos_indisp:
                alertar("👀 Produtos da coleção Pokémon 30 anos apareceram na Copag Loja "
                        "(ainda sem estoque) — a venda deve estar próxima!",
                        novos_indisp, bipe=False)

            if liberados:
                alertar("🚨 VENDA LIBERADA! Pokémon TCG 30 anos disponível na Copag Loja!",
                        liberados, abrir_navegador=True)
                if not args.continuar:
                    log("Objetivo atingido. Encerrando (use --continuar para seguir).")
                    return
        except Exception as e:
            falhas_seguidas += 1
            log(f"❌ Erro na verificação ({falhas_seguidas}x seguidas): {e}")

        if args.uma_vez:
            return

        # Backoff em caso de falhas (bloqueio/instabilidade), máx. 10 min.
        espera = intervalo * (2 ** min(falhas_seguidas, 4)) if falhas_seguidas else intervalo
        espera = min(espera, 600) * random.uniform(0.85, 1.15)
        time.sleep(espera)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrompido pelo usuário.")
