"""Pokeidle - auto clique automático (Python + Playwright).

Abre o jogo num Chrome controlado pelo script, acha sozinho a lista
"Proximidade para capturar" e clica no primeiro item, sem posição fixa
de tela. O login fica salvo na pasta ./perfil-pokeidle.

Instalação (uma vez):
    pip install playwright
    python -m playwright install chromium

Uso:
    python pokeidle_bot.py                 # clica no 1º item da lista
    python pokeidle_bot.py --nome Geodude  # só clica em itens com esse nome
    python pokeidle_bot.py --intervalo 2   # segundos entre cliques
"""

import argparse
import random
import time

from playwright.sync_api import sync_playwright

URL = "https://pokeidle.io/app"

# Procura o título "Proximidade para capturar" e devolve o retângulo
# (na tela) do primeiro item da lista que bate com o filtro de nome.
FIND_ITEM_JS = r"""
(nome) => {
  const all = Array.from(document.querySelectorAll('body *'));
  const title = all.find(e => e.children.length === 0 &&
                         /proximidade\s+para\s+capturar/i.test(e.textContent));
  if (!title) return { erro: 'lista "Proximidade para capturar" não encontrada' };
  const filtro = nome ? new RegExp(nome, 'i') : /Nv\s*\d+/i;
  let box = title.parentElement;
  for (let i = 0; i < 5 && box; i++, box = box.parentElement) {
    const rows = Array.from(box.querySelectorAll('*')).filter(e =>
      e !== title && !e.contains(title) && e.children.length > 0 &&
      filtro.test(e.textContent) && e.getBoundingClientRect().height > 15);
    // Fica com os elementos mais internos (a linha, não o container da lista).
    const leaf = rows.filter(r => !rows.some(o => o !== r && r.contains(o)));
    const row = leaf.find(r => {
      const b = r.getBoundingClientRect();
      return b.width > 0 && b.bottom > 0 && b.top < innerHeight;
    });
    if (row) {
      const b = row.getBoundingClientRect();
      return { x: b.left + b.width / 2, y: b.top + b.height / 2,
               texto: row.textContent.trim().slice(0, 40) };
    }
  }
  return { erro: 'nenhum item na lista' };
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", default="", help="só clica em itens cujo texto contém isso")
    ap.add_argument("--intervalo", type=float, default=1.5, help="segundos entre cliques")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            "perfil-pokeidle", headless=False, no_viewport=True,
            args=["--start-maximized"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(args.url)
        input("Faça login / entre na hunt no navegador que abriu e aperte ENTER aqui...")
        print("Rodando. Ctrl+C para parar.")

        ultimo_erro = None
        try:
            while True:
                r = page.evaluate(FIND_ITEM_JS, args.nome)
                if "erro" in r:
                    if r["erro"] != ultimo_erro:
                        print("aguardando:", r["erro"])
                    ultimo_erro = r["erro"]
                else:
                    ultimo_erro = None
                    # Clique "real" (evento confiável), igual a um clique de mouse.
                    page.mouse.click(r["x"], r["y"])
                    print(time.strftime("%H:%M:%S"), "clicou em", r["texto"])
                # Pequena variação no tempo para não ficar mecânico demais.
                time.sleep(args.intervalo * random.uniform(0.85, 1.15))
        except KeyboardInterrupt:
            print("Parado.")
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
