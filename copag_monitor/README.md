# Robô monitor — Copag Loja × Pokémon TCG 30 anos

Fica rodando até a [Copag Loja](https://www.copagloja.com.br) liberar a venda de
produtos da coleção **Pokémon TCG: Celebração de 30 Anos**, e então avisa você.

## Como funciona
A Copag Loja usa a plataforma VTEX. O robô consulta a API pública de catálogo
(`/api/catalog_system/pub/products/search?ft=...`) com vários termos
("30 anos", "celebração", "30th celebration"...), filtra produtos Pokémon da
coleção e checa o estoque de cada SKU.

- 🚨 **Venda liberada**: algum produto da coleção ficou com estoque → alerta e encerra
  (ou continua, com `--continuar`).
- 👀 **Pré-lançamento**: produto da coleção apareceu no site, ainda sem estoque.

## Uso
Requer só Python 3.8+ (sem dependências).

```bash
python copag_monitor.py                  # checa a cada ~60s até liberar
python copag_monitor.py --intervalo 30   # mais rápido (mínimo 15s)
python copag_monitor.py --continuar      # não para no primeiro alerta
python copag_monitor.py --uma-vez        # uma verificação só
```

## Alertas no celular (opcional)
```bash
# Telegram: crie um bot com o @BotFather e pegue seu chat_id
export TELEGRAM_TOKEN="123:ABC..."
export TELEGRAM_CHAT_ID="123456789"
# Discord: webhook de um canal
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

python copag_monitor.py --testar-alerta  # confere se está chegando
python copag_monitor.py
```
No Windows (PowerShell) use `$env:TELEGRAM_TOKEN="..."`.

O estado fica em `estado_copag.json` (para não repetir alertas). Apague-o para
recomeçar do zero.
