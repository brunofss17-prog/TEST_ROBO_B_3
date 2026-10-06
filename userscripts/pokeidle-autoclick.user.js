// ==UserScript==
// @name         Pokeidle - Auto clique
// @namespace    test-robo-b-3
// @version      1.1
// @description  Clica automaticamente num item da tela (ex.: lista "Proximidade para capturar").
// @match        *://pokeidle.io/*
// @match        *://*.pokeidle.io/*
// @noframes
// @grant        none
// @run-at       document-idle
// ==/UserScript==

(function () {
  'use strict';
  console.log('[auto clique] script carregado');

  const CFG_KEY = 'pkidle-autoclick';
  const cfg = Object.assign(
    { intervalMs: 1500, selector: null },
    JSON.parse(localStorage.getItem(CFG_KEY) || '{}')
  );
  const save = () => localStorage.setItem(CFG_KEY, JSON.stringify(cfg));

  let timer = null;
  let picking = false;

  // Gera um seletor CSS por posição (nth-of-type) até o <body>.
  function cssPath(el) {
    const parts = [];
    while (el && el.nodeType === 1 && el !== document.body) {
      let part = el.tagName.toLowerCase();
      if (el.id) { parts.unshift('#' + CSS.escape(el.id)); break; }
      const sibs = Array.from(el.parentNode.children).filter(s => s.tagName === el.tagName);
      if (sibs.length > 1) part += `:nth-of-type(${sibs.indexOf(el) + 1})`;
      parts.unshift(part);
      el = el.parentNode;
    }
    return parts.join(' > ');
  }

  // Fallback: primeira linha da lista "Proximidade para capturar".
  function findProximityItem() {
    const title = Array.from(document.querySelectorAll('body *')).find(
      e => e.children.length === 0 && /proximidade\s+para\s+capturar/i.test(e.textContent)
    );
    if (!title) return null;
    let box = title.parentElement;
    for (let i = 0; i < 4 && box; i++, box = box.parentElement) {
      const row = Array.from(box.querySelectorAll('*')).find(
        e => e !== title && !e.contains(title) && /Nv\s*\d+/i.test(e.textContent) &&
             e.children.length > 0 && e.getBoundingClientRect().height > 20
      );
      if (row) return row;
    }
    return null;
  }

  function realClick(el) {
    const r = el.getBoundingClientRect();
    const opts = { bubbles: true, cancelable: true, view: window,
                   clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 };
    for (const type of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click']) {
      const Ev = type.startsWith('pointer') ? PointerEvent : MouseEvent;
      el.dispatchEvent(new Ev(type, opts));
    }
  }

  function tick() {
    const el = (cfg.selector && document.querySelector(cfg.selector)) || findProximityItem();
    if (el) {
      realClick(el);
      flash(el);
      status('clicou ' + new Date().toLocaleTimeString());
    } else {
      status('alvo não encontrado');
    }
  }

  function flash(el) {
    const old = el.style.outline;
    el.style.outline = '2px solid #0f0';
    setTimeout(() => (el.style.outline = old), 200);
  }

  // ---------- painel ----------
  const ui = document.createElement('div');
  ui.style.cssText = 'position:fixed;bottom:8px;right:8px;z-index:2147483647;background:#222;color:#fff;' +
    'font:12px sans-serif;padding:8px;border-radius:6px;box-shadow:0 2px 8px #000a;display:flex;' +
    'flex-direction:column;gap:4px;min-width:180px';
  ui.innerHTML = `
    <b>Auto clique</b>
    <button data-a="toggle">▶ Iniciar</button>
    <button data-a="pick">🎯 Escolher alvo</button>
    <button data-a="reset">↺ Alvo padrão (Proximidade)</button>
    <label>Intervalo (ms) <input data-a="ms" type="number" min="200" step="100" style="width:70px"></label>
    <small data-a="status" style="opacity:.8"></small>`;
  // O jogo pode recriar o <body>/<html>; re-anexa o painel se ele sumir.
  const attach = () => {
    const host = document.body || document.documentElement;
    if (host && !host.contains(ui)) host.appendChild(ui);
  };
  attach();
  setInterval(attach, 1000);

  const $ = a => ui.querySelector(`[data-a="${a}"]`);
  const status = t => ($('status').textContent = t);
  $('ms').value = cfg.intervalMs;
  status(cfg.selector ? 'alvo personalizado salvo' : 'alvo: 1º da Proximidade');

  function start() {
    stop();
    timer = setInterval(tick, cfg.intervalMs);
    $('toggle').textContent = '⏸ Parar';
  }
  function stop() {
    clearInterval(timer);
    timer = null;
    $('toggle').textContent = '▶ Iniciar';
  }

  $('toggle').onclick = () => (timer ? stop() : start());
  $('ms').onchange = e => {
    cfg.intervalMs = Math.max(200, +e.target.value || 1500);
    save();
    if (timer) start();
  };
  $('reset').onclick = () => { cfg.selector = null; save(); status('alvo: 1º da Proximidade'); };
  $('pick').onclick = () => { picking = true; status('clique no item desejado...'); };

  // Modo de seleção: o próximo clique (fora do painel) define o alvo.
  document.addEventListener('click', e => {
    if (!picking || ui.contains(e.target)) return;
    e.preventDefault();
    e.stopPropagation();
    picking = false;
    cfg.selector = cssPath(e.target);
    save();
    flash(e.target);
    status('alvo salvo');
  }, true);
})();
