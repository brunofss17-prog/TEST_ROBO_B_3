# Robô de Enquetes do WhatsApp

Quando chega uma enquete em qualquer conversa, o robô vota **na primeira opção** na hora.

## Como usar

1. Instale o [Node.js](https://nodejs.org) 18 ou superior.
2. Nesta pasta, rode:
   ```bash
   npm install
   npm start
   ```
3. Vai aparecer um QR code no terminal. No celular, abra o WhatsApp →
   **Aparelhos conectados** → **Conectar um aparelho** e escaneie.
4. Quando aparecer `✅ Robô conectado`, ele já está votando.

A sessão fica salva em `.wwebjs_auth/`, então das próximas vezes não precisa escanear de novo.
O computador precisa ficar ligado com o robô rodando.

## Opções

- **Só em alguns grupos:** passe os nomes dos chats separados por vírgula:
  ```bash
  CHATS="Futebol de Quinta,Grupo da Família" npm start
  ```
- **Usar o Chrome já instalado** (se o download do Chromium falhar):
  ```bash
  CHROME_PATH="/caminho/para/chrome" npm start
  ```

Enquetes que você mesmo criou são ignoradas.

## Aviso

Ele usa o WhatsApp Web via [whatsapp-web.js](https://github.com/pedroslopez/whatsapp-web.js), que não é uma API oficial.
Automatizar a conta vai contra os termos do WhatsApp e pode levar a bloqueio. Se possível, use um número secundário.
