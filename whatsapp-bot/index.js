// Robô de enquetes do WhatsApp: assim que uma enquete chega, vota na primeira opção.
const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

// Grupos/contatos onde o robô vota (nomes separados por vírgula).
// Pode trocar via variável de ambiente: CHATS="Outro Grupo,Mais Um" npm start
// Use CHATS="*" para votar em qualquer conversa.
const CHATS_PADRAO = 'Loja Pokebola - Grupo VIP';
const CHATS_PERMITIDOS = (process.env.CHATS || CHATS_PADRAO)
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter((s) => s && s !== '*');

const client = new Client({
    // Guarda a sessão em .wwebjs_auth para não precisar ler o QR toda vez
    authStrategy: new LocalAuth(),
    puppeteer: {
        headless: true,
        executablePath: process.env.CHROME_PATH || undefined,
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
    },
});

client.on('qr', (qr) => {
    console.log('Escaneie o QR code abaixo com o WhatsApp (Aparelhos conectados):');
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log('✅ Robô conectado. Aguardando enquetes...');
    if (CHATS_PERMITIDOS.length) {
        console.log('Somente nos chats:', CHATS_PERMITIDOS.join(', '));
    }
});

client.on('auth_failure', (m) => console.error('Falha na autenticação:', m));
client.on('disconnected', (r) => console.log('Desconectado:', r));

client.on('message', async (msg) => {
    if (msg.type !== 'poll_creation' || msg.fromMe) return;

    const opcoes = msg.pollOptions || [];
    if (!opcoes.length) return;

    // A primeira opção é a de menor localId (ordem em que foram criadas)
    const primeira = [...opcoes].sort((a, b) => a.localId - b.localId)[0];

    try {
        if (CHATS_PERMITIDOS.length) {
            const chat = await msg.getChat();
            if (!CHATS_PERMITIDOS.includes((chat.name || '').toLowerCase())) return;
        }

        const inicio = Date.now();
        await msg.vote([primeira.name]);
        console.log(
            `🗳️  "${msg.pollName}" → votei em "${primeira.name}" (${Date.now() - inicio} ms)`
        );
    } catch (err) {
        console.error('Erro ao votar na enquete:', err);
    }
});

client.initialize();
