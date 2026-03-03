# Exam Reconciler - Agente de Ranking de Provas

API que recebe uma planilha Excel com questões de provas, cruza com dados históricos no banco via busca híbrida (semântica + texto), e devolve um Excel com ranking colorido por prioridade.

## Como funciona

1. **Você envia** uma planilha `.xlsx` com colunas de tema/subtema/questão
2. **O agente** normaliza os textos, gera embeddings, e busca correspondências no banco (Supabase + pgvector)
3. **Você recebe** um novo `.xlsx` com as colunas originais + score de similaridade + cor de prioridade

### Cores de prioridade

| Cor      | Hex       | Significado                     |
| -------- | --------- | ------------------------------- |
| Vermelho | `#EF4444` | 6+ questões — alta incidência   |
| Laranja  | `#F97316` | 4-5 questões                    |
| Amarelo  | `#EAB308` | 2-3 questões                    |
| Verde    | `#22C55E` | 0-1 questões — baixa incidência |

## Setup rápido

### 1. Clonar e instalar

```bash
git clone <repo-url>
cd ia-labtracker-exam-agent
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configurar o `.env`

Copie `.env.example` para `.env` e preencha:

```bash
cp .env.example .env
```

O campo obrigatório é o `DATABASE_URL` com a connection string do Supabase (use o Session Pooler):

```
DATABASE_URL=postgresql://postgres.<ref>:<senha>@aws-X-<region>.pooler.supabase.com:5432/postgres
```

### 3. Rodar migrations no banco

```bash
python -c "from src.db.client import DBClient; db = DBClient().connect(); db.run_migrations(); db.close()"
```

Isso cria as tabelas, a função de busca híbrida, e insere os dados seed da FAMERP.

### 4. Subir o servidor

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Teste: `GET http://localhost:8000/health` deve retornar `{"status": "ok"}`

## Endpoints da API

### `GET /health`

Verifica se o servidor está rodando.

### `POST /ingest/pdf`

Ingere questões de um PDF de prova no banco.

| Campo         | Tipo | Obrigatório | Descrição                                |
| ------------- | ---- | ----------- | ---------------------------------------- |
| `file`        | File | Sim         | Arquivo PDF da prova                     |
| `institution` | Text | Não         | Nome da instituição (default: "unknown") |
| `year`        | Text | Não         | Ano da prova                             |

### `POST /ingest/stats`

Ingere estatísticas de temas a partir de um PDF de métricas.

| Campo         | Tipo | Obrigatório | Descrição                                |
| ------------- | ---- | ----------- | ---------------------------------------- |
| `file`        | File | Sim         | Arquivo PDF com estatísticas             |
| `institution` | Text | Não         | Nome da instituição (default: "unknown") |

### `POST /reconcile`

Envia uma planilha Excel e recebe o ranking colorido de volta.

| Campo  | Tipo | Obrigatório | Descrição                     |
| ------ | ---- | ----------- | ----------------------------- |
| `file` | File | Sim         | Planilha `.xlsx` com questões |

> **Importante:** o arquivo deve ser enviado como um **formulário multipart (`multipart/form-data`)**
> com o campo `file`. Se você fizer a requisição com JSON ou apenas o caminho do
> arquivo, o FastAPI retornará um erro 422 com a mensagem `Field required` na
> chave `body.file`.
>
> Exemplo `curl`:
>
> ```bash
> curl -F "file=@/caminho/para/questoes.xlsx" \
>      http://localhost:8000/reconcile -o ranking.xlsx
> ```

**Resposta:** Download automático de `ranking_output.xlsx`

### Logs e debugging

O agente usa um logger nomeado `exam_reconciler` que, por padrão, propaga as
mensagens para o logger raiz. A configuração inicial (`src/utils/logging.py`) já
define `basicConfig` em nível `INFO`, mas o nível pode ser alterado via
environment variable `LOG_LEVEL` (ex: `DEBUG`, `WARNING`, `ERROR`). O valor é
lido em `src/config.py` e aplicado durante o evento de startup.

Se estiver usando `uvicorn` você pode garantir que o nível de log seja alto o
suficiente ao iniciar com:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

(note o `--log-level info`; sem ele apenas avisos/erros serão mostrados).

Durante a execução de `/reconcile` você verá informações como:

```
2026-03-03 12:34:56 INFO exam_reconciler /reconcile called with file: provas.xlsx
2026-03-03 12:34:56 INFO exam_reconciler saved uploaded file to /tmp/tmpabcd1234.xlsx
2026-03-03 12:34:56 INFO exam_reconciler reading Excel: tmpabcd1234.xlsx
2026-03-03 12:34:56 INFO exam_reconciler read 10 rows from tmpabcd1234.xlsx
2026-03-03 12:34:57 INFO exam_reconciler reconciliation produced 10 rows
2026-03-03 12:34:57 INFO exam_reconciler wrote output workbook tmpefgh5678.xlsx
```

Caso o arquivo não seja um `.xlsx` válido ou não contenha a coluna `tema`, o
sistema lança um erro 400 com a mensagem explicando o problema, e a pilha de
erro é logada.

## Usando com Insomnia

1. Importe o arquivo `insomnia_collection.json` no Insomnia (Application → Preferences → Data → Import Data → From File)
2. As 4 requisições já estarão configuradas
3. Para o `/reconcile`: clique no campo `file`, selecione seu `.xlsx`, e envie

> Se der erro de permissão de arquivo, vá em **Preferences → Security → Allowed Directories** e adicione a pasta dos seus arquivos (ex: Downloads).

## Estrutura do projeto

```
├── src/
│   ├── main.py                   # API FastAPI
│   ├── config.py                 # Configurações via .env
│   ├── cli.py                    # Comandos CLI
│   ├── db/client.py              # Cliente PostgreSQL
│   ├── normalize/normalizer.py   # Normalização de texto e sinônimos
│   ├── embeddings/embedder.py    # Geração de embeddings (local ou OpenAI)
│   ├── ingest/                   # Parsers de PDF e Excel
│   ├── retriever/                # Busca híbrida (pgvector + FTS)
│   ├── aggregator/               # Consolidação e ranking
│   └── exporters/                # Exportação Excel com cores
├── sql/                          # Migrations SQL
├── tests/                        # Testes
├── .env.example                  # Template de configuração
├── insomnia_collection.json      # Collection para importar no Insomnia
└── requirements.txt              # Dependências Python
```
