# LabTracker — Agente de Ranking de Provas Médicas

Sistema de IA que recebe uma planilha com temas de provas, cruza com o histórico de questões de 27 instituições, e devolve um Excel com ranking colorido por incidência — ordenado pelo currículo Manchester.

---

## O que o sistema faz

O cliente entrega uma planilha `.xlsx` com os temas que quer analisar. O agente:

1. **Lê e normaliza** cada linha da planilha (trata abreviações médicas, acentuação, variações de escrita)
2. **Busca no banco** o tema mais compatível usando três estratégias em cascata:
   - **Exato** — correspondência direta pelo nome
   - **FTS (Full-Text Search)** — busca textual flexível no PostgreSQL
   - **Semântico** — similaridade por embeddings vetoriais (modelo `all-MiniLM-L6-v2`)
3. **Opcionalmente**, passa os matches de baixa confiança por um **LLM Judge** (GPT-4o-mini) para revisão e melhoria
4. **Monta o Excel de saída** com:
   - Semana do currículo Manchester
   - Nível de confiança do match e score percentual
   - Quantas questões cada uma das 27 instituições teve sobre aquele tema
   - Cores de prioridade por incidência
5. **Gera uma aba de Cobertura Reversa** mostrando quais temas do banco _não_ foram cobertos pela planilha de entrada

A planilha de saída fica salva em `tables/ranking_output_YYYYMMDD_HHMMSS.xlsx` e também é baixada automaticamente como resposta da API.

---

## Fluxo completo

```
Cliente envia planilha.xlsx
         │
         ▼
  POST /reconcile
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  1. Leitura do Excel (src/ingest/excel_reader.py)       │
│     Aceita colunas: tema, subtema, equivalencia         │
│     Nomes em PT/EN reconhecidos automaticamente         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. Normalização (src/normalize/normalizer.py)          │
│     Remove acentos, expande siglas médicas              │
│     Ex: "HAS" → "hipertensão arterial sistêmica"        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  3. Reconciliação por linha (src/aggregator/)           │
│                                                         │
│  Para cada linha da planilha:                           │
│  a) Busca híbrida → candidatos do banco                 │
│  b) Resolve TEMA: exato → FTS → semântico               │
│  c) Resolve SUBTEMA: exato → FTS → semântico            │
│  d) Busca questões por instituição (27 universidades)   │
│  e) Linhas com score baixo passam por retry automático  │
│  f) Se use_llm=true: LLM Judge revisa baixa confiança   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  4. Cobertura Reversa (src/aggregator/consolidate.py)   │
│     Verifica quais temas do banco não foram cobertos    │
│     pela planilha — útil para identificar lacunas       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  5. Exportação Excel (src/exporters/excel_writer.py)    │
│     Ordena pelo currículo Manchester (Manchesters.xlsx) │
│     Aplica cores de incidência por célula e por linha   │
│     Salva em tables/ e retorna como download            │
└─────────────────────────────────────────────────────────┘
```

---

## Colunas da planilha de saída

| Coluna              | Descrição                                                              |
| ------------------- | ---------------------------------------------------------------------- |
| **Semana**          | Semana do currículo Manchester correspondente                          |
| **Confiança**       | Qualidade do match: 🔴 Quente / 🟠 Morno / 🟡 Frio / ⚪ Sem match      |
| **Score**           | Percentual de confiança do match (ex: 80%)                             |
| **Tema (entrada)**  | Tema como veio na planilha original                                    |
| **Equivalência**    | Equivalência resolvida (tema \| subtema do banco)                      |
| **Tema**            | Tema normalizado encontrado no banco                                   |
| **Subtema**         | Subtema normalizado encontrado no banco                                |
| **AMP-PR … USP SP** | Quantidade de questões de cada uma das 27 instituições sobre esse tema |
| **Observações**     | Detalhes do match, fonte dos dados, candidatos encontrados             |

### Cores nas células de instituição

Cada célula de instituição tem cor própria baseada na quantidade de questões daquela instituição para aquele tema:

| Cor         | Questões | Significado                 |
| ----------- | -------- | --------------------------- |
| 🔴 Vermelho | 6+       | Alta incidência — cai muito |
| 🟠 Laranja  | 4–5      | Incidência moderada-alta    |
| 🟡 Amarelo  | 2–3      | Incidência moderada         |
| 🟢 Verde    | 0–1      | Baixa incidência            |

---

## Instituições cobertas (27)

AMP-PR · AMRIGS · CERMAM · FAMENE · FAMERP · FELUMA · HCPA · HEVV · HIAE · IAMSPE · INTO RJ · PSU-MG · PSU-GO · REVALIDA INEP · SCMSP · SES-DF · SES-PE · SÍRIO · SUS BA · SUS SP · UEPA · UERJ · UNESP · UNICAMP · UNIFESP · USP RP · USP SP

---

## Como usar hoje (passo a passo para o suporte)

### Pré-requisitos

- Python 3.11+ instalado
- Servidor rodando (ver seção Setup abaixo)
- Planilha `.xlsx` com pelo menos a coluna **`tema`** preenchida
  - Colunas opcionais reconhecidas: `subtema`, `equivalencia`

### 1. Verificar se o servidor está no ar

```bash
curl http://localhost:8000/health
```

Resposta esperada: `{"status": "ok"}`

Se não responder, subir o servidor (ver passo 6 do Setup).

### 2. Enviar a planilha para reconciliação

**Via curl (Windows — terminal):**

```bash
curl -F "file=@C:\caminho\para\questoes.xlsx" http://localhost:8000/reconcile -o ranking.xlsx
```

**Com LLM Judge ativado** (melhora matches de baixa confiança, mais lento):

```bash
curl -F "file=@C:\caminho\para\questoes.xlsx" "http://localhost:8000/reconcile?use_llm=true" -o ranking.xlsx
```

**Via Insomnia / Postman:**

1. Criar requisição `POST http://localhost:8000/reconcile`
2. Body → `multipart/form-data`
3. Adicionar campo `file` do tipo **File** e selecionar o `.xlsx`
4. Enviar — a resposta já será o download do Excel

> **Atenção:** Nunca enviar como JSON ou form-urlencoded. Deve ser `multipart/form-data`.
> Se aparecer erro `422 Field required`, o arquivo não foi enviado corretamente.

### 3. Onde fica a planilha gerada

A planilha é salva automaticamente em:

```
C:\projetos\ia-labtracker-exam-agent\tables\ranking_output_YYYYMMDD_HHMMSS.xlsx
```

Exemplo: `tables/ranking_output_20260308_224229.xlsx`

O arquivo mais recente é sempre o último gerado. A resposta da API já inclui o download direto — mas o arquivo fica salvo em `tables/` caso precise reenviar ou auditar depois.

---

## Setup completo (primeira vez)

### 1. Clonar e instalar dependências

```bash
git clone <repo-url>
cd ia-labtracker-exam-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar o `.env`

```bash
copy .env.example .env
```

Editar o `.env` e preencher:

```env
# Banco de dados — usar connection string do Supabase (Session Pooler)
DATABASE_URL=postgresql://postgres.<ref>:<senha>@aws-X-<region>.pooler.supabase.com:5432/postgres

# Embeddings — "local" usa modelo embutido (gratuito), "openai" usa API da OpenAI
EMBEDDINGS_PROVIDER=local

# Opcional: LLM Judge para melhorar matches de baixa confiança
# LLM_JUDGE_ENABLED=true
# OPENAI_API_KEY=sk-...
# LLM_JUDGE_MODEL=gpt-4o-mini
```

### 3. Aplicar migrations no banco

```bash
python -c "from src.db.client import DBClient; db = DBClient().connect(); db.run_migrations(); db.close()"
```

Isso cria as tabelas (`theme_stats`, `questions`), a função de busca híbrida, os índices vetoriais, e insere os dados seed da FAMERP.

### 4. Garantir que o `Manchesters.xlsx` está na raiz

```
C:\projetos\ia-labtracker-exam-agent\Manchesters.xlsx
```

Esse arquivo define a ordem das semanas do currículo Manchester. Se não existir, as linhas ainda são geradas mas sem ordenação por semana (aparecem por score).

### 5. (Opcional) Ingerir dados de outras instituições

Para adicionar dados de mais uma prova/instituição ao banco:

```bash
# Ingerir estatísticas de um PDF de ranking/métricas
curl -F "file=@AMRIGS.pdf" -F "institution=AMRIGS" http://localhost:8000/ingest/stats

# Ingerir questões de um PDF de prova
curl -F "file=@prova_amrigs_2024.pdf" -F "institution=AMRIGS" -F "year=2024" http://localhost:8000/ingest/pdf
```

> O nome da instituição deve ser exatamente como está na lista das 27 (ex: `HEVV - HOSPITAL EVANGÉLICO DE VILA VELHA`, `INTO RJ`). Caso contrário as colunas correspondentes aparecerão com 0 questões.

### 6. Subir o servidor

```bash
cd C:\projetos\ia-labtracker-exam-agent
.venv\Scripts\activate
uvicorn src.main:app --host 0.0.0.0 --port 8000 --log-level info
```

---

## Variáveis de ambiente completas

| Variável               | Padrão                                         | Descrição                                                              |
| ---------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| `DATABASE_URL`         | `postgresql://...localhost.../exam_reconciler` | Connection string do banco                                             |
| `EMBEDDINGS_PROVIDER`  | `local`                                        | `local` (gratuito) ou `openai`                                         |
| `OPENAI_API_KEY`       | _(vazio)_                                      | Necessário se `EMBEDDINGS_PROVIDER=openai` ou `LLM_JUDGE_ENABLED=true` |
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2`                             | Modelo local de embeddings                                             |
| `EMBEDDING_DIM`        | `384`                                          | Dimensão dos vetores                                                   |
| `HYBRID_ALPHA`         | `0.7`                                          | Peso dos embeddings na busca híbrida                                   |
| `HYBRID_BETA`          | `0.3`                                          | Peso do FTS na busca híbrida                                           |
| `SIMILARITY_THRESHOLD` | `0.4`                                          | Score mínimo para aceitar um match semântico                           |
| `RETRIEVER_TOP_K`      | `5`                                            | Número de candidatos retornados pela busca                             |
| `LLM_JUDGE_ENABLED`    | `false`                                        | Ativa o LLM Judge para revisão de baixa confiança                      |
| `LLM_JUDGE_MODEL`      | `gpt-4o-mini`                                  | Modelo OpenAI usado pelo Judge                                         |
| `LLM_JUDGE_BASE_URL`   | _(vazio)_                                      | Base URL customizada (para proxies/outros provedores)                  |
| `LLM_JUDGE_THRESHOLD`  | `0.60`                                         | Score abaixo desse valor vai para revisão pelo LLM                     |
| `LOG_LEVEL`            | `INFO`                                         | Nível de log: `DEBUG`, `INFO`, `WARNING`, `ERROR`                      |
| `API_HOST`             | `0.0.0.0`                                      | Host da API                                                            |
| `API_PORT`             | `8000`                                         | Porta da API                                                           |

---

## Endpoints da API

### `GET /health`

Verifica se o servidor está no ar.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

### `POST /reconcile`

**Endpoint principal.** Recebe a planilha, processa e devolve o Excel com ranking.

| Parâmetro | Tipo             | Obrigatório | Descrição                                        |
| --------- | ---------------- | ----------- | ------------------------------------------------ |
| `file`    | File (multipart) | Sim         | Planilha `.xlsx` com coluna `tema`               |
| `use_llm` | Query param      | Não         | `true` para ativar o LLM Judge (padrão: `false`) |

```bash
# Sem LLM Judge (mais rápido)
curl -F "file=@questoes.xlsx" http://localhost:8000/reconcile -o resultado.xlsx

# Com LLM Judge (mais preciso em casos difíceis)
curl -F "file=@questoes.xlsx" "http://localhost:8000/reconcile?use_llm=true" -o resultado.xlsx
```

**Resposta:** Download direto do `ranking_output_YYYYMMDD_HHMMSS.xlsx`.

---

### `POST /ingest/stats`

Ingere estatísticas de temas de um PDF (ranking por instituição).

| Parâmetro     | Tipo             | Obrigatório | Descrição                                 |
| ------------- | ---------------- | ----------- | ----------------------------------------- |
| `file`        | File (multipart) | Sim         | PDF com estatísticas de temas             |
| `institution` | Form field       | Não         | Nome da instituição (padrão: `"unknown"`) |

```bash
curl -F "file=@AMRIGS_stats.pdf" -F "institution=AMRIGS" http://localhost:8000/ingest/stats
```

---

### `POST /ingest/pdf`

Ingere questões de um PDF de prova no banco.

| Parâmetro     | Tipo             | Obrigatório | Descrição           |
| ------------- | ---------------- | ----------- | ------------------- |
| `file`        | File (multipart) | Sim         | PDF da prova        |
| `institution` | Form field       | Não         | Nome da instituição |
| `year`        | Form field       | Não         | Ano da prova        |

```bash
curl -F "file=@prova.pdf" -F "institution=FAMERP" -F "year=2024" http://localhost:8000/ingest/pdf
```

---

## Diagnóstico de problemas comuns

### Erro `422 Field required`

O arquivo não foi enviado como multipart. Verificar que o campo `file` está configurado como **File** (não Text) no Insomnia/Postman.

### Erro `400 could not parse excel file`

A planilha não tem a coluna `tema` (ou `theme`). Verificar os cabeçalhos da planilha de entrada.

### Planilha com "0 questões" para todas as células de uma linha

O tema da linha não foi encontrado no banco. Verificar a coluna **Confiança** — se aparecer "⚪ Sem match", o tema precisa ser adicionado ao banco ou o texto está muito diferente do que foi cadastrado.

### Confiança "Frio" ou "Sem match" em muitas linhas

Significa que o agente não encontrou correspondência forte no banco. Possíveis causas:

- Tema escrito de forma muito diferente do que está cadastrado no banco
- Tema ainda não foi ingerido para aquela instituição
- Usar `use_llm=true` pode melhorar esses casos

### Coluna de uma instituição sempre zerada

O nome da instituição nos dados do banco não está exatamente igual ao da lista das 27. Os nomes são case-sensitive e devem ser idênticos (ex: `HEVV - HOSPITAL EVANGÉLICO DE VILA VELHA`, não `HEVV`).

### Servidor não sobe / erro de banco

Verificar se o `DATABASE_URL` no `.env` está correto e se as migrations foram aplicadas (passo 3 do Setup).

### Logs para debug detalhado

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

Ou no `.env`:

```env
LOG_LEVEL=DEBUG
```

Durante uma requisição `/reconcile` você verá cada etapa: leitura do Excel, normalização, tentativas de match (exato → FTS → semântico), retry de linhas com score baixo, e exportação do resultado.

---

## Estrutura do projeto

```
├── src/
│   ├── main.py                         # API FastAPI — endpoints e startup
│   ├── config.py                       # Configurações via variáveis de ambiente
│   ├── cli.py                          # Comandos CLI alternativos
│   ├── db/
│   │   └── client.py                   # Cliente PostgreSQL com cache e busca híbrida
│   ├── normalize/
│   │   └── normalizer.py               # Normalização de texto + dicionário de siglas médicas
│   ├── embeddings/
│   │   └── embedder.py                 # Geração de embeddings (local ou OpenAI)
│   ├── ingest/
│   │   ├── excel_reader.py             # Leitura da planilha de entrada
│   │   └── pdf_parser.py               # Extração de questões e stats de PDFs
│   ├── retriever/
│   │   └── hybrid_retriever.py         # Busca híbrida (pgvector + FTS)
│   ├── aggregator/
│   │   ├── models.py                   # Dataclasses, constantes, lista de instituições
│   │   ├── matchers.py                 # Lógica de match: exato → FTS → semântico
│   │   ├── consolidate.py              # Pipeline principal + cobertura reversa
│   │   └── llm_refinement.py           # LLM Judge para matches de baixa confiança
│   ├── exporters/
│   │   └── excel_writer.py             # Exportação Excel com cores e estilos
│   └── utils/
│       ├── logging.py                  # Configuração de logger
│       └── manchester_order.py         # Ordenação pelo currículo Manchester
├── sql/
│   ├── 000_init_tables.sql             # Tabelas base (questions, themes, ingest_log)
│   ├── 001_hybrid_search_function.sql  # Função SQL de busca híbrida
│   ├── 002_theme_stats_table.sql       # Tabela de estatísticas por tema/instituição
│   ├── 003_seed_famerp.sql             # Dados iniciais da FAMERP
│   ├── 004_theme_stats_embedding.sql   # Coluna de embedding em theme_stats
│   └── 005_restore_cor_hex.sql         # Correção de cores
├── tables/                             # Planilhas geradas (saída do /reconcile)
├── Manchesters.xlsx                    # Currículo Manchester — define ordem das semanas
├── .env.example                        # Template de configuração
└── requirements.txt                    # Dependências Python
```
