# 🎓 Assistente Acadêmico IF Baiano

Chatbot RAG para responder dúvidas institucionais com base nos documentos oficiais do IF Baiano (editais, regulamentos, manuais de estágio etc.).

## Tecnologias

| Componente | Tecnologia |
|---|---|
| LLM | Anthropic Claude (API única) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vetor store | FAISS (local) |
| Interface | Streamlit |

## Configuração

```bash
# 1. Clone o repositório
git clone ...
cd TCC-IF

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env e insira sua ANTHROPIC_API_KEY

# 4. Indexe os PDFs (apenas na primeira vez ou ao adicionar documentos)
python src/ingestao.py

# 5. Inicie a interface
streamlit run src/interface.py
```

## Adicionando documentos

Coloque arquivos `.pdf` na pasta `data/pdfs/` e re-execute:

```bash
python src/ingestao.py
```

## Estrutura

```
TCC-IF/
├── data/pdfs/          # Documentos institucionais
├── indices/            # Índice FAISS (gerado localmente, não versionado)
├── src/
│   ├── config.py       # Configurações centralizadas
│   ├── chat_engine.py  # Lógica RAG + chamada à API
│   ├── interface.py    # Interface Streamlit
│   ├── ingestao.py     # Indexação dos PDFs
│   └── text_cleaning.py
├── .env.example
├── requirements.txt
└── README.md
```
