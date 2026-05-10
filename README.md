# 🎓 Assistente Acadêmico IF Baiano

## Resumo

Este projeto apresenta um protótipo de assistente conversacional baseado em Recuperação de Informação e Geração de Respostas (RAG), direcionado ao atendimento de dúvidas institucionais do Instituto Federal Baiano. O sistema utiliza documentos oficiais, tais como editais, regulamentos e manuais de estágio, como base de conhecimento para a geração de respostas contextualizadas e fundamentadas.

## Objetivo

O objetivo principal é demonstrar a viabilidade de uma solução híbrida que integra técnicas de embeddings semânticos e busca vetorial a modelos de linguagem, permitindo a extração de informações relevantes de um conjunto de documentos e a posterior composição de respostas coerentes.

## Metodologia

A implementação segue as etapas abaixo:

1. Pré-processamento de textos extraídos de arquivos PDF.
2. Extração de representações vetoriais por meio de embeddings de sentença.
3. Armazenamento e consulta semântica em uma base de vetores local.
4. Recuperação de trechos relevantes e geração de resposta pelo modelo de linguagem.
5. Apresentação da interação por meio de interface web.

## Tecnologias empregadas

| Componente | Tecnologia |
|---|---|
| Modelo de linguagem | Anthropic Claude (via API) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (execução local) |
| Armazenamento vetorial | FAISS (local) |
| Interface web | Streamlit |

## Requisitos

- Python 3.10+ ou compatível.
- Acesso à internet para comunicação com a API de Anthropic.
- Biblioteca `streamlit` instalada.
- Variáveis de ambiente configuradas conforme arquivo de exemplo `.env.example`.

## Instalação e execução

1. Clone o repositório:

```bash
git clone https://github.com/Clara-Maciel/TCC--Chatbot.git
cd TCC-IF-CHATBOT
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:

```bash
cp .env 
```

Edite o arquivo `.env` e insira a sua chave de API do Groq em `GROQ_API_KEY`.

4. Realize a ingestão de documentos PDF:

```bash
python src/ingestao.py
```

Essa etapa processa os arquivos em `data/pdfs/`, extrai texto e constrói o índice vetorial local.

5. Execute a interface web:

```bash
streamlit run src/interface.py
```

6. Acesse a aplicação no navegador no endereço indicado pelo Streamlit.

## Adição de novos documentos

Para incorporar novos arquivos institucionais, copie os arquivos `.pdf` para `data/pdfs/` e execute novamente:

```bash
python src/ingestao.py
```

## Estrutura do repositório

```text
TCC-IF-CHATBOT/
├── data/pdfs/          # Documentos institucionais em formato PDF
├── indices/            # Índice FAISS gerado localmente
├── src/
│   ├── config.py       # Configurações e constantes do projeto
│   ├── chat_engine.py  # Navegação RAG e invocação do modelo de linguagem
│   ├── interface.py    # Interface de usuário em Streamlit
│   ├── ingestao.py     # Rotina de leitura, pré-processamento e indexação
│   ├── text_cleaning.py # Limpeza e normalização de texto
│   └── __init__.py
├── .env.example        # Modelo de variáveis de ambiente
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação do projeto
```

## Considerações finais

Este repositório destina-se a fins acadêmicos e de pesquisa, servindo como prova de conceito para sistemas de assistência automatizada baseados em documentos institucionais. A arquitetura proposta pode ser ampliada com mecanismos adicionais de filtragem, controle de confiabilidade das respostas e suporte a múltiplas fontes de conhecimento.
