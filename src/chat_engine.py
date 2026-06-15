import logging
import os
import re
import time
from functools import lru_cache
import requests

from config import (
    CAMINHO_INDICES,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS,
    MAX_TOKENS,
    MIN_RELEVANCE_SCORE,
    PROMPT_SISTEMA_PADRAO,
    RESPOSTA_FORA_ESCOPO,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    STOPWORDS_BUSCA,
    TEMPERATURE,
    TERMOS_DOMINIO,
    normalizar_texto,
    GROQ_API_KEY,   
    GROQ_MODEL,     
    API_TIMEOUT_SECONDS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _invocar_ia(
    prompt: str,
    system_prompt: str = PROMPT_SISTEMA_PADRAO,
    history: list[dict] | None = None,
    max_history_messages: int = 10,
) -> str:
    """Invoca a API da Groq Cloud utilizando o padrão de requisição da OpenAI."""
    # Usa a chave que veio do config.py
    chave_limpa = str(GROQ_API_KEY).strip().replace('"', '').replace("'", "")

    if not chave_limpa:
        logger.error("Chave GROQ_API_KEY está vazia.")
        return "ERRO: A chave de API não foi configurada corretamente no Streamlit."

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if history:
        messages.extend(_preparar_historico(history, max_messages=max_history_messages))
        
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": str(GROQ_MODEL),
        "messages": messages,
        "temperature": float(TEMPERATURE),
        "max_tokens": int(MAX_TOKENS),
    }

    headers = {
        "Authorization": f"Bearer {chave_limpa}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT_SECONDS,
        )
        
        if not response.ok:
            logger.error(f"Erro na chamada HTTP Groq {response.status_code}: {response.text}")
            return f"Erro de autenticação ou limite na API (Código {response.status_code})."

        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
                
        return "A inteligência artificial não retornou um texto válido."

    except Exception as e:
        logger.error(f"Falha na chamada da IA: {e}")
        return "Erro técnico ao processar sua pergunta."


def _preparar_historico(historico: list[dict] | None, max_messages: int = 10) -> list[dict]:
    mensagens: list[dict] = []
    for item in historico or []:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            mensagens.append({"role": role, "content": content})
    return mensagens[-max_messages:]


def _historico_em_texto(historico: list[dict] | None, max_messages: int = 6) -> str:
    mensagens = _preparar_historico(historico, max_messages=max_messages)
    linhas: list[str] = []
    nomes = {"user": "Usuário", "assistant": "Assistente"}
    for item in mensagens:
        content = item["content"].replace("\n", " ").strip()
        linhas.append(f"{nomes[item['role']]}: {content}")
    return "\n".join(linhas)


def _contextualizar_pergunta(pergunta: str, historico: list[dict] | None) -> str:
    historico_texto = _historico_em_texto(historico, max_messages=6)
    if not historico_texto:
        return pergunta

    prompt = (
        "Reescreva a pergunta atual como uma pergunta completa e independente, usando o histórico recente "
        "apenas para resolver referências como 'isso', 'ele', 'esse prazo' ou 'os documentos'. "
        "Preserve o sentido original, não responda à pergunta e não acrescente informações novas.\n\n"
        f"HISTÓRICO RECENTE:\n{historico_texto}\n\n"
        f"PERGUNTA ATUAL: {pergunta}\n\n"
        "PERGUNTA REESCRITA:"
    )
    resposta = _invocar_ia(
        prompt,
        system_prompt="Você reescreve perguntas para busca em documentos. Responda somente com a pergunta reescrita.",
        history=None,
        max_history_messages=0,
    )
    if resposta.lower().startswith("erro") or len(resposta) < 3:
        return pergunta
    return resposta.strip().strip('"')


def _tokens_relevantes(texto: str) -> list[str]:
    tokens = re.findall(r"\w+", normalizar_texto(texto))
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS_BUSCA]

def _pergunta_tem_termo_de_dominio(pergunta: str) -> bool:
    tokens = _tokens_relevantes(pergunta)
    return any(token in TERMOS_DOMINIO for token in tokens)

def _pontuar_doc(pergunta: str, doc) -> int:
    texto_norm = normalizar_texto(doc.page_content or "")
    score = 0
    tokens_pergunta = _tokens_relevantes(pergunta)
    for token in tokens_pergunta:
        if token in texto_norm: score += 5
    return score

def _formatar_contexto(docs: list) -> str:
    blocos: list[str] = []
    total_chars = 0
    for i, doc in enumerate(docs, start=1):
        conteudo = (doc.page_content or "").strip()
        if not conteudo: continue
        source = os.path.basename((doc.metadata or {}).get("source", "documento"))
        bloco = f"--- TRECHO {i} (Fonte: {source}) ---\n{conteudo}\n"
        if total_chars + len(bloco) > MAX_CONTEXT_CHARS: break
        blocos.append(bloco)
        total_chars += len(bloco)
    return "\n".join(blocos)

def _fontes_dos_docs(docs: list) -> list[str]:
    fontes: list[str] = []
    for doc in docs:
        source = os.path.basename((doc.metadata or {}).get("source", "documento"))
        if source and source not in fontes:
            fontes.append(source)
    return fontes

def _remover_fontes_inline(resposta: str) -> str:
    resposta = re.sub(r"\s*\(?Fonte:\s*[^)\n]+?\)?\s*", " ", resposta, flags=re.IGNORECASE)
    resposta = re.sub(r"\s*\(?Fontes:\s*[^)\n]+?\)?\s*", " ", resposta, flags=re.IGNORECASE)
    resposta = re.sub(r"[ \t]{2,}", " ", resposta)
    resposta = re.sub(r"\n{3,}", "\n\n", resposta)
    return resposta.strip()

def _adicionar_rodape_fontes(resposta: str, fontes: list[str]) -> str:
    resposta_limpa = _remover_fontes_inline(resposta)
    if not fontes or resposta_limpa == RESPOSTA_FORA_ESCOPO:
        return resposta_limpa
    return f"{resposta_limpa}\n\n---\n**Fonte(s):** {', '.join(fontes)}"

def _selecionar_docs(pergunta: str, vetordb) -> tuple[list, int]:
    docs = vetordb.similarity_search(pergunta, k=RETRIEVAL_FETCH_K)
    docs_pontuados = sorted(((doc, _pontuar_doc(pergunta, doc)) for doc in docs), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in docs_pontuados[:RETRIEVAL_K]], (docs_pontuados[0][1] if docs_pontuados else 0)

@lru_cache(maxsize=1)
def carregar_base_conhecimento():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(CAMINHO_INDICES, embeddings, allow_dangerous_deserialization=True)

def responder(pergunta: str, historico: list[dict] | None = None, instrucao_sistema: str = "") -> str:
    try:
        inicio = time.perf_counter()
        vetordb = carregar_base_conhecimento()
        pergunta_contextualizada = _contextualizar_pergunta(pergunta, historico)
        docs, score = _selecionar_docs(pergunta_contextualizada, vetordb)

        if not _pergunta_tem_termo_de_dominio(pergunta_contextualizada) and score < MIN_RELEVANCE_SCORE:
            return RESPOSTA_FORA_ESCOPO

        contexto = _formatar_contexto(docs)
        historico_texto = _historico_em_texto(historico, max_messages=6)
        user_prompt = (
            f"CONTEXTO:\n{contexto}\n\n"
            f"HISTÓRICO RECENTE DA CONVERSA:\n{historico_texto or 'Sem histórico anterior.'}\n\n"
            f"PERGUNTA ORIGINAL DO USUÁRIO: {pergunta}\n"
            f"PERGUNTA CONTEXTUALIZADA PARA CONSULTA: {pergunta_contextualizada}"
        )
        resposta = _invocar_ia(user_prompt, system_prompt=instrucao_sistema or PROMPT_SISTEMA_PADRAO, history=historico)
        resposta = _adicionar_rodape_fontes(resposta, _fontes_dos_docs(docs))
        
        logger.info(f"Resposta gerada em {time.perf_counter() - inicio:.2f}s | Score do RAG: {score}")
        return resposta
    except Exception as e:
        logger.error(f"Erro no RAG: {e}")
        return "Erro interno no processamento."