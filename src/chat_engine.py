import logging
import os
import re
import time
from functools import lru_cache

import requests

# Importações do seu arquivo config.py (Mantidas as configurações locais do RAG)
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Leitura segura das novas variáveis de ambiente configuradas no seu .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "45"))

# CORREÇÃO COMPLETA: Protocolo HTTPS + Domínio + Rota exata de Chat da Groq
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _invocar_ia(
    prompt: str,
    system_prompt: str = PROMPT_SISTEMA_PADRAO,
    history: list[dict] | None = None,
) -> str:
    """Invoca a API da Groq Cloud utilizando o padrão de requisição da OpenAI."""
    # Limpa o token de autenticação removendo possíveis aspas residuais
    chave_limpa = str(GROQ_API_KEY).strip().replace('"', '').replace("'", "")

    if not chave_limpa or "gsk_" not in chave_limpa:
        logger.error("Chave GROQ_API_KEY ausente ou inválida no arquivo .env.")
        return "ERRO: A chave GROQ_API_KEY não foi encontrada ou é inválida no seu arquivo .env"

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if history:
        messages.extend(history[-6:]) # Envia os últimos turnos para manter a memória do chat
        
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": str(GROQ_MODEL),
        "messages": messages,
        "temperature": float(TEMPERATURE or 0.2),
        "max_tokens": int(MAX_TOKENS or 800),
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
            return f"Erro retornado pelo servidor da Groq (Código {response.status_code})."

        try:
            data = response.json()
        except Exception:
            logger.error(f"O servidor Groq não retornou dados estruturados. Resposta: {response.text[:200]}")
            return "Erro crítico na decodificação das informações vindas do servidor."
        
        # Mapeamento do formato de resposta padrão OpenAI/Groq Cloud
        if "choices" in data and len(data["choices"]) > 0:
            primeira_escolha = data["choices"][0]
            if "message" in primeira_escolha and "content" in primeira_escolha["message"]:
                texto = primeira_escolha["message"]["content"]
                if texto and texto.strip():
                    return texto.strip()
                
        logger.error(f"A API retornou uma estrutura de dados vazia ou inesperada: {data}")
        return "A inteligência artificial não conseguiu estruturar um texto de resposta."

    except requests.exceptions.Timeout:
        logger.error("Estouro de tempo limite (Timeout) na conexão com a Groq.")
        return "O servidor de inteligência artificial demorou muito para responder. Tente novamente."
    except Exception as e:
        logger.error(f"Falha de sistema na chamada da IA: {e}")
        return f"Ocorreu um erro técnico inesperado ao processar sua pergunta: {str(e)}"

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
        if token in texto_norm:
            score += 5
            
    keywords = {
        "regulamento": 2, "edital": 2, "auxilio": 2, "pincel": 2,
        "documentos": 3, "identidade": 3, "cpf": 3, "historico": 3
    }
    for termo, peso in keywords.items():
        if termo in texto_norm:
            score += peso
    return score

def _formatar_contexto(docs: list) -> str:
    blocos: list[str] = []
    total_chars = 0
    for i, doc in enumerate(docs, start=1):
        conteudo = (doc.page_content or "").strip()
        if not conteudo: continue
        
        espaco_restante = MAX_CONTEXT_CHARS - total_chars
        if espaco_restante <= 100: break 
        
        source = os.path.basename((doc.metadata or {}).get("source", "documento"))
        bloco = f"--- TRECHO {i} (Documento fonte: {source}) ---\n{conteudo}\n"
        
        if len(bloco) > espaco_restante:
            bloco = bloco[:espaco_restante] + "... [cortado]"
            
        blocos.append(bloco)
        total_chars += len(bloco)
        
    return "\n".join(blocos).strip()

def _selecionar_docs(pergunta: str, vetordb) -> tuple[list, int]:
    docs = vetordb.similarity_search(pergunta, k=RETRIEVAL_FETCH_K)
    
    docs_pontuados = sorted(
        ((doc, _pontuar_doc(pergunta, doc)) for doc in docs),
        key=lambda x: x[1],
        reverse=True,
    )
    
    melhores = [doc for doc, _ in docs_pontuados[:RETRIEVAL_K]]
    melhor_score = docs_pontuados[0][1] if docs_pontuados else 0
    return melhores, melhor_score

def _contextualizar_pergunta(pergunta: str, historico: list[dict] | None) -> str:
    """Utiliza a IA para reescrever perguntas dependentes do histórico de chat."""
    if not historico or len(historico) < 1:
        return pergunta
        
    contexto_recente = "\n".join([f"{m['role']}: {m['content']}" for m in historico[-2:]])
    prompt_refino = (
        f"Com base no histórico, reescreva a pergunta para ser clara e independente.\n"
        f"Histórico:\n{contexto_recente}\n"
        f"Pergunta: {pergunta}"
    )
    
    try:
        refinada = _invocar_ia(
            prompt_refino, 
            system_prompt="Você é um assistente que reescreve perguntas acadêmicas. Retorne apenas a pergunta reformulada.",
        )
        return refinada if len(refinada) > 5 else pergunta
    except:
        return pergunta

@lru_cache(maxsize=1)
def carregar_base_conhecimento():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    logger.info(f"Carregando índice FAISS de: {CAMINHO_INDICES}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(CAMINHO_INDICES, embeddings, allow_dangerous_deserialization=True)

def responder(
    pergunta: str,
    historico: list[dict] | None = None,
    instrucao_sistema: str = "",
) -> str:
    """Função de entrada principal para geração de respostas com arquitetura RAG."""
    try:
        inicio = time.perf_counter()
        vetordb = carregar_base_conhecimento()

        pergunta_busca = _contextualizar_pergunta(pergunta, historico)
        docs, score = _selecionar_docs(pergunta_busca, vetordb)

        if not _pergunta_tem_termo_de_dominio(pergunta) and score < MIN_RELEVANCE_SCORE:
            return RESPOSTA_FORA_ESCOPO

        contexto = _formatar_contexto(docs)
        if not contexto:
            return "Não encontrei informações oficiais sobre isso nos meus documentos."

        system_prompt = instrucao_sistema or PROMPT_SISTEMA_PADRAO
        user_prompt = (
            f"INSTRUÇÃO: Responda de forma objetiva usando apenas o CONTEXTO abaixo.\n"
            f"Quando utilizar informações do CONTEXTO, indique claramente o documento fonte de onde o trecho foi extraído.\n\n"
            f"CONTEXTO:\n{contexto}\n\n"
            f"PERGUNTA: {pergunta}"
        )

        resposta = _invocar_ia(user_prompt, system_prompt=system_prompt, history=historico)
        
        logger.info(f"Resposta gerada em {time.perf_counter() - inicio:.2f}s | Score do RAG: {score}")
        return resposta

    except Exception as e:
        logger.error(f"Erro geral detectado no fluxo do RAG: {e}")
        return "Desculpe, tive um erro interno. Pode tentar reformular a pergunta?"
