import os
import unicodedata
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CAMINHO_PDFS = os.path.join(BASE_DIR, "data", "pdfs")
CAMINHO_INDICES = os.path.join(BASE_DIR, "indices")

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = st.secrets.get("GROQ_MODEL") or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    LLM_PROVIDER = st.secrets.get("LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "groq")
    API_TIMEOUT_SECONDS = int(st.secrets.get("API_TIMEOUT_SECONDS") or os.getenv("API_TIMEOUT_SECONDS", "45"))
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL") or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "45"))
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512
TEMPERATURE = 0.0

RETRIEVAL_K = 6
RETRIEVAL_FETCH_K = 14
MAX_CONTEXT_CHARS = 4500
MIN_RELEVANCE_SCORE = 4

STOPWORDS_BUSCA = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "deve", "devo",
    "do", "dos", "e", "em", "na", "nas", "no", "nos", "o", "os", "ou", "para",
    "por", "que", "qual", "quais", "ser", "se", "só", "so", "um", "uma",
    "meu", "minha", "me", "eu",
}

TERMOS_DOMINIO = {
    "academico", "academica", "aluno", "aluna", "auxilio", "baiano", "bolsa",
    "cae", "campus", "curso", "discente", "documento", "edital", "ensino",
    "estagio", "estudante", "graduacao", "guanambi", "if", "ifbaiano",
    "instituto", "matricula", "monitoria", "pae", "pincel", "propac",
    "regulamento", "secretaria", "suap", "superior",
}

RESPOSTA_FORA_ESCOPO = (
    "Posso ajudar apenas com dúvidas acadêmicas institucionais relacionadas aos documentos do IF Baiano "
    "carregados no sistema. Por favor, pergunte sobre regulamentos, editais, auxílios, "
    "secretaria ou procedimentos acadêmicos em geral."
)

PROMPT_SISTEMA_PADRAO = (
    "Você é o Assistente Virtual Inteligente do IF Baiano (Campus Guanambi), um chatbot "
    "prestativo, educado e focado em ajudar os estudantes de Análise e Desenvolvimento de Sistemas (ADS).\n\n"
    "Sua tarefa é responder à pergunta do usuário utilizando ESTRITAMENTE o contexto dos "
    "documentos institucionais fornecidos abaixo.\n\n"
    "Diretrizes rígidas:\n"
    "1. Responda de forma direta, clara e puramente factual, baseando-se apenas nos fragmentos fornecidos.\n"
    "2. Se a resposta para a pergunta não estiver presente no contexto abaixo, responda exatamente: "
    f"'{RESPOSTA_FORA_ESCOPO}'\n"
    "3. Não utilize o seu conhecimento prévio ou externo para responder a perguntas sobre "
    "procedimentos institucionais, editais ou normas da instituição.\n"
    "4. Mantenha um tom profissional, acolhedor e humilde.\n"
    "Não cite fontes no corpo da resposta; a aplicação adicionará as fontes automaticamente no rodapé."
)

def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()