import os
import unicodedata
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Caminhos
CAMINHO_PDFS = os.path.join(BASE_DIR, "data", "pdfs")
CAMINHO_INDICES = os.path.join(BASE_DIR, "indices")

# Ingestão
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# API OpenRouter (única API usada em todo o projeto)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Parâmetros de geração
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "45"))
MAX_TOKENS = 512
TEMPERATURE = 0.1

# Recuperação de documentos
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
    "Você é um assistente acadêmico do IF Baiano. Sua função é fornecer informações baseadas exclusivamente "
    "nos documentos institucionais carregados no sistema. "
    "Responda de forma completa, clara e acolhedora. "
    "Quando citar ou usar trechos do CONTEXTO, indique explicitamente o documento fonte de onde a informação foi extraída. "
    "Se o usuário perguntar por uma lista (como documentos, requisitos ou prazos), extraia TODOS os itens "
    "mencionados no CONTEXTO. Não se limite a citar o artigo ou o documento; apresente o conteúdo real. "
    "Se a resposta não estiver explicitamente no CONTEXTO fornecido, responda exatamente: "
    f"'{RESPOSTA_FORA_ESCOPO}' "
    "Nunca invente informações. Se o contexto apenas citar que a informação está em outro documento que NÃO está presente "
    "no contexto fornecido, informe isso claramente ao usuário. "
    "Priorize a literalidade dos documentos em relação a interpretações genéricas."
)


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()
