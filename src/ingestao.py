import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import CAMINHO_PDFS, CAMINHO_INDICES, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from text_cleaning import limpar_texto_ingestao

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def limpar_documentos(documentos):
    for doc in documentos:
        doc.page_content = limpar_texto_ingestao(doc.page_content)
    return documentos


def criar_banco():
    try:
        if not os.path.exists(CAMINHO_PDFS):
            logger.error(f"Diretório de PDFs não encontrado: {CAMINHO_PDFS}")
            return

        documentos = []
        arquivos_pdf = [f for f in os.listdir(CAMINHO_PDFS) if f.endswith(".pdf")]
        
        if not arquivos_pdf:
            logger.warning("Nenhum arquivo PDF encontrado para processar.")
            return

        for arquivo in arquivos_pdf:
            logger.info(f"Carregando: {arquivo}")
            loader = PyPDFLoader(os.path.join(CAMINHO_PDFS, arquivo))
            documentos.extend(loader.load())

        documentos = limpar_documentos(documentos)

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.split_documents(documentos)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(CAMINHO_INDICES)
        logger.info(f"Banco vetorial criado com sucesso em {CAMINHO_INDICES}!")

    except Exception as e:
        logger.error(f"Erro ao criar banco vetorial: {str(e)}")

if __name__ == "__main__":
    criar_banco()
