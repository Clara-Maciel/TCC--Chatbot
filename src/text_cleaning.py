import html
import re


SUBSTITUICOES_UNICODE = str.maketrans({
    "\x00": " ",
    "\ufeff": " ",
    "\u00ad": "",
    "\uf0b7": "-",
    "\u2022": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\uf06e": "n",  
    "\uf074": "t",  
    "\uf064": "d",  
    "\u201c": '"',
    "\u201d": '"',
    "&": " ", 
})



def _remover_espacos_em_contatos(match):
    return re.sub(r"\s+", "", match.group(0))


def limpar_texto_ingestao(texto):
    """Remove ruidos gerais de extracao de PDF antes de indexar o documento."""
    if not texto:
        return ""

    texto = html.unescape(texto).translate(SUBSTITUICOES_UNICODE)

    # Junta palavras quebradas por hifenizacao no fim da linha: "auxi-\nlio".
    texto = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", texto)

    # Remove espacos inseridos dentro de e-mails pelo extrator/OCR.
    texto = re.sub(
        r"\b[\w.+-]+(?:\s+[\w.+-]+)?\s*@\s*[\w-]+(?:\s*\.\s*[\w-]+)+",
        _remover_espacos_em_contatos,
        texto,
    )

    # Corrige apenas separadores obvios de URL, sem juntar texto ao redor do link.
    texto = re.sub(r"\b(https?)\s*:\s*/\s*/", r"\1://", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\b(https?://)\s+", r"\1", texto, flags=re.IGNORECASE)
    texto = re.sub(
        r"\bwww\s*\.\s*[\w-]+(?:\s*\.\s*[\w-]+)+",
        _remover_espacos_em_contatos,
        texto,
        flags=re.IGNORECASE,
    )

    # Normaliza quebras de linha sem apagar separacoes de paragrafos.
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
    texto = re.sub(r"\s+([,.;:!?%])", r"\1", texto)
    texto = re.sub(r"([(])\s+", r"\1", texto)
    texto = re.sub(r"\s+([)])", r"\1", texto)

    return texto.strip()
