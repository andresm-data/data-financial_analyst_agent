"""Embeddings de texto con BAAI/bge-small-en-v1.5, calculados en CPU.

La primera ejecución descarga el modelo (~130 MB) a la caché de Hugging Face
(~/.cache/huggingface por defecto), fuera del repo.
"""
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = 'BAAI/bge-small-en-v1.5'


def get_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el modelo de embeddings bge-small forzado a CPU.

    Los vectores se normalizan para que el producto escalar equivalga a la
    similitud coseno, como recomienda BGE.
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )


def main() -> None:
    """Embebe tres frases sobre bancos e imprime el tamaño de cada vector."""
    sentences = [
        'The bank reported a higher net interest margin this year.',
        'Credit losses increased due to weaker commercial real estate loans.',
        'Deposits declined as customers moved cash into money market funds.',
    ]
    vectors = get_embeddings().embed_documents(sentences)

    print(f'Modelo: {MODEL_NAME} (CPU)')
    for sentence, vector in zip(sentences, vectors):
        print(f'{len(vector)} dimensiones <- {sentence}')


# =============================================================================
if __name__ == '__main__':
    main()
