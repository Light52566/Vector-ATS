from sentence_transformers import SentenceTransformer
import numpy as np
from torch import Tensor
from typing import Union, Dict, List, overload
from sentence_transformers.similarity_functions import SimilarityFunction

allowed_models = ('all-MiniLM-L6-v2',
                  'intfloat/e5-small-v2',)

similarity_functions = {
    'cosine': SimilarityFunction.COSINE,
    'dot': SimilarityFunction.DOT_PRODUCT,
    'euclidean': SimilarityFunction.EUCLIDEAN,
    'manhattan': SimilarityFunction.MANHATTAN
}

def generateEmbeddings(text: str, model: str = 'all-MiniLM-L6-v2') -> Tensor:
    """
    Generate embeddings for a string.
    :param text: The input string to generate embeddings for.
    :return: A numpy array containing the embeddings.
    """
    if model not in allowed_models:
        raise ValueError(f"Model {model} is not supported. Allowed models: {allowed_models}")
    
    # Load the pre-trained model
    sentence_model = SentenceTransformer(model, trust_remote_code=True)
    
    # Generate embeddings
    embeddings = sentence_model.encode(text)
    
    return embeddings

def rankEmbeddings(embeddings: Dict[str,Tensor], reference: Tensor, sort:bool = True,
                   metric:str = 'cosine', model: str = 'all-MiniLM-L6-v2') -> Dict[str, Tensor]:
    """
    Rank embeddings based on similarity to a reference embedding.
    """
    if model not in allowed_models:
        raise ValueError(f"Model {model} is not supported. Allowed models: {allowed_models}")
    if metric not in similarity_functions:
        raise ValueError(f"Similarity function {metric} is not supported. "
                         f"Supported functions: {list(similarity_functions.keys())}")
    sentence_model = SentenceTransformer(model, trust_remote_code=True, 
                                         similarity_fn_name=similarity_functions[metric])

    similarities = {}
    for key, embedding in embeddings.items():
        similarity = sentence_model.similarity(embedding, reference)
        similarities[key] = similarity
    
    if sort:
        # Sort the similarities in descending order
        similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

    return similarities

def main():
    # Example sentences
    sentences = [
        "This is an example sentence.",
        "Each sentence is converted into a vector."
    ]

    model = 'intfloat/e5-small-v2'

    # Print the embeddings
    embeddings = {}
    for i, sentence in enumerate(sentences):
        embedding = generateEmbeddings(sentence, model=model)
        embeddings[f"sentence_{i+1}"] = embedding
        print(f"Sentence {i+1} embedding: {embedding}")
    # Example reference embedding
    reference_embedding = generateEmbeddings("This is a reference sentence.", model=model)
    print(f"Reference embedding: {reference_embedding}")

    # Rank embeddings based on similarity to the reference embedding
    ranked_similarities = rankEmbeddings(embeddings, reference_embedding, metric='cosine', model=model)
    print("Ranked similarities:")

    for key, similarity in ranked_similarities.items():
        print(f"{key}({similarity}): {sentences}")

if __name__ == "__main__":
    main()