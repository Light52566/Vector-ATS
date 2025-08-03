from sentence_transformers import SentenceTransformer
import numpy as np
from torch import Tensor
from typing import Union

allowed_models = ('all-MiniLM-L6-v2',
                  'intfloat/e5-small-v2',)

def generateEmbeddings(text: str, model: str = 'all-MiniLM-L6-v2') -> Union[Tensor, np.ndarray]:
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

def main():
    # Example sentences
    sentences = [
        "This is an example sentence.",
        "Each sentence is converted into a vector."
    ]
    
    # Print the embeddings
    for i, sentence in enumerate(sentences):
        embedding = generateEmbeddings(sentence, model='intfloat/e5-small-v2')
        print(f"Sentence {i+1} embedding: {embedding}")

if __name__ == "__main__":
    main()