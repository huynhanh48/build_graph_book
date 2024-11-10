import numpy as np

def cosine_similarity(embedding1, embedding2):
    # Tính tích vô hướng
    dot_product = np.dot(embedding1, embedding2)
    
    # Tính độ dài của mỗi vector
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)
    
    # Tính cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity
    