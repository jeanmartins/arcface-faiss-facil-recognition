
import os
import cv2
from insightface.app import FaceAnalysis
import numpy as np
import ssl
import faiss
import pickle
import time

ssl._create_default_https_context = ssl._create_unverified_context

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

PASTA_REFERENCIAS = 'rostos_dataset'
PASTA_TESTES = 'testes'

def carregar_database(pasta):
    database = {}
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        if not arquivo.lower().endswith(('.jpg', '.jpeg', '.png')): 
            continue
        nome = os.path.splitext(arquivo)[0]
        
        img = cv2.imread(caminho)
        if img is None:
            print(f"[ERRO] Não foi possível carregar a imagem {arquivo}")
            continue
        
        faces = app.get(img)
        if not faces:
            print(f"[AVISO] Nenhum rosto detectado em {arquivo}")
            continue
        emb = faces[0].embedding
        database[nome] = emb
    return database


def reconhecer(imagem_path, index):
    img = cv2.imread(imagem_path)
    if img is None:
        return "Erro ao carregar a imagem", None

    faces = app.get(img)
    if not faces:
        return "Rosto não detectado", None

    emb_teste = faces[0].embedding.reshape(1, -1).astype('float32')
    distances, indices = index.search(emb_teste, k=5)  # k = número de vizinhos mais próximos

    return indices[0]

def criar_index(database):
    nomes = list(database.keys())
    embeddings_referencias = np.array(list(database.values())).astype('float32')
    index = faiss.IndexFlatL2(embeddings_referencias.shape[1])  # L2 = Distância Euclidiana
    index.add(embeddings_referencias)
    
    faiss.write_index(index, "indice_rostos.index")
    with open("nomes.pkl", "wb") as f:
        pickle.dump(nomes, f)
            
    return index, nomes

def main():
    tick = time.time()
    if os.path.exists("indice_rostos.index") and os.path.exists("nomes.pkl"):
        index = faiss.read_index("indice_rostos.index")
        with open("nomes.pkl", "rb") as f:
            nomes = pickle.load(f)
    else:
        database = carregar_database(PASTA_REFERENCIAS)
        print(f"[INFO] {len(database)} rostos carregados na base.")
        index, nomes = criar_index(database)


    for arquivo in os.listdir(PASTA_TESTES):
        if not arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        caminho = os.path.join(PASTA_TESTES, arquivo)
        indices = reconhecer(caminho, index)
        tack = time.time()
        for idx in indices:
            print(f"{arquivo} → {nomes[idx]}")
        print(f"Tempo de execução total : {tack-tick}")
if __name__ == '__main__':
    main()
