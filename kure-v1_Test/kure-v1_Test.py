from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("nlpai-lab/KURE-v1")

# Run inference
with open('kure-v1_Test_input.txt', 'r', encoding='utf-8') as f:
    # readlines()는 파일의 모든 줄을 읽어 리스트로 반환합니다.
    # 각 줄의 끝에 있는 줄바꿈 문자(\n)를 제거하기 위해 strip()을 사용합니다.
    sentences = [line.strip() for line in f.readlines()]


embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# Results for KURE-v1
# tensor([[1.0000, 0.6967, 0.5306],
#         [0.6967, 1.0000, 0.4427],
#         [0.5306, 0.4427, 1.0000]])
