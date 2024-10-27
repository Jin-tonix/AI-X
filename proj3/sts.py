# step1

from sentence_transformers import SentenceTransformer
# step2
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# step3
sentence1 = "집에 갑시다.",
sentence2 = "안녕하세요.",

# step4
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

print(embedding1.shape)

# step5
similarities = model.similarity(embedding1, embedding2)
print(similarities)
