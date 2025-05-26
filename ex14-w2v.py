from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 範例語料
sentences = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["i", "love", "machine", "learning"],
    ["deep", "learning", "is", "a", "subfield", "of", "machine", "learning"],
    ["natural", "language", "processing", "is", "fun"]
]

# CBOW 模型
cbow_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=0, epochs=100)

# Skip-Gram 模型
skipgram_model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1, epochs=100)

# 🧠 抽取詞向量並使用 t-SNE 降維
def tsne_plot(model, title):
    words = list(model.wv.index_to_key)
    word_vectors = np.array([model.wv[word] for word in words])  # ✅ 修正這裡

    tsne = TSNE(n_components=2, random_state=0, perplexity=5, n_iter=1000)
    Y = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(10, 6))
    plt.scatter(Y[:, 0], Y[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
    plt.title(title)
    plt.grid(True)
    plt.show()

# 可視化
tsne_plot(cbow_model, "CBOW Word2Vec t-SNE")
tsne_plot(skipgram_model, "Skip-Gram Word2Vec t-SNE")