from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Counter
from src.core.skip_gram_negative_sampling import Word2VecSGNS


def plot_top_words_map(model: Word2VecSGNS, popular_words: int, word_counts: Counter):
    selected_embeddings = []
    selected_words = []

    for word, _ in word_counts.most_common(popular_words):
        id_ = model.word2id[word]
        selected_embeddings.append(model.V[id_])
        selected_words.append(word)

    selected_embeddings = np.array(selected_embeddings)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(selected_embeddings)

    df = pd.DataFrame(
        {
            "word": selected_words,
            "x": emb_2d[:, 0],
            "y": emb_2d[:, 1],
        }
    )

    fig = px.scatter(df, x="x", y="y", text="word", hover_name="word")

    fig.update_traces(marker=dict(size=7))
    fig.update_layout(width=1000, height=800)

    fig.write_html("outputs/docs/top_words_map.docs")
    fig.show()
