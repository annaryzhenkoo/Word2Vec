from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
import pandas as pd
from src.core.skip_gram_negative_sampling import Word2VecSGNS


def plot_example_cluster(model: Word2VecSGNS):
    groups = {
        "days": [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ],
        "months": [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        "directions": ["north", "south", "east", "west"],
        "colors": ["red", "blue", "green", "yellow", "black", "white"],
        "animals": ["dog", "cat", "horse", "cow", "sheep", "goat"],
    }

    selected_embeddings = []
    selected_words = []
    true_groups = []

    for group, words in groups.items():
        for w in words:
            if w in model.word2id:
                selected_embeddings.append(model.V[model.word2id[w]])
                selected_words.append(w)
                true_groups.append(group)

    if len(selected_embeddings) < 2:
        print("Not enough words for intrinsic_evaluation!")
        return

    selected_embeddings = np.array(selected_embeddings)
    print("Shape:", selected_embeddings.shape)

    n_groups = len(groups)
    n_clusters = min(n_groups, len(selected_embeddings))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(selected_embeddings)

    perplexity = min(5, len(selected_embeddings) - 1)
    reduced = TSNE(
        n_components=2, random_state=42, perplexity=perplexity
    ).fit_transform(selected_embeddings)

    df = pd.DataFrame(
        {
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "word": selected_words,
            "true_group": true_groups,
            "cluster": clusters.astype(str),
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        text="word",
        hover_data=["true_group"],
        title="KMeans clustering of Word2Vec embeddings",
    )

    fig.update_traces(textposition="top center")
    fig.write_html("outputs/docs/clusters_map.docs")
    fig.show()
