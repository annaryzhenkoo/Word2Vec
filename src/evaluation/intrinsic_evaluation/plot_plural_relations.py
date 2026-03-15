import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from src.core.skip_gram_negative_sampling import Word2VecSGNS

def plot_plural_relations(model: Word2VecSGNS):

    plural_pairs = [
        ("cat", "cats"),
        ("dog", "dogs"),
        ("car", "cars"),
        ("tree", "trees"),
        ("house", "houses"),
        ("book", "books"),
        ("river", "rivers"),
        ("city", "cities"),
        ("mouse", "mice"),
    ]

    sing = []
    plur = []
    labels_s = []
    labels_p = []

    for s, p in plural_pairs:
        if s in model.word2id and p in model.word2id:
            sing.append(model.V[model.word2id[s]])
            plur.append(model.V[model.word2id[p]])
            labels_s.append(s)
            labels_p.append(p)

    if len(sing) == 0:
        raise ValueError("No singular/plural pairs found in the dictionary!")

    X = np.vstack(sing + plur)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    n = len(sing)

    sing_pca = X2[:n]
    plur_pca = X2[n:]

    df_sing = pd.DataFrame(
        {"word": labels_s, "x": sing_pca[:, 0], "y": sing_pca[:, 1], "type": "singular"}
    )

    df_plur = pd.DataFrame(
        {"word": labels_p, "x": plur_pca[:, 0], "y": plur_pca[:, 1], "type": "plural"}
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_sing["x"],
            y=df_sing["y"],
            mode="markers+text",
            text=df_sing["word"],
            textposition="middle left",
            name="singular",
            marker=dict(size=10),
            hovertemplate="<b>%{text}</b><br>Type: singular<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plur["x"],
            y=df_plur["y"],
            mode="markers+text",
            text=df_plur["word"],
            textposition="middle right",
            name="plural",
            marker=dict(size=10),
            hovertemplate="<b>%{text}</b><br>Type: plural<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=[sing_pca[i, 0], plur_pca[i, 0]],
                y=[sing_pca[i, 1], plur_pca[i, 1]],
                mode="lines",
                line=dict(dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Singular → Plural relation in embeddings",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        width=1100,
        height=800,
        template="plotly_white",
    )

    fig.show()
    fig.write_html("outputs/docs/plural_relations.docs")
