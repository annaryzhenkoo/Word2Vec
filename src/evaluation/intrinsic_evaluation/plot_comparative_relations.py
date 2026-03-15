import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from src.core.skip_gram_negative_sampling import Word2VecSGNS


def plot_comparative_relations(model: Word2VecSGNS):
    V = model.V

    triples = [
        ("small", "smaller", "smallest"),
        ("big", "bigger", "biggest"),
        ("strong", "stronger", "strongest"),
        ("fast", "faster", "fastest"),
        ("long", "longer", "longest"),
        ("high", "higher", "highest"),
        ("young", "younger", "youngest"),
    ]

    triples = [
        (a, b, c)
        for a, b, c in triples
        if a in model.word2id and b in model.word2id and c in model.word2id
    ]

    if len(triples) == 0:
        raise ValueError("No triples in the dictionary!")

    all_words = [w for triple in triples for w in triple]
    vectors = np.array([V[model.word2id[w]] for w in all_words])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    base_coords = coords[0::3]
    comp_coords = coords[1::3]
    sup_coords = coords[2::3]

    df_base = pd.DataFrame(
        {
            "word": [a for a, _, _ in triples],
            "x": base_coords[:, 0],
            "y": base_coords[:, 1],
            "type": "base",
        }
    )

    df_comp = pd.DataFrame(
        {
            "word": [b for _, b, _ in triples],
            "x": comp_coords[:, 0],
            "y": comp_coords[:, 1],
            "type": "comparative",
        }
    )

    df_sup = pd.DataFrame(
        {
            "word": [c for _, _, c in triples],
            "x": sup_coords[:, 0],
            "y": sup_coords[:, 1],
            "type": "superlative",
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_base["x"],
            y=df_base["y"],
            mode="markers+text",
            text=df_base["word"],
            textposition="bottom center",
            name="base",
            marker=dict(size=10),
            hovertemplate="<b>%{text}</b><br>Type: base<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_comp["x"],
            y=df_comp["y"],
            mode="markers+text",
            text=df_comp["word"],
            textposition="bottom center",
            name="comparative",
            marker=dict(size=10),
            hovertemplate="<b>%{text}</b><br>Type: comparative<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_sup["x"],
            y=df_sup["y"],
            mode="markers+text",
            text=df_sup["word"],
            textposition="bottom center",
            name="superlative",
            marker=dict(size=10),
            hovertemplate="<b>%{text}</b><br>Type: superlative<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )

    for i, (base, comp, sup) in enumerate(triples):
        x1, y1 = base_coords[i]
        x2, y2 = comp_coords[i]
        x3, y3 = sup_coords[i]

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x2, x3],
                y=[y2, y3],
                mode="lines",
                line=dict(dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Adjective → Comparative → Superlative projected by PCA",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        width=970,
        height=800,
        template="plotly_white",
    )

    fig.show()
    fig.write_html("outputs/docs/comparative_relations.docs")
