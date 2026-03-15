import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from src.core.skip_gram_negative_sampling import Word2VecSGNS


def plot_capital_relationships(model: Word2VecSGNS):
    V = model.V

    pairs = [
        ("germany", "berlin"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("netherlands", "amsterdam"),
        ("austria", "vienna"),
        ("czech republic", "prague"),
        ("denmark", "copenhagen"),
        ("finland", "helsinki"),
    ]

    # Leave only pairs that exist in the vocabulary
    pairs = [
        (country, capital)
        for country, capital in pairs
        if country in model.word2id and capital in model.word2id
    ]

    if not pairs:
        raise ValueError("No country-capital pairs were found in the model vocabulary.")

    all_words = [w for pair in pairs for w in pair]
    vectors = np.array([V[model.word2id[w]] for w in all_words])

    # PCA to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    country_coords = coords[0::2]
    capital_coords = coords[1::2]

    # DataFrames for convenient plotting
    df_countries = pd.DataFrame(
        {
            "word": [country for country, _ in pairs],
            "x": country_coords[:, 0],
            "y": country_coords[:, 1],
            "type": "country",
        }
    )

    df_capitals = pd.DataFrame(
        {
            "word": [capital for _, capital in pairs],
            "x": capital_coords[:, 0],
            "y": capital_coords[:, 1],
            "type": "capital",
        }
    )

    fig = go.Figure()

    # Countries
    fig.add_trace(
        go.Scatter(
            x=df_countries["x"],
            y=df_countries["y"],
            mode="markers+text",
            text=df_countries["word"],
            textposition="middle left",
            name="Countries",
            hovertemplate="<b>%{text}</b><br>Type: country<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            marker=dict(size=10),
        )
    )

    # Capitals
    fig.add_trace(
        go.Scatter(
            x=df_capitals["x"],
            y=df_capitals["y"],
            mode="markers+text",
            text=df_capitals["word"],
            textposition="middle right",
            name="Capitals",
            hovertemplate="<b>%{text}</b><br>Type: capital<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            marker=dict(size=10),
        )
    )

    # Dashed lines between country and capital
    for i, (country, capital) in enumerate(pairs):
        x1, y1 = country_coords[i]
        x2, y2 = capital_coords[i]

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

    fig.update_layout(
        title="Country and Capital Vectors Projected by PCA",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        width=800,
        height=600,
        template="plotly_white",
    )

    fig.show()
    fig.write_html("outputs/docs/capital_relationships.docs")
