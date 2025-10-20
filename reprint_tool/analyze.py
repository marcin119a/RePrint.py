import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import plotly.graph_objects as go
import plotly.figure_factory as ff

def normalize(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

def calculate_rmse(x, y):
    x_normalized = normalize(x)
    y_normalized = normalize(y)
    return np.sqrt(np.nanmean((x_normalized - y_normalized) ** 2))

def calculate_cosine(x, y):
    return 1-np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))



def create_heatmap_with_custom_sim(df, calc_func=calculate_rmse, colorscale='Blues', hide_heatmap=False, method='complete'):
    # Transpose data and get labels
    df = df.T
    labels = df.index.tolist()

    n = df.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rmse = calc_func(df.iloc[i, :], df.iloc[j, :])
            dist_matrix[i, j] = rmse
            dist_matrix[j, i] = rmse

    condensed_rmse = squareform(dist_matrix)
    Z = linkage(condensed_rmse, method=method)

    # Create bottom dendrogram
    fig = ff.create_dendrogram(df.values, labels=labels, orientation='bottom', linkagefun=lambda _: Z)
    fig.for_each_trace(lambda trace: trace.update(visible=False))

    # Create side dendrogram
    dendro_side = ff.create_dendrogram(df.values, orientation='right', linkagefun=lambda _: Z)
    if not hide_heatmap:
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'
    
    # Add side dendrogram data to the figure
    for data in dendro_side['data']:
        fig.add_trace(data)
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))
    
    if not hide_heatmap:
        # Create heatmap
        heat_data = dist_matrix[dendro_leaves, :]
        heat_data = heat_data[:, dendro_leaves]

        heatmap = [
            go.Heatmap(
                x=dendro_leaves,
                y=dendro_leaves,
                z=heat_data,
                reversescale=True,
                colorscale=colorscale,
                colorbar=dict(
                    x=1.2,
                    xpad=10
                ),
                hovertemplate='x: %{x}<br>y: %{y}<br>similarity: %{z:.3f}<extra></extra>'
            )
        ]

        heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
        heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

        # Add heatmap data to the figure
        for data in heatmap:
            fig.add_trace(data)

        fig.update_layout(xaxis={'domain': [.15, 1],
                                'mirror': False,
                                'showgrid': False,
                                'showline': False,
                                'zeroline': False,
                                'side': 'top',  # Ustawienie etykiet osi X na górze
                                'tickvals': fig['layout']['xaxis']['tickvals'],
                                'ticktext': [labels[i] + '_reprint' for i in dendro_leaves]
                                })

        fig.update_layout(xaxis2={'domain': [0, .15],
                                'mirror': False,
                                'showgrid': False,
                                'showline': False,
                                'zeroline': False,
                                'showticklabels': False,
                                })

    fig.update_layout({'width': 700, 'height': 700,
                       'showlegend': False, 'hovermode': 'closest',
                       })
    fig.update_layout(yaxis={'domain': [0, 1],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': True,
                             'tickvals': dendro_side['layout']['yaxis']['tickvals'],
                             'ticktext': [labels[i] + '_reprint' for i in dendro_leaves],
                             'side': 'right',
                             })

    fig.update_layout(yaxis2={'domain': [.825, .975],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': True,  
                              'ticks': "",
                              })

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")

    fig.update_layout(
        font=dict(size=8)
    )

    return fig