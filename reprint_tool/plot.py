import plotly.graph_objects as go


def create_main_dashboard(df, signature, title, yaxis_title):
    frequencies = df[signature] * 1

    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    bases = ['A', 'C', 'G', 'T']
    contexts = [f'{x}[{m}]{y}' for m in mutations for x in bases for y in bases]

    colors = {
        'C>A': 'blue',
        'C>G': 'black',
        'C>T': 'red',
        'T>A': 'gray',
        'T>C': 'green',
        'T>G': 'pink'
    }

    fig = go.Figure()
    
    for mutation in mutations:
        mutation_contexts = [c for c in contexts if f'[{mutation}]' in c]
        mutation_frequencies = [frequencies[mc] if mc in frequencies.index else 0 for mc in mutation_contexts]
        
        fig.add_trace(go.Bar(
            x=mutation_contexts,
            y=mutation_frequencies,
            name=mutation,
            marker_color=colors[mutation]
        ))

    y_max = frequencies.max()

    fig.update_layout(
        title=title,
        xaxis_title='Mutation Context',
        yaxis_title=yaxis_title,
        xaxis_tickangle=-90,
        template='plotly_white',
        barmode='group',
        legend_title='Mutation Type',
        yaxis_range=[0, y_max], 
        margin=dict(l=50, r=50, t=50, b=150),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=10))
    )

    return fig

import plotly.io as pio

def save_all_signatures_to_pdf(df, output_dir=".", prefix="reprint_", yaxis_title="Probabilities"):
    """
    Creates and saves PDF plots for all signatures in the DataFrame.
    Each signature is saved as a separate PDF file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    for signature in df.columns:
        fig = create_main_dashboard(
            df,
            signature=signature,
            title=f"{prefix}{signature} - Probabilities of Specific Tri-nucleotide Context Mutations by Mutation Type",
            yaxis_title=yaxis_title
        )
        pdf_path = os.path.join(output_dir, f"{prefix}{signature}.pdf")
        fig.write_image(pdf_path, format="pdf")
        print(f"Saved: {pdf_path}")
