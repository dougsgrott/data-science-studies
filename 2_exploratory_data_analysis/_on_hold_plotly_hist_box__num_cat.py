# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from math import ceil
from typing import List, Dict, Tuple, Any
from itertools import product, permutations
import plotly.io as pio



def create_marginal_dropdown_plot(
    df: pd.DataFrame, 
    numerical_cols: List[str], 
    categorical_cols: List[str], 
    marginal_type: str = 'box'
) -> go.Figure:
    """
    Creates a 2D histogram with marginal plots and a dropdown to select columns.

    The dropdown menu contains all possible combinations where x and y are unique 
    numerical columns and color is a categorical column.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        numerical_cols (List[str]): A list of numerical column names for x and y axes.
        categorical_cols (List[str]): A list of categorical column names for color grouping.
        marginal_type (str, optional): The type of marginal plot. Can be 'box', 
                                       'violin', 'rug', or 'histogram'. Defaults to 'box'.

    Returns:
        go.Figure: A Plotly Figure object with the interactive dropdown plot.
    """
    # --- 1. Validate Inputs and Prepare Combinations ---
    numerical_pairs = list(permutations(numerical_cols, 2))
    
    if not numerical_pairs or not categorical_cols:
        fig = go.Figure()
        fig.update_layout(
            title_text="Insufficient Columns Provided",
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{
                "text": "Please provide at least 2 numerical and 1 categorical column.",
                "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}
            }]
        )
        return fig

    combinations = list(product(numerical_pairs, categorical_cols))

    # --- 2. Initialize Figure and Pre-generate All Traces ---
    (x_initial, y_initial), color_initial = combinations[0]
    fig = px.histogram(df, x=x_initial, y=y_initial, color=color_initial, marginal=marginal_type)
    fig.data = [] # Clear data, keeping only the complex layout structure

    trace_indices = []
    layouts = []

    for i, ((x_col, y_col), color_col) in enumerate(combinations):
        start_index = len(fig.data)
        
        temp_fig = px.histogram(
            df, x=x_col, y=y_col, color=color_col,
            marginal=marginal_type, hover_data=df.columns
        )
        
        for trace in temp_fig.data:
            # Set visibility directly during creation
            trace.visible = (i == 0) 
            fig.add_trace(trace)
        
        end_index = len(fig.data)
        trace_indices.append(list(range(start_index, end_index)))
        
        layouts.append({
            'title.text': f'Distribution of {y_col} vs. {x_col} by {color_col}',
            'xaxis.title.text': temp_fig.layout.xaxis.title.text,
            'yaxis.title.text': temp_fig.layout.yaxis.title.text
        })

    # --- 3. Create Dropdown and Finalize Layout ---
    buttons = []
    for i, ((x_col, y_col), color_col) in enumerate(combinations):
        visibility_mask = [False] * len(fig.data)
        for trace_idx in trace_indices[i]:
            visibility_mask[trace_idx] = True
        
        buttons.append(dict(
            label=f"X:{x_col} | Y:{y_col} | Color:{color_col}",
            method="update",
            args=[{"visible": visibility_mask}, layouts[i]]
        ))
    
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.5, xanchor="center",
            y=1.18, yanchor="top"
        )],
        title=layouts[0]['title.text']
    )

    return fig


if __name__=='__main__':
    from sklearn.datasets import make_classification
    import random

    # n_samples_list = [10, 100, 1000, 10000, 100000]
    n_samples_list = [1000]
    for n_samples in n_samples_list:
        # --- Dataset Configuration ---
        n_small_categorical_features = 5
        n_large_categorical_features = 5
        n_numerical_features = 3+3

        n_categorical_features = n_small_categorical_features + n_large_categorical_features
        n_features = n_numerical_features + n_categorical_features


        # --- Dataset Creation ---
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
        y = y.reshape(-1, 1)
        df = pd.DataFrame(np.concatenate([X, y], axis=1))


        # --- Dataset Processing ---
        numerical_columns = [f'num_feature_{i}' for i in range(n_numerical_features)]
        categorical_columns = [f'cat_feature_{i}' for i in range(n_categorical_features)]
        df.columns = numerical_columns + categorical_columns + ['target']

        # --- Small Categorical Features ---
        for i in range(n_small_categorical_features):
            col = numerical_columns[i % n_numerical_features]
            n_bins = random.randint(2, 5)
            df[f'cat_feature_{i}_small'] = pd.cut(df[col], bins=n_bins, labels=[f'Small_{j}' for j in range(1, n_bins + 1)])

        # --- Large Categorical Features ---
        for i in range(n_small_categorical_features, n_small_categorical_features + n_large_categorical_features):
            col = numerical_columns[i % n_numerical_features]
            n_bins = random.randint(20, 50)
            df[f'cat_feature_{i}_large'] = pd.cut(df[col], bins=n_bins, labels=[f'Large_{j}' for j in range(1, n_bins + 1)])

        numerical_features = [c for c in df.columns if c.startswith('num')]
        small_categorical_features = [c for c in df.columns if c.endswith('_small')]
        large_categorical_features = [c for c in df.columns if c.endswith('_large')]

        # display(df)
        # display(pd.DataFrame(df))

        # categories = pd.cut(data, bins=5, labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"])

        fig_hist_box = create_marginal_dropdown_plot(df, numerical_features, small_categorical_features)
        # pio.write_html(fig_hist_box, file=f'fig_hist_box__{n_samples}.html', auto_open=False)

        fig_hist_box.show()
