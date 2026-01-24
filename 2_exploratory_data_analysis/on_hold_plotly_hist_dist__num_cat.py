# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from math import ceil
from typing import List, Dict, Tuple, Any
import plotly.figure_factory as ff
import plotly.graph_objects as go
from itertools import product



def create_interactive_distplot(
    df: pd.DataFrame, 
    numerical_cols: List[str], 
    categorical_cols: List[str]
) -> go.Figure:
    """
    Creates an interactive density plot with a dropdown to select columns.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        numerical_cols (List[str]): A list of numerical column names to plot.
        categorical_cols (List[str]): A list of categorical column names for grouping.

    Returns:
        go.Figure: A Plotly Figure object with the interactive dropdown plot.
    """
    # --- 1. Validate Inputs and Prepare Combinations ---
    if not numerical_cols or not categorical_cols:
        # Return an empty, informative figure if inputs are insufficient
        fig = go.Figure()
        fig.update_layout(
            title_text="Insufficient Columns Provided",
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{
                "text": "Please provide at least 1 numerical and 1 categorical column.",
                "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}
            }]
        )
        return fig

    combinations = list(product(numerical_cols, categorical_cols))

    # --- 2. Initialize Figure with Correct Layout ---
    # ⭐ KEY FIX: Create a template figure from the first combination to get the correct layout structure.
    # This is essential for the y-axis annotations and range needed by distplots.
    
    # Find the first combination that yields valid data
    (initial_num, initial_cat) = combinations[0]
    initial_hist_data = []
    initial_group_labels = []
    for group in sorted(df[initial_cat].dropna().unique()):
        data_subset = df[df[initial_cat] == group][initial_num]
        if len(data_subset) >= 2:
            initial_hist_data.append(data_subset)
            initial_group_labels.append(str(group))
    
    # Initialize the main figure using this first valid set of data
    main_fig = ff.create_distplot(initial_hist_data, initial_group_labels, show_hist=False, show_rug=False)
    # Now, clear the data traces, keeping the essential layout
    main_fig.data = []

    # --- 3. Generate and Add All Traces to the Structured Figure ---
    trace_indices = []
    layouts = []

    for i, (num_col, cat_col) in enumerate(combinations):
        start_index = len(main_fig.data)
        
        # Prepare data, filtering for groups large enough for KDE
        hist_data_filtered = []
        group_labels_filtered = []
        for group in sorted(df[cat_col].dropna().unique()):
            group_data = df[df[cat_col] == group][num_col]
            if len(group_data) >= 2:
                hist_data_filtered.append(group_data)
                group_labels_filtered.append(str(group))
        
        # Only add traces if there is valid data
        if hist_data_filtered:
            max_ = max([data.max() for data in hist_data_filtered])
            min_ = min([data.min() for data in hist_data_filtered])
            bin_size = (max_ - min_) / 30
            temp_fig = ff.create_distplot(hist_data_filtered, group_labels_filtered, show_hist=True, show_rug=False, bin_size=bin_size)
            for trace in temp_fig.data:
                # Set visibility for the initial view
                trace.visible = (i == 0)
                main_fig.add_trace(trace)
        
        end_index = len(main_fig.data)
        trace_indices.append(list(range(start_index, end_index)))
        
        layouts.append({
            'title.text': f'Density Plot of <b>{num_col}</b> by <b>{cat_col}</b>',
            'xaxis.title.text': num_col,
            'annotations': temp_fig.layout.annotations if hist_data_filtered else []
        })

    # --- 4. Create Dropdown and Finalize Layout ---
    buttons = []
    for i, (num_col, cat_col) in enumerate(combinations):
        visibility_mask = [False] * len(main_fig.data)
        for trace_idx in trace_indices[i]:
            visibility_mask[trace_idx] = True
        
        buttons.append(dict(
            label=f"Plot: {num_col} | Group by: {cat_col}",
            method="update",
            args=[{"visible": visibility_mask}, layouts[i]]
        ))
    
    # Update the layout with the dropdown menu and initial titles/annotations
    main_fig.update_layout(
        updatemenus=[dict(
            active=0, buttons=buttons, direction="down",
            pad={"r": 10, "t": 10}, showactive=True,
            x=0.5, xanchor="center", y=1.15, yanchor="top"
        )],
        yaxis_title="Density",
        title=layouts[0]['title.text'],
        xaxis_title=layouts[0]['xaxis.title.text'],
        annotations=layouts[0]['annotations']
    )

    return main_fig


if __name__=='__main__':
    from sklearn.datasets import make_classification
    import random

    # --- Dataset Configuration ---
    n_small_categorical_features = 5
    n_large_categorical_features = 5
    n_numerical_features = 3+30

    n_categorical_features = n_small_categorical_features + n_large_categorical_features
    n_features = n_numerical_features + n_categorical_features


    # --- Dataset Creation ---
    X, y = make_classification(n_samples=100, n_features=n_features, random_state=42)
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

    fig_bar_box = create_interactive_distplot(df, numerical_features, small_categorical_features)
    fig_bar_box.show()


# if __name__ == '__main__':
#     # Use the tips dataset for a reliable demonstration
#     tips_df = px.data.tips()

#     numerical_features = ['total_bill', 'tip']
#     categorical_features = ['sex', 'smoker', 'day', 'time']

#     fig = create_interactive_distplot(
#         df=tips_df,
#         numerical_cols=numerical_features,
#         categorical_cols=categorical_features
#     )

#     fig.show()