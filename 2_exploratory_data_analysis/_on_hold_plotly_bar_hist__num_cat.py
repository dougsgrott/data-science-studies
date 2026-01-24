# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from math import ceil
from typing import List, Dict, Tuple, Any


def plotly_bar_hist_features(data, categorical_columns, numerical_columns):
    """
    Displays a bar plot of category counts and histograms for numerical features,
    with one subplot dedicated to each numerical feature. A dropdown menu allows
    selecting the categorical column used for grouping (hue).
    """
    # 1. Create a figure with subplots
    K = len(numerical_columns)
    total_plots = K + 1
    ncols = min(4, total_plots)
    nrows = ceil(total_plots / ncols)

    # Update titles for histograms
    subplot_titles = ["Category Counts"] + [f'Histogram of<br>{col}' for col in numerical_columns]
    
    horizontal_spacing = 0.03
    vertical_spacing = 0.08 # Increased slightly for better title spacing

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )
    
    # 2. Add all traces for all categorical options upfront
    trace_indices_by_category = {cat_col: [] for cat_col in categorical_columns}

    for cat_col_index, cat_col in enumerate(categorical_columns):
        sorted_values = sorted(data[cat_col].unique())
        color_palette = px.colors.qualitative.Plotly
        color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(sorted_values)}

        # --- Add Bar Plot Trace ---
        counts = data[cat_col].value_counts().reindex(sorted_values)
        fig.add_trace(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[color_map[val] for val in counts.index],
            name='Count',
            showlegend=False,
            visible=(cat_col_index == 0)
        ), row=1, col=1)
        trace_indices_by_category[cat_col].append(len(fig.data) - 1)

        # --- Add Histogram Traces ---
        subplot_idxs = [(r + 1, c + 1) for r in range(nrows) for c in range(ncols)][1:]
        
        for i, num_col in enumerate(numerical_columns):
            row, col = subplot_idxs[i]
            for val in sorted_values:
                filtered_data = data[data[cat_col] == val]
                if not filtered_data.empty:
                    # REPLACED go.Box with go.Histogram
                    fig.add_trace(go.Histogram(
                        x=filtered_data[num_col], # Use x for histogram data
                        name=str(val),
                        marker_color=color_map[val],
                        legendgroup=str(val),
                        showlegend=(i == 0), # Show legend only for the first set of histograms
                        visible=(cat_col_index == 0)
                    ), row=row, col=col)
                    trace_indices_by_category[cat_col].append(len(fig.data) - 1)

    # 3. Create the dropdown menu
    buttons = []
    for cat_col in categorical_columns:
        visibility = [i in trace_indices_by_category[cat_col] for i in range(len(fig.data))]
        buttons.append(dict(
            label=cat_col,
            method="update",
            args=[
                {"visible": visibility},
                # Update title text for histograms
                {"title": f"Barplot and Histograms for <b>{cat_col}</b>"}
            ]
        ))
    
    # Calculate a consistent figure size
    figure_height = ceil(total_plots / ncols) * 300 + 150
    figure_width = ncols * 400
    
    fig.update_annotations(font_size=14)
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.08,
            yanchor="bottom"
        )],
        height=figure_height,
        width=figure_width,
        # Update main title text
        title_text=f"Barplot and Histograms for <b>{categorical_columns[0]}</b>",
        title_x=0.5,
        legend_title_text="Category",
        # Set barmode to 'overlay' to see distributions clearly
        barmode='overlay',
        margin=dict(t=100, b=50, l=50, r=50) 
    )
    # Make histograms semi-transparent for better visibility when overlapping
    fig.update_traces(opacity=0.75, selector=dict(type='histogram'))

    return fig


def plotly_bar_hist_features_double_dropdown(data: pd.DataFrame, categorical_columns: List[str], numerical_columns: List[str]) -> go.Figure:
    """
    Displays a bar plot and an overlapping histogram in a single figure, with a 
    single dropdown to select a combination of categorical and numerical columns for analysis.
    This version pre-generates all traces and toggles visibility to prevent artifacts.

    Args:
        data (pd.DataFrame): The input dataframe.
        categorical_columns (List[str]): A list of categorical column names.
        numerical_columns (List[str]): A list of numerical column names.

    Returns:
        go.Figure: The Plotly figure with a bar plot and a histogram.
    """
    # 1. Create a figure with two subplots side-by-side
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Category Counts", "Distribution of Numerical Feature"],
        horizontal_spacing=0.08,
    )

    # 2. Pre-generate ALL traces for every combination and add them to the figure
    visibility_mapping = []

    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            # --- Prepare data for this combination ---
            sorted_cat_values = sorted(data[cat_col].unique())
            color_palette = px.colors.qualitative.Plotly
            color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(sorted_cat_values)}
            
            # ⭐ KEY CHANGE: Define shared bins for all categories of this numerical feature
            # This ensures that all histograms in the subplot are directly comparable.
            min_val = data[num_col].min()
            max_val = data[num_col].max()
            bin_size = (max_val - min_val) / 20  # Using 20 bins as a default
            shared_bins = dict(start=min_val, end=max_val, size=bin_size)
            
            current_visibility = [False] * len(fig.data)

            # --- Add Bar Plot Trace ---
            counts = data[cat_col].value_counts().reindex(sorted_cat_values)
            fig.add_trace(go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[color_map[val] for val in counts.index],
                name='Count',
                showlegend=False,
                visible=False
            ), row=1, col=1)
            current_visibility.append(True)

            # --- Add Histogram Traces ---
            for val in sorted_cat_values:
                filtered_data = data[data[cat_col] == val]
                if not filtered_data.empty:
                    fig.add_trace(go.Histogram(
                        x=filtered_data[num_col],
                        name=str(val),
                        marker_color=color_map[val],
                        legendgroup=str(val),
                        xbins=shared_bins,  # Apply the shared bins
                        visible=False
                    ), row=1, col=2)
                    current_visibility.append(True)
            
            visibility_mapping.append(current_visibility)

    # Pad all visibility lists to the same length (total number of traces)
    total_traces = len(fig.data)
    for i, vis_list in enumerate(visibility_mapping):
        visibility_mapping[i] = vis_list + [False] * (total_traces - len(vis_list))

    # 3. Create dropdown buttons that toggle trace visibility
    buttons = []
    option_index = 0
    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            buttons.append(dict(
                label=f"{cat_col} vs {num_col}",
                method="update",
                args=[
                    {'visible': visibility_mapping[option_index]},
                    {
                        'title': f"Comparison for <b>{cat_col}</b> and <b>{num_col}</b>",
                        'xaxis.title.text': cat_col,
                        'yaxis.title.text': 'Count',
                        'xaxis2.title.text': num_col,
                        'yaxis2.title.text': 'Frequency'
                    }
                ]
            ))
            option_index += 1

    # 4. Make the first set of traces visible by default
    if fig.data:
      for i, is_visible in enumerate(visibility_mapping[0]):
          fig.data[i].visible = is_visible

    # 5. Finalize figure layout
    initial_cat_col = categorical_columns[0]
    initial_num_col = numerical_columns[0]
    
    fig.update_annotations(font_size=14)
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                active=0,
                buttons=buttons,
                x=0.01, y=1.2, xanchor="left", yanchor="top",
            )
        ],
        height=450,
        width=900,
        title_text=f"Comparison for <b>{initial_cat_col}</b> and <b>{initial_num_col}</b>",
        title_x=0.5,
        legend_title_text="Category",
        barmode='overlay',
        margin=dict(t=120, b=50, l=50, r=50),
        xaxis_title_text=initial_cat_col,
        yaxis_title_text='Count',
        xaxis2_title_text=initial_num_col,
        yaxis2_title_text='Frequency'
    )
    
    fig.update_traces(opacity=0.75, selector=dict(type='histogram'))

    return fig


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

    # fig_bar_box = plotly_bar_hist_features(df, small_categorical_features, numerical_features)
    fig_bar_box = plotly_bar_hist_features_double_dropdown(df, small_categorical_features, numerical_features)

    fig_bar_box.show()
