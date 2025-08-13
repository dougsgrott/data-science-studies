# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from math import ceil
from typing import List, Dict, Tuple, Any
import plotly.io as pio


def sample_outliers(data: pd.Series, max_samples: int):
    """
    Selects a random sample of outliers if their count exceeds max_samples.
    """
    if len(data) <= max_samples:
        return data
    return data.sample(n=max_samples, random_state=42)


def plotly_bar_box_features_optimized(data: pd.DataFrame, 
                                      categorical_columns: list, 
                                      numerical_columns: list, 
                                      max_outliers: int = 500):
    """
    Displays an optimized bar plot and multiple boxplots with a dropdown menu.
    
    Optimization is achieved by pre-calculating boxplot statistics and sampling
    outliers to keep the plot lightweight.

    Args:
        data (pd.DataFrame): The input DataFrame.
        categorical_columns (list): A list of categorical columns for the dropdown.
        numerical_columns (list): A list of numerical columns for the boxplots.
        max_outliers (int): The maximum number of outlier points to display per category.
    """
    # 1. Create a figure with subplots
    K = len(numerical_columns)
    total_plots = K + 1
    ncols = min(4, total_plots)
    nrows = ceil(total_plots / ncols)

    subplot_titles = ["Category Counts"] + [f'Boxplot of<br>{col}' for col in numerical_columns]
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=subplot_titles,
        horizontal_spacing=0.03, vertical_spacing=0.1
    )
    
    # 2. Add all traces for all categorical options upfront
    trace_indices_by_category = {cat_col: [] for cat_col in categorical_columns}

    for cat_col_index, cat_col in enumerate(categorical_columns):
        is_visible = (cat_col_index == 0)
        sorted_values = sorted(data[cat_col].dropna().unique())
        color_palette = px.colors.qualitative.Plotly
        color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(sorted_values)}

        # --- Add Bar Plot Trace ---
        counts = data[cat_col].value_counts().reindex(sorted_values)
        fig.add_trace(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[color_map[val] for val in counts.index],
            name='Count', showlegend=False, visible=is_visible
        ), row=1, col=1)
        trace_indices_by_category[cat_col].append(len(fig.data) - 1)

        # --- Add Optimized Box Plot and Outlier Traces ---
        subplot_idxs = [(r + 1, c + 1) for r in range(nrows) for c in range(ncols)][1:]
        
        for i, num_col in enumerate(numerical_columns):
            row, col = subplot_idxs[i]
            for val in sorted_values:
                cat_data = data[data[cat_col] == val][num_col].dropna()
                if cat_data.empty:
                    continue

                # Calculate stats for the boxplot
                q1, median, q3 = cat_data.quantile(0.25), cat_data.median(), cat_data.quantile(0.75)
                iqr = q3 - q1
                lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr

                # Add the main box trace (no raw data)
                fig.add_trace(go.Box(
                    q1=[q1], median=[median], q3=[q3],
                    lowerfence=[lower_fence], upperfence=[upper_fence],
                    x=[str(val)], name=str(val),
                    boxpoints=False, legendgroup=str(val),
                    showlegend=(i == 0), # Show legend only for the first row of boxplots
                    marker_color=color_map[val], visible=is_visible
                ), row=row, col=col)
                trace_indices_by_category[cat_col].append(len(fig.data) - 1)

                # Identify, sample, and plot outliers
                outliers = cat_data[(cat_data < lower_fence) | (cat_data > upper_fence)]
                sampled_outliers = sample_outliers(outliers, max_outliers)
                
                fig.add_trace(go.Scatter(
                    x=[str(val)] * len(sampled_outliers),
                    y=sampled_outliers,
                    mode='markers', name=str(val),
                    legendgroup=str(val), showlegend=False,
                    marker=dict(color=color_map[val], opacity=0.6, size=5),
                    visible=is_visible
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
                {"title.text": f"Barplot and Boxplots for <b>{cat_col}</b>"}
            ]
        ))
    
    figure_height = nrows * 350
    figure_width = ncols * 250 + 100
    
    fig.update_annotations(font_size=12)
    fig.update_layout(
        updatemenus=[dict(
            active=0, buttons=buttons, direction="down",
            pad={"r": 10, "t": 10}, showactive=True,
            x=0.0, xanchor="left", y=1.1, yanchor="top"
        )],
        height=figure_height, width=figure_width,
        title_text=f"Barplot and Boxplots for <b>{categorical_columns[0]}</b>",
        title_x=0.5,
        legend_title_text="Category",
        margin=dict(t=100, b=50, l=50, r=50) 
    )

    return fig




def plotly_bar_box_features_double_dropdown_optimized(data: pd.DataFrame, 
                                                      categorical_columns: List[str], 
                                                      numerical_columns: List[str],
                                                      max_outliers: int = 200) -> go.Figure:
    """
    Displays an optimized bar plot and boxplot with a single dropdown to select
    a combination of categorical and numerical columns.

    Args:
        data (pd.DataFrame): The input dataframe.
        categorical_columns (List[str]): A list of categorical column names.
        numerical_columns (List[str]): A list of numerical column names.
        max_outliers (int): The maximum number of outliers to display per category.

    Returns:
        go.Figure: The Plotly figure with the interactive plots.
    """
    # 1. Create a figure with two subplots side-by-side
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Category Counts", "Distribution of Numerical Feature"],
        horizontal_spacing=0.08,
    )

    # 2. Pre-generate ALL traces for every combination
    trace_map = {} # Maps dropdown label to list of its trace indices

    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            dropdown_label = f"{cat_col} vs {num_col}"
            trace_map[dropdown_label] = []

            # --- Prepare data for this combination ---
            sorted_cat_values = sorted(data[cat_col].dropna().unique())
            color_palette = px.colors.qualitative.Plotly
            color_map = {val: color_palette[i % len(color_palette)] for i, val in enumerate(sorted_cat_values)}
            
            # --- Add Bar Plot Trace ---
            counts = data[cat_col].value_counts().reindex(sorted_cat_values)
            fig.add_trace(go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[color_map[val] for val in counts.index],
                name='Count', showlegend=False, visible=False
            ), row=1, col=1)
            trace_map[dropdown_label].append(len(fig.data) - 1)

            # --- Add Optimized Box Plot and Outlier Traces ---
            for val in sorted_cat_values:
                cat_data = data[data[cat_col] == val][num_col].dropna()
                if cat_data.empty:
                    continue
                
                # Calculate stats
                q1, median, q3 = cat_data.quantile(0.25), cat_data.median(), cat_data.quantile(0.75)
                iqr = q3 - q1
                lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr

                # Add Box trace (no raw data)
                fig.add_trace(go.Box(
                    q1=[q1], median=[median], q3=[q3],
                    lowerfence=[lower_fence], upperfence=[upper_fence],
                    name=str(val),
                    x=[str(val)], # Set x for vertical grouping
                    boxpoints=False, legendgroup=str(val),
                    width=0.4,
                    orientation='v',
                    marker_color=color_map[val], visible=False
                ), row=1, col=2)
                trace_map[dropdown_label].append(len(fig.data) - 1)

                # Add sampled Outlier trace
                outliers = cat_data[(cat_data < lower_fence) | (cat_data > upper_fence)]
                sampled_outliers = sample_outliers(outliers, max_outliers)
                fig.add_trace(go.Scatter(
                    x=[str(val)] * len(sampled_outliers), y=sampled_outliers,
                    mode='markers', name=str(val),
                    legendgroup=str(val), showlegend=False,
                    marker=dict(color=color_map[val], opacity=0.6, size=5),
                    visible=False
                ), row=1, col=2)
                trace_map[dropdown_label].append(len(fig.data) - 1)

    # 3. Create dropdown buttons
    buttons = []
    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            dropdown_label = f"{cat_col} vs {num_col}"
            
            # Create visibility mask for this option
            visibility_mask = [False] * len(fig.data)
            for trace_idx in trace_map[dropdown_label]:
                visibility_mask[trace_idx] = True
            
            buttons.append(dict(
                label=dropdown_label,
                method="update",
                args=[
                    {'visible': visibility_mask},
                    {
                        'title.text': f"Comparison for<br><b>{cat_col}</b> and <b>{num_col}</b>",
                        'xaxis.title.text': cat_col,
                        'yaxis2.title.text': num_col,
                        'xaxis2.title.text': cat_col,
                    }
                ]
            ))

    # 4. Make the first set of traces visible by default
    if buttons:
        first_option_label = buttons[0]['label']
        first_visibility_mask = [False] * len(fig.data)
        for trace_idx in trace_map[first_option_label]:
            first_visibility_mask[trace_idx] = True
        
        for i, is_visible in enumerate(first_visibility_mask):
            fig.data[i].visible = is_visible

    # 5. Finalize figure layout
    initial_cat_col = categorical_columns[0]
    initial_num_col = numerical_columns[0]
    
    fig.update_annotations(font_size=14)
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown", active=0, buttons=buttons,
            x=0.1, y=1.3, xanchor="center", yanchor="top"
        )],
        height=500, width=900,
        boxmode='group',
        title=dict(
            text=f"Comparison for<br><b>{initial_cat_col}</b> and <b>{initial_num_col}</b>",
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        legend_title_text="Category",
        margin=dict(t=120, b=50, l=50, r=50),
        yaxis_title_text='Count',
        xaxis_title_text=initial_cat_col,
        yaxis2_title_text=initial_num_col,
        xaxis2_title_text=initial_cat_col
    )

    return fig




if __name__=='__main__':
    from sklearn.datasets import make_classification
    import random


    # --- Dataset Configuration ---
    # n_samples = 100
    for n_samples in [100, 1000, 10000]:#, 100000]:
        n_small_categorical_features = 5
        n_large_categorical_features = 5
        n_numerical_features = 3+24

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

        # categories = pd.cut(data, bins=5, labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"])

        # fig_bar_box = plotly_bar_box_features_optimized(df, small_categorical_features, numerical_features)


        fig_bar_box = plotly_bar_box_features_optimized(df, small_categorical_features, numerical_features)
        pio.write_html(fig_bar_box, file=f'fig_bar_box_1__{n_samples}.html', auto_open=False, include_plotlyjs=True)

        fig_bar_box = plotly_bar_box_features_double_dropdown_optimized(df, small_categorical_features, numerical_features)
        pio.write_html(fig_bar_box, file=f'fig_bar_box_2__{n_samples}.html', auto_open=False, include_plotlyjs=True)
        
        # fig_bar_box.to_html('n_samples.html')

    # fig_bar_box.show()
