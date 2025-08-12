# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio


def sample_outliers(data: pd.Series, max_samples: int):
    """
    Selects a random sample of outliers if their count exceeds max_samples.
    """
    if len(data) <= max_samples:
        return data
    return data.sample(n=max_samples, random_state=42)


def plotly_histogram_plus_boxplot(df: pd.DataFrame, columns: list, max_outliers: int = 50):
    """
    Creates a highly optimized, interactive Plotly figure with a dropdown menu 
    to display a histogram and a horizontal boxplot for each specified column.
    
    Optimization:
    1. Pre-aggregates histogram data to avoid embedding raw data.
    2. Manually calculates boxplot statistics and samples outliers to prevent
       plotting an excessive number of points.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of numerical column names to plot.
        max_outliers (int): The maximum number of outlier points to display.
    """
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.8, 0.2]
    )

    # Define a consistent color scheme
    base_color = '#636EFA' # A standard Plotly blue
    outlier_color = 'rgba(99, 110, 250, 0.5)' # Same color with 50% transparency

    # Add traces for each column
    for col_name in columns:
        is_visible = (col_name == columns[0])
        col_data = df[col_name].dropna()

        # --- Pre-aggregate histogram data ---
        counts, bin_edges = np.histogram(col_data, bins='auto')
        fig.add_trace(
            go.Bar(
                x=(bin_edges[:-1] + bin_edges[1:]) / 2,
                y=counts,
                width=np.diff(bin_edges),
                name='Histogram',
                visible=is_visible,
                marker_color=base_color,
            ),
            row=1, col=1
        )
        
        # --- Manually calculate Boxplot and sample outliers ---
        q1 = col_data.quantile(0.25)
        median = col_data.median()
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # Add the main boxplot trace using calculated values
        fig.add_trace(
            go.Box(
                q1=[q1], median=[median], q3=[q3],
                lowerfence=[lower_fence], upperfence=[upper_fence],
                name='Boxplot',
                boxpoints=False,
                orientation='h',
                visible=is_visible,
                marker_color=base_color,
                y=[0]
            ), 
            row=2, col=1
        )

        # Identify, sample, and plot the outliers
        outliers = col_data[(col_data < lower_fence) | (col_data > upper_fence)]
        sampled_outliers = sample_outliers(outliers, max_outliers)
        fig.add_trace(
            go.Scatter(
                x=sampled_outliers,
                y=[0] * len(sampled_outliers),
                mode='markers',
                name='Outliers',
                marker=dict(color=outlier_color, size=5),
                visible=is_visible
            ),
            row=2, col=1
        )

    # --- Create the dropdown menu ---
    buttons = []
    for i, col_name in enumerate(columns):
        # Each column now has 3 traces: histogram, box, and outliers
        visibility_mask = [False] * (len(columns) * 3)
        visibility_mask[i*3] = True
        visibility_mask[i*3 + 1] = True
        visibility_mask[i*3 + 2] = True
        
        button = dict(
            label=col_name,
            method="update",
            args=[
                {"visible": visibility_mask},
                {"title.text": f"<b>Distribution of {col_name}</b>"}
            ]
        )
        buttons.append(button)

    # Update the figure layout
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0, xanchor="left",
            y=1.15, yanchor="top"
        )],
        title_text=f"<b>Distribution of {columns[0]}</b>",
        title_x=0.5,
        height=600,
        width=800,
        showlegend=False,
        bargap=0, # Remove gap between bars for histogram look
        yaxis_title="Count",
        xaxis2_title="Value"
    )
    
    # Hide the y-axis ticks and labels for the boxplot for a cleaner look
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    return fig


if __name__=='__main__':
    from sklearn.datasets import make_classification
    import random

    # --- Dataset Configuration ---
    n_samples_list = [10, 100, 1000, 10000, 100000]
    for n_samples in n_samples_list:
        n_small_categorical_features = 5
        n_large_categorical_features = 5
        n_numerical_features = 3+30

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

        fig_hist_box = plotly_histogram_plus_boxplot(df, numerical_features)
        pio.write_html(fig_hist_box, file=f'fig_hist_box__{n_samples}.html', auto_open=False)
        # fig_bar_box.show()

# %%