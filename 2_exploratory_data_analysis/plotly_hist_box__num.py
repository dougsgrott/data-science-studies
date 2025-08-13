# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio
from itertools import product


def sample_outliers(data: pd.Series, max_samples: int):
    """
    Selects a random sample of outliers if their count exceeds max_samples.
    """
    if len(data) <= max_samples:
        return data
    return data.sample(n=max_samples, random_state=42)


def plotly_histogram_plus_boxplot(df: pd.DataFrame, 
                                     numerical_columns: list, 
                                     categorical_columns: list = None, 
                                     max_outliers: int = 500):
    """
    Creates a highly optimized, interactive Plotly figure with a dropdown menu.

    - If `categorical_columns` is None, it displays the distribution for each numerical column.
    - If `categorical_columns` is provided, it displays the distribution of each numerical
      column, grouped by each categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_columns (list): A list of numerical column names to plot.
        categorical_columns (list, optional): A list of categorical columns for grouping.
        max_outliers (int): The maximum number of outlier points to display.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.02, row_heights=[0.8, 0.2]
    )

    if categorical_columns is None:
        # --- Original functionality (no grouping) ---
        dropdown_options = numerical_columns
    else:
        # --- New functionality (grouping by category) ---
        dropdown_options = list(product(numerical_columns, categorical_columns))
        
    # A dictionary to map each dropdown option to its trace indices
    trace_map = {str(option): [] for option in dropdown_options}

    # --- Add all traces up front ---
    for option_idx, option in enumerate(dropdown_options):
        is_visible = (option_idx == 0)
        
        if categorical_columns is None:
            # --- Handle simple case (no categories) ---
            num_col = option
            col_data = df[num_col].dropna()
            
            base_color = '#636EFA'
            outlier_color = 'rgba(99, 110, 250, 0.5)'

            counts, bin_edges = np.histogram(col_data, bins='auto')
            fig.add_trace(go.Bar(x=(bin_edges[:-1] + bin_edges[1:])/2, y=counts, width=np.diff(bin_edges), name='Histogram', marker_color=base_color, visible=is_visible), row=1, col=1)
            trace_map[str(option)].append(len(fig.data)-1)
            
            q1, median, q3 = col_data.quantile(0.25), col_data.median(), col_data.quantile(0.75)
            iqr = q3 - q1
            lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            fig.add_trace(go.Box(q1=[q1], median=[median], q3=[q3], lowerfence=[lower_fence], upperfence=[upper_fence], name='Boxplot', boxpoints=False, orientation='h', marker_color=base_color, visible=is_visible, y=[0]), row=2, col=1)
            trace_map[str(option)].append(len(fig.data)-1)

            outliers = col_data[(col_data < lower_fence) | (col_data > upper_fence)]
            sampled_outliers = sample_outliers(outliers, max_outliers)
            fig.add_trace(go.Scatter(x=sampled_outliers, y=[0]*len(sampled_outliers), mode='markers', name='Outliers', marker=dict(color=outlier_color, size=5), visible=is_visible), row=2, col=1)
            trace_map[str(option)].append(len(fig.data)-1)

        else:
            # --- Handle categorical grouping case ---
            num_col, cat_col = option
            
            unique_categories = sorted(df[cat_col].dropna().unique())
            colors = px.colors.qualitative.Plotly
            color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}

            overall_min, overall_max = df[num_col].min(), df[num_col].max()
            bin_edges = np.histogram_bin_edges(df[num_col].dropna(), bins='auto', range=(overall_min, overall_max))

            for category in unique_categories:
                cat_data = df[df[cat_col] == category][num_col].dropna()
                if cat_data.empty:
                    continue

                # Overlaid Histogram with opacity
                counts, _ = np.histogram(cat_data, bins=bin_edges)
                fig.add_trace(go.Bar(x=(bin_edges[:-1] + bin_edges[1:])/2, y=counts, name=str(category), legendgroup=str(category), marker_color=color_map[category], opacity=0.7, visible=is_visible), row=1, col=1)
                trace_map[str(option)].append(len(fig.data)-1)

                # Grouped Boxplot (linked to legend group, but not creating a new legend item)
                q1, median, q3 = cat_data.quantile(0.25), cat_data.median(), cat_data.quantile(0.75)
                iqr = q3 - q1
                lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                fig.add_trace(go.Box(q1=[q1], median=[median], q3=[q3], lowerfence=[lower_fence], upperfence=[upper_fence], name=str(category), legendgroup=str(category), showlegend=False, boxpoints=False, orientation='h', marker_color=color_map[category], visible=is_visible, y=[str(category)]), row=2, col=1)
                trace_map[str(option)].append(len(fig.data)-1)

                # Grouped Outliers (no legend item)
                outliers = cat_data[(cat_data < lower_fence) | (cat_data > upper_fence)]
                sampled_outliers = sample_outliers(outliers, max_outliers)
                fig.add_trace(go.Scatter(x=sampled_outliers, y=[str(category)]*len(sampled_outliers), mode='markers', name=str(category), showlegend=False, legendgroup=str(category), marker=dict(color=color_map[category], opacity=0.6, size=5), visible=is_visible), row=2, col=1)
                trace_map[str(option)].append(len(fig.data)-1)

    # --- Create the dropdown menu ---
    buttons = []
    for option in dropdown_options:
        visibility_mask = [False] * len(fig.data)
        for trace_idx in trace_map[str(option)]:
            visibility_mask[trace_idx] = True
        
        if categorical_columns is None:
            label = option
            title = f"Distribution of<br><b>{option}</b>"
        else:
            num_col, cat_col = option
            label = f"{num_col} by {cat_col}"
            title = f"Distribution of<br><b>{num_col}</b> by<br><b>{cat_col}</b>"

        buttons.append(dict(
            label=label,
            method="update",
            args=[{"visible": visibility_mask}, {"title.text": title}]
        ))

    # --- Update layout ---
    initial_title = buttons[0]['args'][1]['title.text']
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=-0.1, xanchor="left", y=1.15, yanchor="top")],
        title_text=initial_title, title_x=0.5,
        height=600, width=900,
        showlegend=(categorical_columns is not None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0, barmode='overlay', # Set histogram mode to overlay
        yaxis_title="Count", xaxis2_title="Value"
    )
    fig.update_yaxes(showticklabels=True, row=2, col=1)

    return fig


if __name__=='__main__':
    from sklearn.datasets import make_classification
    import random

    # --- Dataset Configuration ---
    n_samples_list = [10, 100, 1000, 10000, 100000]
    for n_samples in n_samples_list:
        n_small_categorical_features = 5
        n_large_categorical_features = 5
        n_numerical_features = 3#+30

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
        pio.write_html(fig_hist_box, file=f'fig_hist_box___num_{n_samples}.html', auto_open=False)

        fig_hist_box = plotly_histogram_plus_boxplot(df, numerical_features, small_categorical_features)
        pio.write_html(fig_hist_box, file=f'fig_hist_box__numcat_{n_samples}.html', auto_open=False)

        # fig_bar_box.show()

# %%