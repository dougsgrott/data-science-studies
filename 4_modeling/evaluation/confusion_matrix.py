import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.metrics import confusion_matrix

from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib
import seaborn as sns
import matplotlib.colors as mcolors



# --- Factory ---
def create_confusion_matrix(dataset_dict, classifier=None, **kwargs):
    output_type = kwargs.pop('output_type', True)
    fmt_table = kwargs.pop('fmt_table', False)
    if output_type == 'image':
        max_cols = kwargs.pop('max_cols', 3)
        return plot_confusion_matrix(dataset_dict, max_cols)
    if output_type == 'plotly_table':
        return create_confusion_table_plotly(dataset_dict)
    if fmt_table:
        return create_fmt_confusion_table(dataset_dict)
    return create_confusion_table(dataset_dict)


# --- Pandas DataFrame ---
def create_confusion_table(dataset_dict, **kwargs):
    index = ['Actual Negative', 'Actual Positive']
    df_list = []

    for set_name in dataset_dict:
        y_ = dataset_dict[set_name]['y']
        y_pred = dataset_dict[set_name]['y_pred']

        cm = confusion_matrix(y_, y_pred)
        total = cm.sum()

        # Extract counts from the confusion matrix
        tn, fp, fn, tp = cm.ravel()

        # Format with percentages
        data = {
            'Predicted Negative': [f'{tn} ({tn/total:.2%})', f'{fn} ({fn/total:.2%})'],
            'Predicted Positive': [f'{fp} ({fp/total:.2%})', f'{tp} ({tp/total:.2%})'],
            'Dataset Label': [set_name, set_name],  # One row per actual class
        }

        df_list.append(pd.DataFrame(data, index=index))

    # Combine all datasets' tables
    df = pd.concat(df_list)
    return df


# --- Stylized HTML ---
def create_fmt_confusion_table(dataset_dict, **kwargs):
    # Define the index and columns
    index = ['Actual Negative', 'Actual Positive']
    columns = ['Predicted Negative', 'Predicted Positive']

    formatted_dfs = []
    # Iterate over each dataset in the dictionary
    for set_name, data in dataset_dict.items():
        y_ = data['y']
        y_pred = data['y_pred']

        # Compute the confusion matrix
        cm = confusion_matrix(y_, y_pred)
        total = cm.sum()

        # Create the DataFrame for the counts
        df_counts = pd.DataFrame(cm, index=index, columns=columns)
        df_aux_strings = pd.DataFrame([['True Negative<br>', 'False Positive<br>'], ['False Negative<br>', 'True Positive<br>']], index=index, columns=columns)

        # Create the DataFrame for the percentages
        df_percent = df_counts / total
        df_formatted = df_aux_strings + df_counts.astype(str) + ' (' + (df_percent * 100).round(2).astype(str) + '%)'

        # Add Dataset Label and reset the index
        df_formatted['Dataset Label'] = set_name
        df_formatted = df_formatted.reset_index().rename(columns={'index': 'Actual'})

        # Also create a percentage-only DataFrame for styling
        df_percent = df_percent.reset_index().rename(columns={'index': 'Actual'})

        # Append both formatted count & percentage DataFrames for styling
        formatted_dfs.append((df_formatted, df_percent, set_name))

    # Concatenate all formatted DataFrames (counts and percentages)
    combined_formatted = pd.concat([t[0] for t in formatted_dfs], ignore_index=True)
    combined_percent = pd.concat([t[1] for t in formatted_dfs], ignore_index=True)

    def style_per_row(row):
        # Get the set name for current row
        set_name = row['Dataset Label']
        cols = columns
        idx = row.name
        pct = combined_percent.loc[idx, cols]

        # Mask to get the subset of rows belonging to the current dataset
        dataset_mask = combined_formatted['Dataset Label'] == set_name
        dataset_percents = combined_percent.loc[dataset_mask, cols]
        vmin = dataset_percents.min().min()
        vmax = dataset_percents.max().max()

        # Normalize the values to apply a color gradient
        norm_values = pct * 0
        if vmax > vmin:
            norm_values = (pct - vmin) / (vmax - vmin)

        # Use a colormap for styling (Blues)
        cmap = matplotlib.colormaps['Blues']
        colors = [rgb2hex(cmap(v)) for v in norm_values]

        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        def brightness(rgb):
            # Perceived brightness formula (YIQ)
            r, g, b = rgb
            return (r * 299 + g * 587 + b * 114) / 1000

        # Define the cell styling
        style = {}
        for col, color in zip(cols, colors):
            rgb = hex_to_rgb(color)
            bright = brightness(rgb)
            font_color = 'black' if bright > 128 else 'white'
            style[col] = f'background-color: {color}; color: {font_color}'

        # Reset the 'Dataset Label' and 'Actual' columns (no styling needed)
        style['Dataset Label'] = ''
        style['Actual'] = ''
        return pd.Series(style)

    # Apply the styling function to the combined DataFrame
    styled = combined_formatted.style.apply(style_per_row, axis=1)
    return styled

# --- Matplotlib Image ---
def plot_confusion_matrix(dataset_dict, max_cols=3, **kwargs):
    names = ['True Neg','False Pos','False Neg','True Pos']

    num_datasets = len(dataset_dict)
    num_rows = (num_datasets + max_cols - 1) // max_cols
    fig_width = 4*max_cols
    fig_height = 3*num_rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=max_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    for i, (name, data_dict) in enumerate(dataset_dict.items()):
        y = data_dict['y']
        y_pred = data_dict['y_pred']
        cm = confusion_matrix(y, y_pred)
        counts = [value for value in cm.flatten()]
        percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
        labels = np.asarray(labels).reshape(2,2)
        ax = axes[i]
        sns.heatmap(cm, annot=labels, cmap='Blues', fmt='', ax=ax)
        ax.set_title(name)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.close()
    return fig


# --- Plotly Table ---
def get_contrasting_font_color(hex_color):
    r, g, b = mcolors.to_rgb(hex_color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.5 else 'white'


def create_confusion_table_plotly(dataset_dict, **kwargs):
    index = ['Actual Negative', 'Actual Positive']
    dataset_figures = []
    buttons = []

    for i, (set_name, data) in enumerate(dataset_dict.items()):
        y_ = data['y']
        y_pred = data['y_pred']
        conf_matrix = confusion_matrix(y_, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        total = conf_matrix.sum()

        pct_tn = tn / total
        pct_fp = fp / total
        pct_fn = fn / total
        pct_tp = tp / total

        # Table data
        df = pd.DataFrame([
            {
                'Dataset Label': set_name,
                'Actual': 'Actual Negative',
                'Predicted Negative': f'True Negative<br>{tn} ({pct_tn:.2%})',
                'Predicted Positive': f'False Positive<br>{fp} ({pct_fp:.2%})',
            },
            {
                'Dataset Label': set_name,
                'Actual': 'Actual Positive',
                'Predicted Negative': f'False Negative<br>{fn} ({pct_fn:.2%})',
                'Predicted Positive': f'True Positive<br>{tp} ({pct_tp:.2%})',
            },
        ])

        # Colors
        neg_percents = [pct_tn, pct_fn]
        pos_percents = [pct_fp, pct_tp]
        all_percents = np.array(neg_percents + pos_percents)
        norm = mcolors.Normalize(vmin=all_percents.min(), vmax=all_percents.max())
        cmap = matplotlib.colormaps['Blues']
        neg_colors = [mcolors.to_hex(cmap(norm(p))) for p in neg_percents]
        pos_colors = [mcolors.to_hex(cmap(norm(p))) for p in pos_percents]
        neg_font_colors = [get_contrasting_font_color(c) for c in neg_colors]
        pos_font_colors = [get_contrasting_font_color(c) for c in pos_colors]

        fill_colors = [
            ['white'] * 2,  # Dataset Label
            ['white'] * 2,  # Actual
            neg_colors,
            pos_colors,
        ]
        font_colors = [
            ['black'] * 2,  # Dataset Label
            ['black'] * 2,  # Actual
            neg_font_colors,
            pos_font_colors,
        ]

        table = go.Table(
            header=dict(
                values=["Dataset Label", "Actual", "Predicted Negative", "Predicted Positive"],
                fill_color='lightgrey',
                font=dict(color='black'),
                align='left'
            ),
            cells=dict(
                values=[
                    df['Dataset Label'],
                    df['Actual'],
                    df['Predicted Negative'],
                    df['Predicted Positive'],
                ],
                fill_color=fill_colors,
                font=dict(color=font_colors),
                align='left'
            ),
            visible=(i == 0)  # Only first visible initially
        )

        dataset_figures.append(table)

        # Dropdown button
        buttons.append(dict(
            label=set_name,
            method="update",
            args=[
                {"visible": [j == i for j in range(len(dataset_dict))]},
                {"title": f"Confusion Matrix<br>Dataset: {set_name}"}
            ]
        ))

    # Create figure with all table traces
    fig = go.Figure(data=dataset_figures)

    # Add dropdown to layout
    fig.update_layout(
        title=f"Confusion Matrix<br>Dataset: {list(dataset_dict.keys())[0]}",
        title_x=0.5,
        width=800,
        height=330,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.0,
            "xanchor": "left",
            "y": 1.45,
            "yanchor": "top"
        }]
    )

    return fig

