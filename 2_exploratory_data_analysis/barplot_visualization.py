import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


def plotly_bar_grid(df, columns):
    """
    Create bar plots for each categorical column in the dataframe using Plotly.
    Each bar in a column will have a distinct color using a discrete palette.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - columns (list or str): List of column names or a single column name to plot.
    
    Returns:
    - fig (go.Figure): The Plotly figure object with the bar plots.
    """
    # Ensure 'columns' is a list
    columns = columns if isinstance(columns, list) else [columns]

    # Calculate the number of rows and columns for the subplots
    ncols = min(4, len(columns))  # Maximum 4 columns per row
    nrows = math.ceil(len(columns) / ncols)

    # Create a subplot figure
    fig = make_subplots(
        rows=nrows, cols=ncols, 
        subplot_titles=columns,  # Titles for each subplot
        # horizontal_spacing=0.1, vertical_spacing=0.05  # Adjust spacing
    )

    # Access Plotly's discrete color sequence
    color_palette = px.colors.qualitative.Plotly  # You can choose other palettes

    # Iterate over the columns and add bar plots to the subplots
    for idx, column in enumerate(columns):
        row = (idx // ncols) + 1
        col = (idx % ncols) + 1

        # Get value counts for the current column
        value_counts = df[column].value_counts()
        categories = value_counts.index

        # Map each category to a color from the palette
        colors = [color_palette[i % len(color_palette)] for i in range(len(categories))]

        # Create a bar trace with discrete colors
        fig.add_trace(
            go.Bar(
                x=categories, 
                y=value_counts.values, 
                marker=dict(color=colors),  # Assign discrete colors
                name=column
            ), 
            row=row, col=col
        )

    # Update layout for better visualization
    fig.update_layout(
        height=300 * nrows,  # Height scales with rows
        width=100 + 250 * ncols,  # Width scales with columns
        showlegend=False,  # Disable legend (subplot titles are enough)
        title_text="Categorical Features Bar Plots",  # Main title
        title_x=0.5  # Center the title
    )

    return fig


def plotly_bar_dropdown(df: pd.DataFrame, columns: list):
    """
    Creates an interactive Plotly figure with a dropdown menu to display
    a bar plot for each specified categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of categorical column names to plot.
    """
    # 1. Create a single-plot figure
    fig = go.Figure()

    # Access Plotly's discrete color sequence
    color_palette = px.colors.qualitative.Plotly

    # 2. Add a bar trace for each column, initially hidden
    for col_name in columns:
        value_counts = df[col_name].value_counts()
        categories = value_counts.index
        
        # Create a unique color mapping for each column's categories
        colors = [color_palette[i % len(color_palette)] for i in range(len(categories))]

        fig.add_trace(
            go.Bar(
                x=categories, 
                y=value_counts.values,
                name=col_name,
                marker=dict(color=colors),
                visible=(col_name == columns[0]) # Only the first is visible
            )
        )

    # 3. Create the dropdown menu
    buttons = []
    for i, col_name in enumerate(columns):
        # Create a visibility mask. Each column has 1 trace.
        visibility_mask = [False] * len(columns)
        visibility_mask[i] = True
        
        button = dict(
            label=col_name,
            method="update",
            args=[
                {"visible": visibility_mask},
                {"title.text": f"<b>Distribution of {col_name}</b>"}
            ]
        )
        buttons.append(button)

    # 4. Update the figure layout with the dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.25,
                yanchor="top"
            )
        ],
        title_text=f"Distribution of<br><b>{columns[0]}</b>",
        title_x=0.5,
        height=500,
        width=800,
        showlegend=False,
        yaxis_title="Count"
    )
    
    return fig
