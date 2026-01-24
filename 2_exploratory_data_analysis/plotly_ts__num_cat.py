# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plotly_timeseries(df: pd.DataFrame, 
                         y: str,
                         x: str,
                         group_col: str = None,
                         title: str = None,
                         markers: bool = True,
                         line_color: str = None):
    fig = go.Figure()
    
    mode = 'lines+markers' if markers else 'lines'

    if group_col:
        # --- Grouped Time Series ---
        unique_groups = sorted(df[group_col].dropna().unique())
        color_palette = px.colors.qualitative.Plotly
        
        for i, group_val in enumerate(unique_groups):
            group_df = df[df[group_col] == group_val]
            color = color_palette[i % len(color_palette)]
            
            fig.add_trace(go.Scatter(
                x=group_df[x],
                y=group_df[y],
                mode=mode,
                name=str(group_val),
                line=dict(color=color),
                marker=dict(color=color)
            ))
    else:
        # --- Single Time Series ---
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], mode=mode, name=y,
            line=dict(color=line_color) if line_color else None
        ))

    # Update layout for better visualization
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        xaxis_title=x,
        yaxis_title=y,
        hovermode='x unified',
        template='plotly_white',
        showlegend=(group_col is not None)
    )

    return fig


if __name__=='__main__':

    def augment_dataset_datetime_random(df, start_date, end_date):
        _df = df.copy()
        n_rows = len(_df)
        random_days = np.random.randint(0, (end_date - start_date).days + 1, size=n_rows)
        _df['Date'] = start_date + pd.to_timedelta(random_days, unit='D')
        return _df

    # df = pd.read_csv('telco_churn_data.csv')
    # numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-05-01') # 04-31 (4 months)
    # df = augment_dataset_datetime_random(df, start_date, end_date)

    # _df = df.set_index('Date')

    # y_col = 'TotalCharges'
    # color_col = 'gender'
    # # freq, freq_label = 'D', 'Daily'
    # # freq, freq_label = 'W', 'Weekly'
    # freq, freq_label = 'ME', 'Monthly'

    # if color_col == None:
    #     time_data = _df[y_col].resample(freq).sum().reset_index()
    #     title = f'{y_col} Over Time ({freq_label})'
    # else:
    #     time_data = _df.groupby([pd.Grouper(freq=freq), color_col])[y_col].sum().reset_index()
    #     title = f'{y_col} Over Time ({freq_label}) by {color_col}'

    # fig_ts = plotly_timeseries(
    #     df=time_data,
    #     title=title,
    #     x='Date',
    #     y=y_col,
    #     group_col=color_col
    # )
    # fig_ts.show()

    from sklearn.datasets import make_classification
    import random


    # --- Dataset Configuration ---
    for n_samples in [100]:#, 100000]:
    # for n_samples in [100, 1000, 10000]:#, 100000]:
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

        df = augment_dataset_datetime_random(df, start_date, end_date)

        _df = df.set_index('Date')
        y_col = 'num_feature_0'
        color_col = 'cat_feature_1_small'
        # freq, freq_label = 'D', 'Daily'
        # freq, freq_label = 'W', 'Weekly'
        freq, freq_label = 'ME', 'Monthly'

        if color_col == None:
            time_data = _df[y_col].resample(freq).sum().reset_index()
            title = f'{y_col} Over Time ({freq_label})'
        else:
            time_data = _df.groupby([pd.Grouper(freq=freq), color_col])[y_col].sum().reset_index()
            title = f'{y_col} Over Time ({freq_label}) by {color_col}'

        # categories = pd.cut(data, bins=5, labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"])

        # fig_bar_box = plotly_bar_box_features_optimized(df, small_categorical_features, numerical_features)


        fig_ts = plotly_timeseries(time_data, x='Date', y=y_col, group_col=color_col)
        # pio.write_html(fig_bar_box, file=f'_ign_fig_ts__{n_samples}.html', auto_open=False, include_plotlyjs=True)

        # fig_bar_box.to_html('n_samples.html')

    fig_ts.show()



    # fig_ts = plotly_timeseries(
    #     df=time_data,
    #     title=title,
    #     x='Date',
    #     y=y_col,
    #     group_col=color_col
    # )