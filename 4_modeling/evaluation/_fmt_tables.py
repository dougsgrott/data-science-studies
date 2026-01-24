
import pandas as pd
import numpy as np
import re


class FormattedSummaryTableCreator():

    def __init__(self, model):
        self.model = model

    def _calculate_default_rate_by_feature(self, feature_list, X, y):
        default_rate_list = []
        for feature in feature_list:
            try:
                df = X[X[feature] == 1]
                default_rate = y[df.index].mean() * 100
                default_rate_list.append(default_rate)
            except KeyError:
                default_rate_list.append(-1)
            except Exception as e:
                raise e
        return default_rate_list

    def calculate_default_rate_by_feature(self, results_df, X, y):
        default_rate = self._calculate_default_rate_by_feature(results_df['feature_input'], X, y)
        results_df['inadimplencia'] = default_rate
        return results_df

    def _calculate_delta(self, betas):
        #! betas[0] = modelo_params_0 = intercept
        a = np.exp(betas[0] + betas)
        b = np.exp(betas[0])
        # delta_score = ((1 - a/(1+a)) - (1 - b/(1+b))) * 1000
        delta_score = ((a/(1+a)) - (b/(1+b))) * 1000
        return delta_score

    def calculate_delta(self, results_df):
        delta_score = self._calculate_delta(results_df['betas'])
        results_df['delta_score'] = delta_score
        return results_df

    def _calculate_scr(self, X, y):
        scr = (1 - self.model.predict_prob(X)) * 1000
        scr_bin = pd.qcut(scr, 5, labels=None, retbins=False, precision=0, duplicates='raise')
        scr_bin = scr_bin.apply(lambda x: str(x).replace('(', '').replace(']', '').replace('.0', '').replace(', ', '-'))
        X['Score Range'] = scr_bin

        default_rate_list = []
        for scr in scr_bin.unique():
            try:
                df = X[X['Score Range'] == scr]
                default_rate = y[df.index].mean() * 100
                default_rate_list.append(default_rate)
            except KeyError:
                default_rate_list.append(-1)
            except Exception as e:
                raise e
        return default_rate_list

    def calculate_default_rate_by_scr(self, X, y):
        scr = self._calculate_scr(X, y)
        df = pd.DataFrame(scr, columns=['Score'])
        return df

    def _get_model_summary(self):
        """
        Extract model summary information based on model type.
        Returns DataFrame with coefficients, standard errors and p-values.
        """
        if self.model.__class__.__name__ in ['SklearnLikeLogit', 'SklearnLikeGLM', 'OldSklearnLikeGLM']:
            results_df = pd.read_html(
                self.model.results.summary().tables[1].as_html(), 
                header=0, 
                index_col=0
            )[0]
            results_df = results_df[['coef', 'std err', 'P>|z|']]
            return results_df.reset_index().rename(columns={
                'index': 'feature_input',
                'coef': 'betas',
                'P>|z|': 'p_value'
            })
        return None

    def color_delta_score(self, val, min_val, max_val):
        if pd.isna(val):
            return ''
        
        # Calculate the midpoint
        mid_val = (min_val + max_val) / 2

        # Normalize value between -1 and 1 relative to the midpoint
        if val <= mid_val:
            normalized = (val - mid_val) / (mid_val - min_val)
        else:
            normalized = (val - mid_val) / (max_val - mid_val)

        if normalized <= 0:  # Red to White (min to mid)
            r = 255
            g = int(255 * (1 + normalized))  # Increasing green
            b = int(255 * (1 + normalized))  # Increasing blue
        else:  # White to Green (mid to max)
            r = int(255 * (1 - normalized))  # Decreasing red
            g = int(255 * (1 - normalized) + 128 * normalized)  # Increasing green
            b = int(255 * (1 - normalized))  # Decreasing blue

        return f'background-color: rgb({r},{g},{b})'

    # Define the styling function
    def color_inadimplencia(self, val):
        # Only color values between 0 and 100
        if pd.isna(val) or val < 0 or val > 100:
            return ''
        
        # Normalize value between -1 and 1 where:
        # -1 represents 0 (green)
        # 0 represents 50 (white)
        # 1 represents 100 (red)
        normalized = (val - 50) / 50
        
        if normalized <= 0:  # Green to White (0 to 50)
            # Mix between rgb(0,128,0) and rgb(255,255,255)
            r = int(255 * (1 + normalized))
            g = int(255 * (1 + normalized) - 128 * normalized)
            b = int(255 * (1 + normalized))
        else:  # White to Red (50 to 100)
            # Mix between rgb(255,255,255) and rgb(255,0,0)
            r = 255
            g = int(255 * (1 - normalized))
            b = int(255 * (1 - normalized))
        
        return f'background-color: rgb({r},{g},{b})'

    def color_betas(self, val, min_val, max_val):
        if pd.isna(val):
            return ''
        
        # Calculate the midpoint
        mid_val = (min_val + max_val) / 2

        # Normalize value between -1 and 1 relative to the midpoint
        if val <= mid_val:
            normalized = (val - mid_val) / (mid_val - min_val)
        else:
            normalized = (val - mid_val) / (max_val - mid_val)

        if normalized <= 0:  # Red to White (min to mid)
            r = 255
            g = int(255 * (1 + normalized))  # Increasing green
            b = int(255 * (1 + normalized))  # Increasing blue
        else:  # White to Green (mid to max)
            r = int(255 * (1 - normalized))  # Decreasing red
            g = int(255 * (1 - normalized) + 128 * normalized)  # Increasing green
            b = int(255 * (1 - normalized))  # Decreasing blue

        return f'background-color: rgb({r},{g},{b})'

    def _parse_parent_feature(self, df):
        # index
        df['parent_index'] = [c.split(':')[0] for c in df['feature_input'].values]
        df = df[ ['parent_index'] + [ col for col in df.columns if col != 'parent_index' ] ]
        return df

    def sort_feature_input(self, df: pd.DataFrame) -> pd.DataFrame:
        def extract_prefix_and_bounds(feature):
            match = re.match(r"(.+?):\s*(\(-inf|[\[\(].*?\))", feature)
            if match:
                prefix = match.group(1)
                bounds = re.findall(r"[-+]?\d*\.\d+|\d+", match.group(2))
                bounds = [float(b) for b in bounds] if bounds else []
                # Handle -inf and inf explicitly
                if "(-inf" in feature:
                    bounds.insert(0, -np.inf)
                if "inf)" in feature:
                    bounds.append(np.inf)
                return prefix, tuple(bounds)  # Convert list to tuple for sorting
            # If no match, assign default values to ensure consistency.
            return feature, (np.nan,)
        
        # Create separate columns for prefix and bounds.
        df["prefix"], df["bounds"] = zip(*df["feature_input"].apply(extract_prefix_and_bounds))
        
        # Create a combined sort key column. Sorting by prefix first, then by bounds.
        df["sort_key"] = df.apply(lambda row: (row["prefix"], row["bounds"]), axis=1)
        
        # Now sort by the combined sort key.
        df_sorted = df.sort_values("sort_key")
        
        # Drop helper columns before returning.
        df_sorted = df_sorted.drop(columns=["prefix", "bounds", "sort_key"])
        
        return df_sorted

    def format_table(self, df):
        # Apply the styling

        # min_delta_val = df['delta_score'].min() * 2
        # if df['delta_score'].min() < 0:
        #     min_delta_val = df['delta_score'].min() * 2
        # else:
        #     min_delta_val = df['delta_score'].min() / 2
        # max_delta_val = df['delta_score'].max() * 2
        max_delta_val = df['delta_score'].abs().max() * 1.5
        min_delta_val = - max_delta_val

        # min_betas_val = df['betas'].min() * 2
        # if df['betas'].min() < 0:
        #     min_betas_val = df['betas'].min() * 2
        # else:
        #     min_betas_val = df['betas'].min() / 2
        # max_betas_val = df['betas'].max() * 2
        max_betas_val = df['betas'].abs().max() * 1.5
        min_betas_val = - max_betas_val

        df = (
          df.style
          .hide(axis="index") 
          .set_properties(subset=df.columns, **{"text-align": "center"})
          .format({
              "betas":  "{:.2f}", 
              "p_value":  "{:.2f}", 
              "delta_score":    "{:.2f}", 
              "inadimplencia": "{:.2f}%", 
          })
          .set_table_styles(
              [
                  # Minimum width for all headers
                  {
                      "selector": "th.col_heading",  
                      "props": [("min-width", "50px")]
                  },
                  # Minimum width for all data cells
                  {
                      "selector": "td",  
                      "props": [("min-width", "50px")]
                  },
                  {
                      "selector": "th.col_heading",
                      "props": [("background-color", "lightgray")]
                  },
              ]
          )
          .apply(
              lambda x: [
                  self.color_inadimplencia(v) if col == 'inadimplencia' 
                  else self.color_delta_score(v, min_delta_val, max_delta_val) if col == 'delta_score'
                  else self.color_betas(v, min_betas_val, max_betas_val) if col == 'betas'
                  else '' for col, v in x.items()
              ], axis=1)
          )

        return df

    def create(self, X, y):
        summary = self._get_model_summary()
        summary = self.sort_feature_input(summary)
        summary = self._parse_parent_feature(summary)

        summary = self.calculate_delta(summary)
        summary = self.calculate_default_rate_by_feature(summary, X, y)

        summary.drop(columns=['parent_index', 'std err'], inplace=True)

        stylized_summary = self.format_table(summary)

        return summary, stylized_summary


class ScoreTableCreator():

    def create_scr_range(self, model, q, X, y):
        y_pred = model.predict(X)
        # scr = np.rint((1 - model.predict_prob(X)) * 1000)
        scr = np.rint((model.predict_proba(X)) * 1000) [:,1]

        scr_bin = pd.Series(pd.qcut(scr, q, labels=None, retbins=False, precision=0, duplicates='raise'))
        scr_bin = scr_bin.apply(lambda x: str(x).replace('(', '').replace(']', '').replace('.0', '').replace(', ', '-'))
        X['Score Range'] = scr_bin
        bins = sorted(scr_bin.unique())

        stop_K  = [int(b.split('-')[1]) for b in bins]
        start_K = [int(b.split('-')[0]) for b in bins]
        df = pd.DataFrame({
          'Score Range': bins,
          'Start K': start_K,
          'Stop K': stop_K,
          })
        df = df.sort_values(by='Start K').reset_index(drop=True)
        scr_df = pd.DataFrame({
            'y': y,
            'y_pred': y_pred,
            'scr': scr
        })
        # scr_df = pd.concat([y, y_pred, scr], axis=1)
        # scr_df.columns = ['y', 'y_pred', 'scr']
        return df, scr_df

    def add_bottom_top_rows(self, df):
        # Add top rows
        first_row = pd.DataFrame({col: [0] for col in df.columns})
        k = df['Start K'].to_list()[0]
        second_row = pd.DataFrame(data=[[f'0-{k}', 0, k]], columns=df.columns)
        # Add bottom row
        k = df['Stop K'].to_list()[-1]
        bottom_row = pd.DataFrame(data=[[f'{k}-1000', k, 1000]], columns=df.columns)

        full_df = pd.concat([first_row, second_row, df, bottom_row], axis=0).reset_index(drop=True)
        return full_df

    def create(self, model, q, X, y, format=True):
        self.X_columns = X.columns
        binned_scr_df, scr_df = self.create_scr_range(model, q, X.copy(), y.copy())
        ext_binned_scr_df = self.add_bottom_top_rows(binned_scr_df)

        self.binned_scr_df = binned_scr_df
        self.ext_binned_scr_df = ext_binned_scr_df
        self.scr_df = scr_df
        return self


class FormattedSegmentedScoreTableCreator():

    def calculate_default_rate_by_scr(self, df, scr_df):
        for i in range(len(df)):
            start_k = df.loc[i, 'Start K']
            end_k = df.loc[i, 'Stop K']
            range_df = scr_df[(scr_df['scr'] > start_k) & (scr_df['scr'] <= end_k)]

            maus_count = sum(range_df['y'] == 0)
            bons_count = sum(range_df['y'] == 1)
            maus_pct = maus_count / (maus_count + bons_count) * 100
            bons_pct = bons_count / (maus_count + bons_count) * 100
            # ks = ks_2samp(range_df['y_pred'], range_df['y'])[0] * 100

            df.loc[i, 'Maus'] = maus_pct
            df.loc[i, 'Bons'] = bons_pct
            df.loc[i, 'B/M'] = bons_pct / maus_pct
            df.loc[i, 'Total'] = bons_pct + maus_pct
            # df.loc[i, 'KS'] = ks
        return df

    def drop_rename_columns(self, df):
        # df = df.rename(columns={'Stop K': 'K'})
        df.drop(columns=['Start K', 'Stop K'], inplace=True)

        df = df.reset_index()
        df['index'] = df['index'] + 1
        df.rename(columns={'index': 'Faixa'}, inplace=True)
        return df

    def color_inadimplencia(self, val):
        # Only color values between 0 and 100
        if pd.isna(val) or val < 0 or val > 100:
            return ''

        # Normalize value between -1 and 1 where:
        # -1 represents 0 (green)
        # 0 represents 50 (white)
        # 1 represents 100 (red)
        normalized = (val - 50) / 50
        start_color_rgb = [79, 189, 99]
        stop_color_rgb = [230, 131, 122]

        if normalized <= 0:
            # Interpolate from #27b03f (rgb(39,176,63)) to #ffffff (rgb(255,255,255))
            # fraction goes from 0 (val=0) to 1 (val=50)
            fraction = 1 + normalized  # maps [-1..0] to [0..1]
            r = int(start_color_rgb[0]  * (1 - fraction) + 255 * fraction)
            g = int(start_color_rgb[1] * (1 - fraction) + 255 * fraction)
            b = int(start_color_rgb[2]  * (1 - fraction) + 255 * fraction)
        else:
            # Interpolate from #ffffff (rgb(255,255,255)) to #cc4033 (rgb(204,64,51))
            # fraction goes from 0 (val=50) to 1 (val=100)
            fraction = normalized  # maps [0..1] to [0..1]
            r = int(255 * (1 - fraction) + stop_color_rgb[0] * fraction)
            g = int(255 * (1 - fraction) + stop_color_rgb[1]  * fraction)
            b = int(255 * (1 - fraction) + stop_color_rgb[2]  * fraction)

        return f'background-color: rgb({r},{g},{b})'

    def format_table(self, df):
        # Apply the styling
        if format:
            df = (
              df.style
              .hide(axis="index") 
              .set_properties(subset=df.columns, **{"text-align": "center"})
              .format({
                  "Maus":  "{:.1f}%", 
                  "Bons":  "{:.1f}%", 
                  "KS":    "{:.1f}%", 
                  "Total": "{:.1f}%", 
                  "B/M":   "{:.2f}"
              })
              .set_table_styles(
                  [
                      # Minimum width for all headers
                      {
                          "selector": "th.col_heading",  
                          "props": [("min-width", "100px")]
                      },
                      # Minimum width for all data cells
                      {
                          "selector": "td",  
                          "props": [("min-width", "100px")]
                      },
                      {
                          "selector": "th.col_heading",
                          "props": [("background-color", "lightgray")]
                      },
                  ]
              )
              .apply(
                lambda x: [
                    self.color_inadimplencia(v) if col == 'Maus' 
                    else self.color_inadimplencia(v) if col == 'Bons'
                    else '' for col, v in x.items()], axis=1))
        return df

    def create(self, model, q, X, y, format=True):
        self.X_columns = X.columns
        scr_creator = ScoreTableCreator().create(model, q, X, y)
        table = scr_creator.binned_scr_df
        scr_df = scr_creator.scr_df
        table = self.calculate_default_rate_by_scr(table, scr_df)
        table = self.drop_rename_columns(table)

        stylized_table = self.format_table(table)

        return table, stylized_table


class FormattedSegmentedPercentageTableCreator():

    def calculate_default_rate_by_scr(self, df, model, X, y, scr_df):
        maus_total = sum(scr_df['y'] == 0)
        bons_total = sum(scr_df['y'] == 1)
        maus_pct, bons_pct = 0, 0

        for i in range(len(df)):
            start_k = df.loc[i, 'Start K']
            end_k = df.loc[i, 'Stop K']
            range_df = scr_df[(scr_df['scr'] > start_k) & (scr_df['scr'] <= end_k)]
            range_X = X.iloc[range_df.index]
            prog_df = scr_df[scr_df['scr'] <= end_k]

            if range_df.empty:
                # maus_pct, bons_pct
                ks = 0
            else:
                maus_count = sum(prog_df['y'] == 0)
                bons_count = sum(prog_df['y'] == 1)
                # maus_pct = maus_count / (maus_count + bons_count)
                # bons_pct = bons_count / (maus_count + bons_count)
                maus_pct = maus_count / maus_total
                bons_pct = bons_count / bons_total

                # ks = ks_2samp(range_df['y_pred'], range_df['y'])[0]
                # ks = calculate_ks(model, range_X, range_df['y'])[0]
                ks = np.abs(maus_pct - bons_pct)

            df.loc[i, 'Maus'] = maus_pct * 100
            df.loc[i, 'Bons'] = bons_pct * 100
            df.loc[i, 'KS'] = ks * 100
        return df

    def drop_rename_columns(self, df):
        df = df.rename(columns={'Stop K': 'K'})
        df.drop(columns=['Start K'], inplace=True)

        df = df.reset_index()
        df.rename(columns={'index': 'Faixa'}, inplace=True)
        return df

    # Define the styling function
    def color_inadimplencia(self, val):
        # Only color values between 0 and 100
        if pd.isna(val) or val < 0 or val > 100:
            return ''

        # Normalize value between -1 and 1 where:
        # -1 represents 0 (green)
        # 0 represents 50 (white)
        # 1 represents 100 (red)
        normalized = (val - 50) / 50
        start_color_rgb = [79, 189, 99]
        stop_color_rgb = [230, 131, 122]

        if normalized <= 0:
            # Interpolate from #27b03f (rgb(39,176,63)) to #ffffff (rgb(255,255,255))
            # fraction goes from 0 (val=0) to 1 (val=50)
            fraction = 1 + normalized  # maps [-1..0] to [0..1]
            r = int(start_color_rgb[0]  * (1 - fraction) + 255 * fraction)
            g = int(start_color_rgb[1] * (1 - fraction) + 255 * fraction)
            b = int(start_color_rgb[2]  * (1 - fraction) + 255 * fraction)
        else:
            # Interpolate from #ffffff (rgb(255,255,255)) to #cc4033 (rgb(204,64,51))
            # fraction goes from 0 (val=50) to 1 (val=100)
            fraction = normalized  # maps [0..1] to [0..1]
            r = int(255 * (1 - fraction) + stop_color_rgb[0] * fraction)
            g = int(255 * (1 - fraction) + stop_color_rgb[1]  * fraction)
            b = int(255 * (1 - fraction) + stop_color_rgb[2]  * fraction)

        return f'background-color: rgb({r},{g},{b})'

    def format_table(self, df):
        # Apply the styling
        if format:
            df = (
              df.style
              .hide(axis="index") 
              .set_properties(subset=df.columns, **{"text-align": "center"})
              .format({
                  "Maus":  "{:.1f}%", 
                  "Bons":  "{:.1f}%", 
                  "KS":    "{:.1f}%", 
                  "Total": "{:.1f}%", 
                  "B/M":   "{:.2f}"
              })
              .set_table_styles(
                  [
                      # Minimum width for all headers
                      {
                          "selector": "th.col_heading",  
                          "props": [("min-width", "100px")]
                      },
                      # Minimum width for all data cells
                      {
                          "selector": "td",  
                          "props": [("min-width", "100px")]
                      },
                      {
                          "selector": "th.col_heading",
                          "props": [("background-color", "lightgray")]
                      },
                  ]
              )
              .apply(
                lambda x: [
                    self.color_inadimplencia(v) if col == 'Maus' 
                    else self.color_inadimplencia(v) if col == 'Bons'
                    else '' for col, v in x.items()], axis=1))
        return df

    def create(self, model, q, X, y):
        self.X_columns = X.columns
        scr_creator = ScoreTableCreator().create(model, q, X, y)
        table = scr_creator.ext_binned_scr_df
        scr_df = scr_creator.scr_df

        table = self.calculate_default_rate_by_scr(table, model, X, y, scr_df)
        table = self.drop_rename_columns(table)

        stylized_table = self.format_table(table)

        return table, stylized_table


# foo = FormattedSegmentedScoreTableCreator()
# foo.create(
#     # model=classifier_rf,
#     model=classifier_logreg,
#     q=7,
#     X=x_test_df,
#     y=y_test,
#     format=True
# )[1]


# foo = FormattedSegmentedPercentageTableCreator()
# pct_table, fmt_pct_table = foo.create(
#     model=classifier_rf,
#     # model=classifier_logreg,
#     q=7,
#     X=x_test_df,
#     y=y_test,
# )
# fmt_pct_table


# foo = ScoreTableCreator()
# foo.create(
#     # model=classifier_rf,
#     model=classifier_logreg,
#     q=5,
#     X=x_test_df,
#     y=y_test,
#     format=False
# )
