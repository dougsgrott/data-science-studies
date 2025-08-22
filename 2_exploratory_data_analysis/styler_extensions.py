import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_hex, LinearSegmentedColormap
# Import the DataFrame accessor decorator
from pandas.api.extensions import register_dataframe_accessor

@register_dataframe_accessor("style_ext")
class StyleExtension:
    def __init__(self, pandas_obj):
        self._df = pandas_obj
        # The Styler object is created and managed internally by the accessor
        self.styler = pandas_obj.style

    def _repr_html_(self):
        """
        Allows the final object to be displayed as a styled table in Jupyter.
        """
        return self.styler._repr_html_()

    def numeric_ranges(self, columns=None, value_range=(-1, 1), cmap='RdYlGn'):
        """
        Stylizes numeric columns using a color gradient.
        """
        min_val, max_val = value_range
        norm = Normalize(vmin=min_val, vmax=max_val)
        cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

        def colorize_numeric(val):
            if pd.isna(val): return ''
            try:
                rgba = cmap(norm(float(val)))
                return f'background-color: {to_hex(rgba)}'
            except (ValueError, TypeError):
                return ''

        target_cols = columns if columns is not None else self._df.select_dtypes(include=np.number).columns
        self.styler.map(colorize_numeric, subset=list(target_cols))
        
        return self

    def boolean_highlights(self, columns=None, true_color='limegreen', false_color='salmon'):
        """
        Stylizes boolean columns by highlighting True/False values.
        """
        def colorize_boolean(val):
            if pd.isna(val) or not isinstance(val, bool): return ''
            return f'background-color: {true_color}' if val else f'background-color: {false_color}'

        target_cols = columns if columns is not None else self._df.select_dtypes(include=bool).columns
        self.styler.map(colorize_boolean, subset=list(target_cols))
        
        return self

    def categorical_map(self, columns=None, cmap='viridis'):
        """
        Stylizes categorical columns by mapping unique values to a colormap.
        """
        target_cols = columns if columns is not None else self._df.select_dtypes(include=['category', 'object']).columns
        cmap_func = cm.get_cmap(cmap)

        for col in target_cols:
            unique_vals = self._df[col].dropna().unique()
            color_map = {val: to_hex(cmap_func(i / len(unique_vals))) for i, val in enumerate(unique_vals)}
            self.styler.map(lambda v: f'background-color: {color_map.get(v, "")}', subset=[col])
            
        return self