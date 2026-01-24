import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


class Statsmodels2SklearnInterface():
    pass
    # implements methods from the sklearn interface


class SklearnLikeBase:

    def predict(self, X, threshold=0.5):
        """
        Predict binary outcomes (0 or 1) based on a threshold.
        """
        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int)

    def summary(self):
        """
        Returns the summary of the fitted GLM model.
        """
        if self.results is None:
            raise ValueError("Model is not fitted yet. Call `fit` before `summary`.")
        return self.results.summary()

    def get_params(self, deep=True):
        """
        Returns the parameters of the estimator (mimics sklearn API).
        """
        return {"family": self.family}

    def set_params(self, **params):
        """
        Sets the parameters of the estimator (mimics sklearn API).
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class OldSklearnLikeGLM(SklearnLikeBase):
    def __init__(self, family=None):
        """
        Initializes the GLM model.
        :param family: A statsmodels family object. Default is sm.families.Binomial for logistic regression.
        """
        self.family = family if family else sm.families.Binomial()
        self.model = None
        self.results = None
        self.has_constant = False # Tracks whether a constant was added during `fit`
        self._X_columns = None # Store original column names
        self._formula_columns = None # Store column names used in the formula
        self._inv_formula_columns = None

    def fit(self, X, y):
        """
        Fits the GLM model with a constant term.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Store original column names
        self._X_columns = X.columns.tolist()

        # Add constant term
        X = sm.add_constant(X)
        self.has_constant = True

        # Combine X and y into a single DataFrame for formula-based modeling
        data = X.copy()
        data['y'] = y

        # Create safe column names for the formula
        self._formula_columns = {
            col: f"col_{i}" for i, col in enumerate(X.columns)
        }

        # Broken below
        # self._inv_formula_columns = {v:k for k,v in model._formula_columns.items()}

        # data.rename(columns=self._formula_columns, inplace=True)

        # # Construct the formula
        # formula = "y ~ " + " + ".join(self._formula_columns.values())

        # # Fit GLM model
        # self.model = smf.glm(formula=formula, data=data, family=Binomial(link=logit())) #family=self.family)
        # self.results = self.model.fit(maxiter=1000)

        return self

    def predict_prob(self, X):
        """
        Predict probabilities using the fitted model.
        """
        if self.results is None:
            raise ValueError("Model is not fitted yet. Call `fit` before `predict`.")

        # Ensure X is a DataFrame with the same columns as training data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._X_columns)

        if self.has_constant:
            X = sm.add_constant(X, has_constant='add')

        # Rename columns to match those used during fitting
        X.rename(columns=self._formula_columns, inplace=True)
        return self.results.predict(X)

    # def summary(self):
    #     """
    #     Returns the summary of the fitted GLM model.
    #     """
    #     if self.results is None:
    #         raise ValueError("Model is not fitted yet. Call `fit` before `summary`.")
    #     summary = self.results.summary()
    #     data.rename(columns=self._formula_columns, inplace=True)
    #     model._inv_formula_columns

    def summary(self, data):
        """
        Returns the summary of the fitted GLM model.
        """
        if self.results is None:
            raise ValueError("Model is not fitted yet. Call `fit` before `summary`.")
        summary = self.results.summary()
        data.rename(columns=self._formula_columns, inplace=True)
        self.model._inv_formula_columns


class CustomGLMResults(GLMResultsWrapper):
    """Custom results wrapper to clean up parameter names in summary"""
    def summary(self, *args, **kwargs):
        summary = super().summary(*args, **kwargs)
        # Get the cleaned parameter names (remove Q('...'))
        param_names = [name.replace("Q('", "").replace("')", "") 
                      if name != "const" else "Intercept"
                      for name in self.model.exog_names]
        # Replace in summary
        for i, name in enumerate(param_names):
            summary.tables[1].data[i][0] = name
        return summary


class CustomGLM(sm.GLM):
    """Custom GLM class that uses our custom results wrapper"""
    _results_class = CustomGLMResults


class SklearnLikeGLM(SklearnLikeBase):

    # TODO(1.8): Remove this attribute
    _estimator_type = "classifier"


    def __init__(self, family=None):
        """
        Initializes the GLM model.
        :param family: A statsmodels family object. Default is sm.families.Binomial for logistic regression.
        """
        self.family = family if family else sm.families.Binomial()
        self.model = None
        self.results = None
        self.has_constant = False  # Tracks whether a constant was added during fit
        self._X_columns = None  # Store original column names
        self.is_fitted = False
        self.classes_ = np.array([1, 0])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        # tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

    def fit(self, X, y):
        """
        Fits the GLM model with a constant term.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Store original column names
        self._X_columns = X.columns.tolist()
        
        # Add constant term with the name 'Intercept'
        X = pd.DataFrame(sm.add_constant(X, has_constant='add'))
        X.rename(columns={'const': 'Intercept'}, inplace=True)
        self.has_constant = True
        
        # Create the model using the DataFrame with proper column names
        self.model = GLM(y, X, family=self.family)
        # self.model = smf.glm(formula=formula, data=data, family=Binomial(link=logit())) #family=self.family)
        self.results = self.model.fit(maxiter=1000)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        # return self.predict_prob(X)
        return np.stack([
            self.predict_prob(X),
            np.array(1) - self.predict_prob(X),
        ], axis=1)

    def predict(self, X):
        y_prob = self.predict_prob(X)
        y_pred = (y_prob >= 0.5).astype(int).values
        return y_pred

    def predict_prob(self, X):
        """
        Predict probabilities using the fitted model.
        """
        if self.results is None:
            raise ValueError("Model is not fitted yet. Call fit before predict.")
        
        # Ensure X is a DataFrame with the same columns as training data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._X_columns)
            
        # Add constant if it was used during fitting
        if self.has_constant:
            X = sm.add_constant(X, has_constant='add')
            X = pd.DataFrame(X)
            X.rename(columns={'const': 'Intercept'}, inplace=True)
            
        return self.results.predict(X)

    def __sklearn_is_fitted__(self):
        return self.is_fitted


class SklearnLikeLogit(SklearnLikeBase):
    def __init__(self):
        self.model = None
        self.results = None
        self.has_constant = False  # Tracks whether a constant was added during `fit`
        self._X_columns = None  # Store original column names

    def fit(self, X, y):
        """
        Fits the logistic regression model with a constant term.
        """
        # if self.results is None:
        #     raise ValueError("Model is not fitted yet. Call `fit` before `predict`.")

        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Store original column names
        self._X_columns = X.columns.tolist()

        # Add constant term
        X = sm.add_constant(X, has_constant='add')
        self.has_constant = True

        self.model = sm.Logit(y, X)
        self.results = self.model.fit() #disp=False # method='newton', maxiter=100, disp=True
        # self.results = self.model.fit_regularized(start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0.01, trim_mode='size', auto_trim_tol=0.01, size_trim_tol=0.001, qc_tol=0.03,)

        return self

    def predict_prob(self, X):
        """
        Predict probabilities using the fitted model.
        """
        if self.results is None:
            raise ValueError("Model is not fitted yet. Call `fit` before `predict`.")

        # Ensure X is a DataFrame with the same columns as training data
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._X_columns)

        # Add constant term
        if self.has_constant:
            X = sm.add_constant(X, has_constant='add')

        return self.results.predict(X)


# \/ \/ \/ This should go to classification anatomy \/ \/ \/

# classifier_logreg = SklearnLikeGLM()
# classifier_logreg.fit(x_train_df, y_train)
# classifier_logreg.predict(x_train_df)
# classifier_logreg.predict_proba(x_train_df)

# RocCurveDisplay.from_estimator(classifier_logreg, x_test, y_test)