# License: Apache-2.0
import numpy as np
import pandas as pd

from ..util import util
from ._base_feature_selection import _BaseFeatureSelection

from gators import DataFrame, Series


class InformationValue(_BaseFeatureSelection):
    """Feature selection based on the information value.

    `InformationValue` accepts only binary variable targets.

    Parameters
    ----------
    k : int
        Number of features to keep.
    regularization : float, default to 0,5.
        Insure that the weights of evidence is finite.
    max_iv : int, default to 10.
        Drop columns with an information larger than `max_iv`.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.feature_selection import InformationValue
    >>> from gators.binning import Binning
    >>> obj = InformationValue(k=3, binning=Binning(n_bins=4))

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({
    ... 'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ... 'B': [1, 1, 0, 1, 0, 0],
    ... 'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ... 'F': [1, 2, 3, 1, 2, 4]}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([1, 1, 1, 0, 0, 0], name='TARGET'), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... 'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ... 'B': [1, 1, 0, 1, 0, 0],
    ... 'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ... 'F': [1, 2, 3, 1, 2, 4]})
    >>> y = ks.Series([1, 1, 1, 0, 0, 0], name='TARGET')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... 'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ... 'B': [1, 1, 0, 1, 0, 0],
    ... 'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ... 'F': [1, 2, 3, 1, 2, 4]})
    >>> y = pd.Series([1, 1, 1, 0, 0, 0], name='TARGET')

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X, y)
           A  B  C
    0  87.25  1  a
    1   5.25  1  b
    2  70.25  0  b
    3   5.25  1  b
    4   0.25  0  a
    5   7.25  0  a

    >>> X = pd.DataFrame({
    ... 'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ... 'B': [1, 1, 0, 1, 0, 0],
    ... 'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ... 'F': [1, 2, 3, 1, 2, 4]})
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[87.25, 1, 'a'],
           [5.25, 1, 'b'],
           [70.25, 0, 'b'],
           [5.25, 1, 'b'],
           [0.25, 0, 'a'],
           [7.25, 0, 'a']], dtype=object)
    """

    def __init__(
        self,
        k: int,
        regularization: float = 0.1,
        max_iv: float = 10.0,
    ):
        if not isinstance(k, int):
            raise TypeError("`k` should be an int.")
        if not isinstance(regularization, (int, float)) or regularization < 0:
            raise TypeError("`k` should be a positive float.")
        if not isinstance(max_iv, (int, float)) or max_iv < 0:
            raise TypeError("`max_iv` should be a positive float.")
        _BaseFeatureSelection.__init__(self)
        self.k = k
        self.regularization = regularization
        self.max_iv = max_iv

    def fit(self, X: DataFrame, y: Series) -> "InformationValue":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X: DataFrame
            Input dataframe.
        y: Series
            Target values.

        self : "InformationValue"
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        X, self.feature_importances_ = self.compute_information_value(
            X, y, self.regularization
        )
        mask = self.feature_importances_ < self.max_iv
        self.feature_importances_ = self.feature_importances_[mask]
        self.feature_importances_.sort_values(ascending=False, inplace=True)
        self.selected_columns = list(self.feature_importances_.index[: self.k])
        self.columns_to_drop = [c for c in X.columns if c not in self.selected_columns]
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self

    @staticmethod
    def compute_information_value(
        X: DataFrame, y: Series, regularization: float
    ) -> pd.Series:
        """Compute information value.

        Parameters
        ----------
        X: DataFrame
            Input dataframe.
        y: np.ndarray
            Target values.

        Returns
        -------
        pd.Series
            Information value.
        """
        y_name = y.name
        counts = (
            util.get_function(X)
            .melt(util.get_function(X).join(X, y.to_frame()), id_vars=y_name)
            .groupby(["variable", "value"])
            .agg(["sum", "count"])[y_name]
        )
        counts = util.get_function(X).to_pandas(counts)
        counts.columns = ["1", "count"]
        counts["0"] = (counts["count"] - counts["1"] + regularization) / counts["count"]
        counts["1"] = (counts["1"] + regularization) / counts["count"]
        iv = (
            ((counts["1"] - counts["0"]) * np.log(counts["1"] / counts["0"]))
            .groupby(level=[0])
            .sum()
        )
        return X, iv.sort_values(ascending=False).fillna(0)
