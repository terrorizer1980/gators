# License: Apache-2.0
import warnings
from typing import Dict, List

import numpy as np

from binning import bin_rare_events

from ..transformers.transformer import Transformer
from ..data_cleaning.replace import Replace
from ..util import util

from gators import DataFrame, Series


class BinSingleTargetClassCategories(Transformer):
    """Bin single target class categories.

    Ensure that the target class ratio for each categy is between 0 and 1 excluded.
    Note that this transformer should only be used for binary classification problems.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.binning import BinRareCategories
    >>> obj = BinRareCategories(min_ratio=0.5)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
            A       B
    0       a  OTHERS
    1       a  OTHERS
    2  OTHERS  OTHERS

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'OTHERS'],
           ['a', 'OTHERS'],
           ['OTHERS', 'OTHERS']], dtype=object)
    """

    def __init__(self):
        Transformer.__init__(self)
        self.replace = None
        self.columns = []
        self.idx_columns: np.ndarray = np.array([])
        self.mapping = {}
        self.is_binned = False

    def fit(self, X: DataFrame, y: Series) -> "BinSingleTargetClassCategories":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series.
            Target values.

        Returns
        -------
        BinSingleTargetClassCategories
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        if object not in X.dtypes.to_numpy():
            self.is_binned = False
            warnings.warn(
                """`X` does not contain object columns:
                `BinSingleTargetClassCategories` is not needed"""
            )
            return self
        y_name = y.name
        self.columns = util.get_datatype_columns(X, datatype=object)
        means = (
            util.get_function(X)
            .melt(
                util.get_function(X).join(X[self.columns], y.to_frame()), id_vars=y_name
            )
            .groupby(["variable", "value"], )
            .mean()[y_name]
        )
        means = util.get_function(X).to_pandas(means)
        means = means.groupby("variable").apply(lambda x: x.sort_index(level=1).sort_values()).droplevel(0)

        extreme_columns = (
            means[(means == 0) | (means == 1)].index.get_level_values(0).unique()
        )
        self.mapping = {c: {} for c in extreme_columns}
        for c in extreme_columns:
            cats_0, cats_1 = [], []
            idx = (means[c] == 0).sum()
            if idx:
                cats_0 = list(means[c].index[:idx+1])
            idx = (means[c] == 1).sum()
            if idx:
                cats_1 = list(means[c].index[-idx-1:])
            if bool(set(cats_0) & set(cats_1)):
                cats_0 = sorted(list(set(cats_0+cats_1)))
                cats_1 = sorted(list(set(cats_0+cats_1)))
            
            d_0 = dict(zip(cats_0, len(cats_0)*['|'.join(cats_0)]))
            d_1 = dict(zip(cats_1, len(cats_1)*['|'.join(cats_1)]))
            self.mapping[c] = {**d_0, **d_1}
        self.is_binned = True if sum([len(val) for val in self.mapping.values()]) else False
        if not self.is_binned:
            return self
        self.replace = Replace(to_replace_dict=self.mapping).fit(X)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        self.columns_ = list(X.columns)
        if not self.is_binned:
            return X
        return self.replace.transform(X)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X : np.ndarray
             Array.

        Returns
        -------
        X : np.ndarray
            Transformed NumPy array.

        """
        self.check_array(X)
        if not self.is_binned:
            return X
        return self.replace.transform_numpy(X)
