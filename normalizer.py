from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
from scipy.optimize import brentq


class Normalizer():
    """     A scaler that maps from an input space X to the space of a normally distributed random variable with mean 0 and variance 1
            Note: this is intended to work with 2D dataset arrays.
    """

    def __init__(self):
        """     Initialize the transformer
            Args:
                None
        """

        self.ecdf = None
        self.data_mean = None
        self.data_max = None
        self.data_min = None

    def fit(self, values, axis=-1):
        """     Computes an approximation to the cumulative distribution function characterized by 'values.'
                The returned function returns the probability (0,1) of being less than or equal to the supplied value.
            Args:
                values (array)  :   Array with shape (n, m) with observation values.
            Out:
                eCDF (callable)    :   Callable with input shape (n,) that will return the value of the empirical CDF for a given input
        """

        values = np.array(values)

        percentiles = np.linspace(0, 100, 200)

        max = []
        min = []
        for i in range(values.shape[axis]):
            max += [np.amax(values[:, i])]
            min += [np.amin(values[:, i])]

        self.data_max = np.array(max)
        self.data_min = np.array(min)

        means = np.empty(values.shape[axis])
        for i in range(means.shape[axis]):
            means[i] = np.mean(values[:, i])

        self.data_mean = means

        f_array = []

        for i in range(values.shape[axis]):
            f_array += [ECDF(values[:, i])]

        def eCDF(array, axis=-1):
            holder = np.empty(array.shape)
            for i in range(array.shape[axis]):
                holder[:, i] = f_array[i](array[:, i])

            return holder

        self.ecdf = eCDF

    def transform(self, x_values, axis=-1):
        """     Computes the CDF values of the supplied x_values and uses ppf to map data to N(0, 1).
                f : X -> N(0,1)
            Args:
                x_values (array)    : Array with shape (n, m)
            Out:
                norm_values (array) : Array with shape (n, m) where all
        """

        x_values = np.array(x_values)
        # find the CDF values in the original distribution.
        cdf_values = self.ecdf(x_values)

        # Clip eCDF values to (0, 1) and normalize them
        norm_values = np.empty(cdf_values.shape)
        for i in range(cdf_values.shape[axis]):
            clipped_cdf = np.clip(cdf_values[:, i], a_min=0.00001, a_max=0.99999)
            norm_values[:, i] = norm.ppf(clipped_cdf)

        return norm_values

    def fit_transform(self, x_values):
        """     Does 'fit' and then 'transform' at the same timeself.
            Args:
                x_values (array) : Array with shape (n, m).
            Out:
                norm_values (array) : Array with shape (n, m) where all.
        """

        x_values = np.array(x_values)
        self.fit(x_values)
        norm_values = self.transform(x_values)

        return norm_values

    def inverse_transform(self, N_values, axis=-1):
        """     Computes the normCDF and uses the inverse eCDF to map N(0,1) to the dataself.
                f : N(0,1) -> X
            Args:
                N_values (array)    :   Normalized array with shape (n, m).
            Out:
                X_values (array)    :   Denormalized array with shape (n, m).
        """

        N_values = np.array(N_values)

        # Define a statement where 't' is equivalent to the inverse empirircal cumulative distribution function.
        def inv_ecdf_expression(t, s, a):
            t = np.array(t).reshape(1, -1)
            tpad1 = np.zeros((1, a))
            if a < 4:
                tpad2 = np.zeros((1, 4 - a))
                t_p = np.hstack((np.hstack((tpad1, t)), tpad2))
            elif a == 4:
                t_p = np.hstack((tpad1, t))

            s = np.array(s).reshape(1, -1)
            return np.asscalar((self.ecdf(t_p)[0, a] - s).flatten())

        def inv_ecdf(x, arr):
            v = self.data_mean[arr]
            try:
                v = brentq(inv_ecdf_expression, a=self.data_min[arr], b=self.data_max[arr], args=(x, arr))
            except:
                if inv_ecdf_expression(self.data_max[arr], x, arr) == 0:
                    v = self.data_max[i]
                elif inv_ecdf_expression(self.data_min[arr], x, arr) == 0:
                    v = self.data_min[i]

            return v

        holder = np.empty(N_values.shape)
        for i in range(N_values.shape[0]):
            for j in range(N_values.shape[1]):
                holder[i][j] = inv_ecdf(norm.cdf(N_values[i][j]), j)

        return holder

    def _reset(self):
        """     Resets the initial parameters.   """

        self.ecdf = None
        self.data_mean = None
