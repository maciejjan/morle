from models.features.generic import *
import settings

import numpy as np
from scipy.special import betaln, gammaln

class PointFeature(Feature):
    pass


class PointBinomialFeature(PointFeature):
    def __init__(self, trials, prob=0.5, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.trials = trials
        self.prob = prob
    
    def cost(self, values):
        value = sum(values)
#        if (self.prob == 0.0 and value == 0) or\
#                (self.prob == 1.0 and value == self.trials):
#            return 0.0
#        elif self.prob <= 0.0 or self.prob >= 1.0:
#            raise Exception('Impossible event: %f %d %d ' % (self.prob, value, self.trials))
        return -value*(np.log(self.prob)-np.log(1-self.prob))
    
    def weighted_cost(self, values):
        value = sum(val*w for val, w in values)
#        if (self.prob == 0.0 and value == 0) or\
#                (self.prob == 1.0 and value == self.trials):
#            return 0.0
#        elif self.prob <= 0.0 or self.prob >= 1.0:
#            raise Exception('Impossible event: %f %d %d ' % (self.prob, value, self.trials))
        return -value*(np.log(self.prob)-np.log(1-self.prob))
    
    # the normalizing constant of the distribution
    def null_cost(self):
        return betaln(self.alpha, self.beta)-\
            (self.alpha-1)*np.log(self.prob) - \
            (self.trials+self.beta-1) * np.log(1-self.prob)
    
    def update(self, prob):
        self.prob = prob
    
    def fit(self, values):
        self.prob = (len(values) + self.alpha-1) / (self.trials + self.alpha + self.beta - 2)
#        self.prob = min(max(self.prob, 1e-10), 0.9999)
    
    def weighted_fit(self, values):
        self.prob = (sum(val*w for val, w in values) + self.alpha - 1) / (self.trials + self.alpha + self.beta - 2)
#        self.prob = min(max(self.prob, 1e-10), 0.9999)
    
    def num_args(self):
        return 2
    
    def parse_string_args(self, trials, prob):
        self.trials = int(trials)
        self.prob = float(prob)


class PointExponentialFeature(Feature):
    def __init__(self, rate=1.0):
        self.rate = rate

    def cost(self, values):
        return -sum(np.log(self.rate) + self.rate*val for val in values)
    
    # TODO add/remove edges

    def fit(self, values):
        if not values: return
        values = np.asarray(list(values))
        self.rate = 1.0 / np.mean(values)


class PointCauchyFeature(Feature):
    '''Feature with a (multivariate) Cauchy distribution.'''
    pass


class PointGaussianGammaGammaFeature(Feature):
    '''Gaussian feature with Gamma priors on mean and variance.'''

    def __init__(self, alpha_1, beta_1, alpha_2, beta_2):
        self.alpha_1 = alpha_1
        self.beta_1 = beta_1
        self.alpha_2 = alpha_2
        self.beta_2 = beta_2
        self.mean = alpha_1 / beta_1
        self.var = alpha_2 / beta_2
    
    ### auxiliary methods: likelihood and its derivatives ###
    
    def gamma_logl(self, alpha, beta, value):
        '''Log-likelihood of the Gamma distribution.'''
        return -alpha*np.log(beta) - gammaln(alpha) +\
            (alpha-1)*np.log(value) - beta*value
    
    def statistics(self, values):
        '''Sufficient statistics for the Gaussian distribution.'''
        s_0 = sum(1 for val in values)
        s_1 = sum(val for val in values)
        s_2 = sum(val**2 for val in values)
        return s_0, s_1, s_2
    
    def weighted_statistics(self, values):
        '''Sufficient statistics for the weighted Gaussian distribution.'''
        s_0 = sum(w for val, w in values)
        s_1 = sum(w*val for val, w in values)
        s_2 = sum(w*val**2 for val, w in values)
        return s_0, s_1, s_2

    def logl(self, s_0, s_1, s_2):
        '''Log-likelihood function.'''
        return lambda x: self.gamma_logl(self.alpha_1, self.beta_1, x[0]) +\
            self.gamma_logl(self.alpha_2, self.beta_2, x[1]) -\
            0.5*s_0*np.log(2*np.pi*x[1]) -\
            (0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2) / x[1]
    
    def dlogl_dmean(self, s_0, s_1, s_2):
        '''First derivative of log-likelihood wrt. mean.'''
#        return lambda x: (self.alpha_1-1)/x[0] - self.beta_1 +\
#            (s_1 - s_0*x[0]) / x[1]
        # multiplied by x[1]**2 for better convergence of numeric methods
        return lambda x: (self.alpha_1-1)*x[1] - x[0]*x[1]*self.beta_1 +\
            (s_1 - s_0*x[0]) * x[0]

    def dlogl_dvar(self, s_0, s_1, s_2):
        '''First derivative of log-likelihood wrt. variance.'''
#        return lambda x: (self.alpha_2-1-0.5*s_0)/x[1] - self.beta_2 +\
#            (0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2)/x[1]**2
        # multiplied by x[1]**2 for better convergence of numeric methods
        return lambda x: (self.alpha_2-1-0.5*s_0)*x[1] - self.beta_2*x[1]**2 +\
            (0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2)

    def dlogl_dmean_dmean(self, s_0, s_1, s_2):
        '''Second derivative of log-likelihood wrt. mean.'''
        return lambda x: -(self.alpha_1-1)/x[0]**2 - s_0/x[1]

    def dlogl_dvar_dvar(self, s_0, s_1, s_2):
        '''Second derivative of log-likelihood wrt. variance.'''
        return lambda x: -(self.alpha_2-1-0.5*s_0)/x[1]**2 -\
            (s_2-2*s_1*x[0]+s_0*x[0]**2)/x[1]**3

    def dlogl_dmean_dvar(self, s_0, s_1, s_2):
        '''The mixed second derivative of the log-likelihood.'''
        return lambda x: (s_0*x[0] - s_1) / x[1]**2
    
    def gradient(self, s_0, s_1, s_2):
        '''The gradient of the log-likelihood.'''
        return lambda x: (self.dlogl_dmean(s_0, s_1, s_2)(x),\
            self.dlogl_dvar(s_0, s_1, s_2)(x))
    
    def hessian(self, s_0, s_1, s_2):
        '''The determinant of the Hessian matrix of the log-likelihood.'''
        return lambda x: self.dlogl_dmean_dmean(s_0, s_1, s_2)(x)*\
            self.dlogl_dvar_dvar(s_0, s_1, s_2)(x) - \
            self.dlogl_dmean_dvar(s_0, s_1, s_2)(x)**2
    
    def fit_to_stats(self, s_0, s_1, s_2, max_iter=100000, max_results=10):
        num_iter, num_results = 1, 0
        fun = self.gradient(s_0, s_1, s_2)
        hes = self.hessian(s_0, s_1, s_2)
        mm = self.dlogl_dmean_dmean(s_0, s_1, s_2)
        ll = self.logl(s_0, s_1, s_2)
        result, result_logl = None, None

#        print()
#        print('s_0 =', s_0)
#        print('s_1 =', s_1)
#        print('s_2 =', s_2)
#        print('mean =', s_1/s_0)
#        print('var =', s_2/s_0 - (s_1/s_0)**2)
#        print()

        while num_iter < max_iter and num_results < max_results:
            x = fsolve(fun, (uniform(0, 10), uniform(0, 10)))
            fx = fun(x)
            if x[0] > 0 and x[1] > 0 and fx[0] < 0.001 and fx[1] < 0.001 and\
                    hes(x) > 0.001 and mm(x) < -0.001:
                num_results += 1
#                print(num_iter, x, ll(x), hes(x), mm(x))
                if result_logl is None or ll(x) > result_logl:
                    result = x
                    result_logl = ll(x)
            num_iter += 1
        if result is None:
            mean = s_1/s_0
            var = s_2/s_0 - (s_1/s_0)**2
            print('fit_to_stats: failed to find a result. m=%f, var=%f, s_0=%f' % (mean, var, s_0))
        else:
            self.mean, self.var = result

    ### main methods (for export) ###

    def null_cost(self):
        return -self.gamma_logl(self.alpha_1, self.beta_1, self.mean) -\
            self.gamma_logl(self.alpha_2, self.beta_2, self.var)

    def cost(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        return 0.5*s_0*np.log(2*np.pi*self.var) +\
            (0.5*s_2 - s_1*self.mean + 0.5*s_0*self.mean**2) / self.var

    def weighted_cost(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        return 0.5*s_0*np.log(2*np.pi*self.var) +\
            (0.5*s_2 - s_1*self.mean + 0.5*s_0*self.mean**2) / self.var

    def fit(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        if s_0 <= 0: return
#        if s_0 <= 0 or s_2/s_0 - (s_1/s_0)**2 <= 0: return
        self.fit_to_stats(s_0, s_1, s_2)

    def weighted_fit(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        if s_0 <= 0: return
#        if s_0 <= 0 or s_2/s_0 - (s_1/s_0)**2 <= 0: return
        self.fit_to_stats(s_0, s_1, s_2)

    def num_args(self):
        raise Exception('Not implemented!')


class PointGaussianGaussianGammaFeature(Feature):
    '''Multivariate Gaussian feature with independent coordinates,
       zero-centered prior on means and Gamma prior on variances.'''

    def __init__(self, dim, var_0, alpha, beta):
        self.dim = dim            # TODO multidimensional!
        self.var_0 = var_0
        self.alpha = alpha
        self.beta = beta
        self.mean = np.zeros(dim)
        self.var = alpha/beta*np.ones(dim)
    
    ### auxiliary methods: likelihood and its derivatives ###
    
    def gamma_logl(self, alpha, beta, value):
        '''Log-likelihood of the Gamma distribution.'''
        return -alpha*np.log(beta) - gammaln(alpha) +\
            (alpha-1)*np.log(value) - beta*value
    
    def statistics(self, values):
        '''Sufficient statistics for the Gaussian distribution.'''
        matrix = np.stack(values)
        s_0 = matrix.shape[0]
        s_1 = np.sum(matrix, axis=0)
        s_2 = np.sum(matrix**2, axis=0)
        return s_0, s_1, s_2
    
    def weighted_statistics(self, values):
        '''Sufficient statistics for the weighted Gaussian distribution.'''
        matrix = np.stack([val*w for val, w in values])
        s_0 = matrix.shape[0]
        s_1 = np.sum(matrix, axis=0)
        s_2 = np.sum(matrix**2, axis=0)
        return s_0, s_1, s_2

    def logl(self, s_0, s_1, s_2):
        '''Log-likelihood function.'''
        return lambda x: -0.5*np.log(2*np.pi*self.var_0) -\
            0.5*x[0]**2 / self.var_0 +\
            self.gamma_logl(self.alpha, self.beta, x[1]) -\
            0.5*s_0*np.log(2*np.pi*x[1]) -\
            (0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2) / x[1]
    
    def dlogl_dmean(self, s_0, s_1, s_2):
        '''First derivative of log-likelihood wrt. mean.'''
        # multiplied by x[1]**2 for better convergence of numeric methods
        return lambda x: -x[0]/self.var_0*x[1] + (s_1-s_0*x[0])

    def dlogl_dvar(self, s_0, s_1, s_2):
        '''First derivative of log-likelihood wrt. variance.'''
        # multiplied by x[1]**2 for better convergence of numeric methods
        return lambda x: (self.alpha-1-0.5*s_0)*x[1] - self.beta*x[1]**2 +\
            (0.5*s_2 - s_1*x[0] + 0.5*s_0*x[0]**2)

    def dlogl_dmean_dmean(self, s_0, s_1, s_2):
        '''Second derivative of log-likelihood wrt. mean.'''
        return lambda x: -1/self.var_0 - s_0/x[1]

    def dlogl_dvar_dvar(self, s_0, s_1, s_2):
        '''Second derivative of log-likelihood wrt. variance.'''
        return lambda x: -(self.alpha-1-0.5*s_0)/x[1]**2 -\
            (s_2-2*s_1*x[0]+s_0*x[0]**2)/x[1]**3

    def dlogl_dmean_dvar(self, s_0, s_1, s_2):
        '''The mixed second derivative of the log-likelihood.'''
        return lambda x: (s_0*x[0] - s_1) / x[1]**2
    
    def gradient(self, s_0, s_1, s_2):
        '''The gradient of the log-likelihood.'''
        return lambda x: (self.dlogl_dmean(s_0, s_1, s_2)(x),\
            self.dlogl_dvar(s_0, s_1, s_2)(x))
    
    def hessian(self, s_0, s_1, s_2):
        '''The determinant of the Hessian matrix of the log-likelihood.'''
        return lambda x: self.dlogl_dmean_dmean(s_0, s_1, s_2)(x)*\
            self.dlogl_dvar_dvar(s_0, s_1, s_2)(x) - \
            self.dlogl_dmean_dvar(s_0, s_1, s_2)(x)**2
    
    def fit_coordinate_to_stats(self, i, s_0, s_1, s_2, max_iter=100000, max_results=10):
        num_iter, num_results = 1, 0
        fun = self.gradient(s_0, s_1, s_2)
        hes = self.hessian(s_0, s_1, s_2)
        mm = self.dlogl_dmean_dmean(s_0, s_1, s_2)
        ll = self.logl(s_0, s_1, s_2)
        result, result_logl = None, None

#        print()
#        print('s_0 =', s_0)
#        print('s_1 =', s_1)
#        print('s_2 =', s_2)
#        print('mean =', s_1/s_0)
#        print('var =', s_2/s_0 - (s_1/s_0)**2)
#        print()

        while num_iter < max_iter and num_results < max_results:
            x = fsolve(fun, (uniform(-10, 10), uniform(0, 20)))
            fx = fun(x)
            if x[1] > 0.000001 and fx[0] < 0.001 and fx[1] < 0.001 and\
                    hes(x) > 0.001 and mm(x) < -0.001:
                num_results += 1
                print(x)
#                print(num_iter, x, ll(x), hes(x), mm(x))
                if result_logl is None or ll(x) > result_logl:
                    result = x
                    result_logl = ll(x)
            num_iter += 1
        if result is None:
            mean = s_1/s_0
            var = s_2/s_0 - (s_1/s_0)**2
            print('''fit_coordinate_to_stats: failed to find a result.
                m=%f, var=%f, s_0=%f''' % (mean, var, s_0))
        else:
            self.mean[i], self.var[i] = result
    
    def fit_to_stats(self, s_0, s_1, s_2, max_iter=100000, max_results=10):
        for i in range(self.dim):
            self.fit_coordinate_to_stats(i, s_0, s_1[i], s_2[i], max_iter, max_results)

    ### main methods (for export) ###

    def null_cost(self):
        return sum(-0.5*np.log(2*np.pi*self.var_0)-0.5*self.mean[i]**2/self.var_0 -\
            self.gamma_logl(self.alpha, self.beta, self.var[i]) for i in range(self.dim))

    def cost(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        return sum(0.5*s_0*np.log(2*np.pi*self.var[i]) +\
            (0.5*s_2[i] - s_1[i]*self.mean[i] +\
              0.5*s_0*self.mean[i]**2) /\
            self.var[i] for i in range(self.dim))

    def weighted_cost(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        return sum(0.5*s_0*np.log(2*np.pi*self.var[i]) +\
            (0.5*s_2[i] - s_1[i]*self.mean[i] +\
              0.5*s_0*self.mean[i]**2) /\
            self.var[i] for i in range(self.dim))

    def fit(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        if s_0 <= 0: return
#        if s_0 <= 0 or np.any(s_2/s_0 - (s_1/s_0)**2 <= 0): return
        self.fit_to_stats(s_0, s_1, s_2)

    def weighted_fit(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        if s_0 <= 0: return
#        if s_0 <= 0 or np.any(s_2/s_0 - (s_1/s_0)**2 <= 0): return
        self.fit_to_stats(s_0, s_1, s_2)

    def num_args(self):
        raise Exception('Not implemented!')


class PointGaussianInverseChiSquaredFeature(Feature):
    '''Multivariate Gaussian feature with independent coordinates,
       zero-centered prior on means and Inv-Chi-Sq prior on variances.'''

    def __init__(self, dim, kappa_0, mu_0, nu_0, var_0):
        self.dim = dim
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.var_0 = var_0
        self.mean = kappa_0 * mu_0 * np.ones(self.dim)
        self.var = nu_0 * var_0 * np.ones(self.dim)
    
    def statistics(self, values):
        matrix = np.stack(values)
        shape = (matrix.shape[0],) if self.dim == 1 else (matrix.shape[0], 1)
        s_0 = matrix.shape[0]
        s_1 = np.sum(matrix, axis=0) / s_0
        s_2 = np.sum((matrix-np.tile(s_1, shape))**2, axis=0) / s_0
        if self.dim == 1:
            s_1, s_2 = s_1 * np.ones(self.dim), s_2 * np.ones(self.dim)
        return s_0, s_1, s_2
    
    def weighted_statistics(self, values):
        matrix = np.stack([val for val, w in values])
        weights = np.array([w for val, w in values])
        shape = (matrix.shape[0],) if self.dim == 1 else (matrix.shape[0], 1)
        s_0 = np.sum(weights)
        if s_0 == 0:
            return 0, None, None
        s_1 = np.average(matrix, axis=0, weights=weights)
        s_2 = np.average((matrix-np.tile(s_1, shape))**2, axis=0, weights=weights)
        if self.dim == 1:
            s_1, s_2 = s_1 * np.ones(self.dim), s_2 * np.ones(self.dim)
        return s_0, s_1, s_2
    
    def fit_with_statistics(self, s_0, s_1, s_2):
        self.mean = (self.kappa_0*self.mu_0*np.ones(self.dim) +\
            s_0*s_1) / (self.kappa_0 + s_0)
        self.var = (self.nu_0*self.var_0*np.ones(self.dim) +\
            s_0*s_2 + s_0*self.kappa_0/(s_0+self.kappa_0)*\
                (self.mu_0*np.ones(self.dim)-s_1)**2) /\
            (self.nu_0 + s_0 - 1)
    
    ### coordinate-wise functions ###

    def coordinate_null_cost(self, mean, var):
        return -0.5*np.log(0.5*self.kappa_0/np.pi) + gammaln(0.5*self.nu_0) -\
            0.5*self.nu_0*np.log(0.5*self.nu_0*self.var_0) +\
            (0.5*self.nu_0+1.5)*np.log(var) +\
            0.5/var*(self.nu_0*self.var_0 +\
                self.kappa_0*(self.mu_0-mean)**2)
    
    def coordinate_cost(self, mean, var, s_0, s_1, s_2):
        return 0.5*s_0*np.log(2*np.pi*var) +\
             0.5*s_0/var*(s_2 + (s_1 - mean)**2)

    ### main methods (for export) ###

    def null_cost(self):
        return sum(self.coordinate_null_cost(self.mean[i], self.var[i])
                   for i in range(self.dim))

    def cost(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        if s_0 == 0:
            return 0
        return sum(self.coordinate_cost(\
                                        self.mean[i], self.var[i],
                                        s_0, s_1[i], s_2[i])\
                   for i in range(self.dim))

    def weighted_cost(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        if s_0 == 0:
            return 0
        return sum(self.coordinate_cost(\
                                        self.mean[i], self.var[i],
                                        s_0, s_1[i], s_2[i])\
                   for i in range(self.dim))

    def fit(self, values):
        s_0, s_1, s_2 = self.statistics(values)
        if s_0 > 0:
            self.fit_with_statistics(s_0, s_1, s_2)

    def weighted_fit(self, values):
        s_0, s_1, s_2 = self.weighted_statistics(values)
        if s_0 > 0:
            self.fit_with_statistics(s_0, s_1, s_2)

    def num_args(self):
        raise Exception('Not implemented!')


class PointFeatureSet(FeatureSet):
    '''Creates sets of features according to the program configuration.'''

    def __init__(self):
        self.features = ()
        self.weights = ()
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    @staticmethod
    def new_edge_feature_set(domsize):
        result = PointFeatureSet()
        features = [PointBinomialFeature(domsize, alpha=1.1, beta=1.1)]
        weights = [1.0]
        if settings.WORD_FREQ_WEIGHT > 0.0:
#            features.append(PointGaussianFeature(dim=1))
#            features.append(PointGaussianGammaGammaFeature(2, 1, 2, 10))
            features.append(PointGaussianInverseChiSquaredFeature(
                1, 1, 1, 1, 1))
            weights.append(settings.WORD_FREQ_WEIGHT)
        if settings.WORD_VEC_WEIGHT > 0.0:
#            features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
#            features.append(PointGaussianGaussianGammaFeature(
#                settings.WORD_VEC_DIM, 1, 2, 10))
            features.append(PointGaussianInverseChiSquaredFeature(
                settings.WORD_VEC_DIM, 10, 0, 10, 0.01))
            weights.append(settings.WORD_VEC_WEIGHT)
        result.features = tuple(features)
        result.weights = tuple(weights)
        return result

    @staticmethod
    def new_root_feature_set():
        result = PointFeatureSet()
        features = [StringFeature()]
        weights = [1.0]
        if settings.WORD_FREQ_WEIGHT > 0.0:
            features.append(PointExponentialFeature())
            weights.append(settings.WORD_FREQ_WEIGHT)
        if settings.WORD_VEC_WEIGHT > 0.0:
#            features.append(PointGaussianFeature(dim=settings.WORD_VEC_DIM))
            features.append(PointGaussianInverseChiSquaredFeature(\
                settings.WORD_VEC_DIM, 10, 0, 10, 1))
            weights.append(settings.WORD_VEC_WEIGHT)
        result.features = tuple(features)
        result.weights = tuple(weights)
        return result

    @staticmethod
    def new_rule_feature_set():
        result = PointFeatureSet()
        result.features = (StringFeature(),)
        result.weights = (1.0,)
        return result
    
    def weighted_cost(self, values):
        return sum(w*f.weighted_cost(val) for f, w, val in\
            zip(self.features, self.weights, values))

    def cost(self, values):
        return sum(w*f.cost(val) for f, w, val in\
            zip(self.features, self.weights, values))

    def null_cost(self):
        return sum(w*f.null_cost()\
            for f, w in zip(self.features, self.weights))
    
    def fit(self, values):
        for feature, f_val in zip(self.features, values):
            feature.fit(f_val)

    def weighted_fit(self, values):
        for feature, f_val in zip(self.features, values):
            feature.weighted_fit(f_val)

