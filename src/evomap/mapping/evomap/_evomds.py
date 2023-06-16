"""
EvoMap, implemented for Stress-Based MDS.
"""

import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._core import EvoMap
from ._core import _evomap_cost_function, _get_positions_for_period
from evomap.mapping._core import DivergingGradientError

class EvoMDS(EvoMap):

    def __init__(
        self,
        alpha = 0,                      
        p = 1,                          
        n_dims = 2, 
        metric = True,  
        n_iter = 2000,  
        n_iter_check = 50,
        init = None, 
        mds_type = 'absolute',
        verbose = 0, 
        input_type = 'distance', 
        maxhalves = 5, 
        tol = 1e-3,  
        n_inits = 1, 
        step_size = 1,
        max_tries = 5
    ):

        super().__init__(alpha = alpha, p = p)
        self.n_dims = n_dims
        self.metric = metric
        self.n_iter = n_iter
        self.n_iter_check = n_iter_check
        self.init = init
        self.verbose = verbose
        self.mds_type = mds_type
        self.input_type = input_type
        self.maxhalves = maxhalves
        self.tol = tol
        self.n_inits = n_inits
        self.step_size = step_size
        self.max_tries = max_tries
        self.method_str = "EvoMDS"

    def fit(self, Xs, inclusions = None):
        self.fit_transform(Xs, inclusions)
        return self

    def fit_transform(self, Xs, inclusions = None):
        from evomap.mapping._core import gradient_descent_line_search
        from evomap.mapping._mds import _normalized_stress_function

                
        # Check and prepare input data
        super()._validate_input(Xs, inclusions)

        n_periods = len(Xs)
        if self.input_type == 'distance':
            Ds = Xs
        elif self.input_type == 'vector':
            Ds = []
            for t in range(n_periods):
                Ds.append(cdist(Xs[t],Xs[t]))
        else:
            raise ValueError("Input type should be 'distance' or 'vector', not {0}".format(self.input_type))

        W = super()._calc_weights(Xs)
        self.W_ = W

        n_samples = Ds[0].shape[0]
        n_dims = self.n_dims

        if not self.init is None:
            self.n_inits = 1

        best_cost = np.inf
        for i in range(self.n_inits):
            if self.verbose > 0:
                print("[{0}] Initialization {1}/{2}".format(self.method_str,i+1, self.n_inits))
                
            init = super()._initialize(Xs)

            # Set gradient descent arguments
            opt_args = {
                'init': init,
                'method_str': self.method_str,
                'n_iter': self.n_iter,
                'n_iter_check': self.n_iter_check,
                'step_size': self.step_size,
                'maxhalves': self.maxhalves,
                'min_grad_norm': self.tol,
                'verbose': self.verbose,
                'kwargs': {
                    'static_cost_function': _normalized_stress_function, 
                    'Ds': Ds,
                    'alpha': self.alpha, 
                    'p': self.p,
                    'inclusions': inclusions,
                    'weights' : W,
                    'static_cost_kwargs': {'mds_type': self.mds_type}}
            }

            for ith_try in range(self.max_tries):
                try:
                    Y, cost = gradient_descent_line_search(_evomap_cost_function, **opt_args)                        
                    break
                except DivergingGradientError:
                    print("[{0}] Adjusting step sizes..".format(self.method_str))
                    self.step_size /= 2
                    opt_args.update({'step_size': self.step_size})

                if ith_try == self.max_tries -1:
                    print("[{0}] ERROR: Gradient descent failed to converge.".format(self.method_str))
                    return -1

            if cost < best_cost:
                Ys = []
                for t in range(n_periods):
                    Ys.append(_get_positions_for_period(Y, n_samples, t))
                self.Ys_ = Ys
                self.cost_ = cost
                self.cost_static_ = super()._calc_static_cost(
                    Xs = Ds, 
                    Y_all_periods= Y, 
                    inclusions = inclusions, 
                    static_cost_function=_normalized_stress_function,
                    static_cost_kwargs = {'mds_type': self.mds_type})
                
                self.cost_static_avg_ = self.cost_static_ / n_periods    
                best_cost = cost

        return self.Ys_