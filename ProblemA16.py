import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A16:

    @staticmethod
    def paper_solution():
        value_1 = [10.403965, 13.035817, 15.407354, 17.381556, 18.771308]
        value_2 = [14.050088, 17.798379, 20.907187, 23.111429, 24.132916]
        value_3 = [23.588799, 28.684248, 32.021533, 33.287258, 32.418182]
        value_4 = [35.785329, 40.748959, 42.802485, 41.966381, 38.696846]
        value_5 = [36.912, 41.842, 43.705, 42.665, 39.182]
        return [value_1, value_2, value_3, value_4, value_5]
    
    @staticmethod
    def define_players(
            N = 5,
            c = np.array([10, 8, 6, 4, 2], dtype=float),
            K = np.array([5] * N, dtype=float),
            B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float),
            Q = 100):  # Default total quantity
        return N, c, K, B, Q

