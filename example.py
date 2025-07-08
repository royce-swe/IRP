import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A3:

    @staticmethod
    def paper_solution():
        value_1 = [-9.38046562696258, -9.12266997083581, -9.99322817120517, 8.39034789088544, 1.16385412687962, 8.05039533464000] 
        value_2 = [-0.38046562696294, -0.12266997083590, -0.99322817120634, 8.39834789880558, 1.16385412688026, 0.05039533464023]
        value_3 = [-8.38046562696275, -8.12265997083484, -0.99322817120582, 8.39834789880555, 1.16385412688162, 0.05039533463988]
        return [value_1, value_2, value_3,]

    @staticmethod
    def define_players():
        player_vector_sizes = [3, 2, 2]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [2], [3]]
        bounds = [(-10, 19), (-10, 18), (-18, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 100),
        (0, 100), (0, 100), (0, 100)]
        bounds_training = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 100),
        (0, 100), (0, 100), (0, 100)]

        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A3.obj_func_1, A3.obj_func_2, A3.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A3.obj_func_der_1, A3.obj_func_der_2, A3.obj_func_der_3]
    
    @staticmethod
    def constraints():
        return [A3.g0, A3.g1, A3.g2, A3.g3]
    
    @staticmethod 
    def constraint_derivatives():
        return [A3.g0_der, A3.g1_der, A3.g2_der, A3.g3_der]
    
    A1 = np.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]])
    A2 = np.array([[11, -1], [[-1, 9]]])
    A3 = np.array([[48, 39], [39, 53]])

    @staticmethod
    def obj_func():
        x: npt.NDArray[np.float64]


