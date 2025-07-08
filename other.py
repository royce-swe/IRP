import numpy as np
import numpy.typing as npt
from typing import Callable, List

class A16:
    # Problem parameters
    N = 5
    c = np.array([10, 8, 6, 4, 2], dtype=float)
    K = np.array([5] * N, dtype=float)
    B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
    Q = 100  # Default total output

    @staticmethod
    def paper_solution() -> List[npt.NDArray[np.float64]]:
        value_1 = np.array([10.403965, 13.035817, 15.407354, 17.381556, 18.771308])
        value_2 = np.array([14.050088, 17.798379, 20.907187, 23.111429, 24.132916])
        value_3 = np.array([23.588799, 28.684248, 32.021533, 33.287258, 32.418182])
        value_4 = np.array([35.785329, 40.748959, 42.802485, 41.966381, 38.696846])
        value_5 = np.array([36.912, 41.842, 43.705, 42.665, 39.182])
        return [value_1, value_2, value_3, value_4, value_5]

    @staticmethod
    def price(Q: float) -> float:
        return 5000**(1 / 1.1) * Q**(-1 / 1.1)

    @staticmethod
    def price_derivative(Q: float) -> float:
        return -(1 / 1.1) * 5000**(1 / 1.1) * Q**(-1 / 1.1 - 1)

    @staticmethod
    def cost(q_i: float, i: int) -> float:
        c = A16.c[i]
        B_i = A16.B[i]
        K_i = A16.K[i]
        return c * q_i + (B_i / (B_i + 1)) * K_i**(-1 / B_i) * q_i**((B_i + 1) / B_i)

    @staticmethod
    def objective_functions() -> List[Callable[[float, float, float], float]]:
        return [A16.player_objective_i(i) for i in range(A16.N)]

    @staticmethod
    def player_objective_i(i: int) -> Callable[[float, float, float], float]:
        def obj(q_i: float, total_q: float, d: float) -> float:
            p = A16.price(total_q)
            cost = A16.cost(q_i, i)
            return -(q_i * p - cost - d * q_i)  # Negative for minimization
        return obj

    @staticmethod
    def objective_function_derivatives() -> List[Callable[[float, float, float], float]]:
        return [A16.player_objective_derivative_i(i) for i in range(A16.N)]

    @staticmethod
    def player_objective_derivative_i(i: int) -> Callable[[float, float, float], float]:
        def grad(q_i: float, total_q: float, d: float) -> float:
            dp = A16.price(total_q) + q_i * A16.price_derivative(total_q)
            dc = A16.c[i] + A16.K[i]**(-1 / A16.B[i]) * q_i**(1 / A16.B[i])
            return -(dp - dc - d)
        return grad

    @staticmethod
    def constraints() -> List[Callable[[npt.NDArray[np.float64]], float]]:
        return [A16.shared_constraint]

    @staticmethod
    def constraint_derivatives() -> List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
        return [A16.shared_constraint_derivative]

    @staticmethod
    def g0(q: npt.NDArray[np.float64]) -> float:
        return np.sum(q) - A16.Q

    @staticmethod
    def g0_der(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ones_like(q)


def g0(x):
    x1, x2, x3 = x
    return (sum[x1] + sum[x2] + sum[x3] - 20)[0]