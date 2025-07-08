import numpy as np
from scipy.optimize import basinhopping

# --- Problem Setup ---
N = 5  # number of firms
c = np.array([10, 8, 6, 4, 2], dtype=float)
K = np.array([5] * N, dtype=float)
B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
Q = 200  # market capacity

# --- Price and Cost Functions ---
def price(total_q):
    return 5000 ** (1 / 1.1) * total_q ** (-1 / 1.1)

def price_derivative(total_q):
    return - (1 / 1.1) * 5000 ** (1 / 1.1) * total_q ** (-1 / 1.1 - 1)

def cost(q, i):
    return c[i] * q + (B[i] / (B[i] + 1)) * K[i] ** (-1 / B[i]) * q ** ((B[i] + 1) / B[i])

def cost_derivative(q, i):
    return c[i] + K[i] ** (-1 / B[i]) * q ** (1 / B[i])

#Gradient of Modified Profit (including d)
def firm_gradient(q_firms, i):
    total_q = np.sum(q_firms)
    dp = price_derivative(total_q)
    p = price(total_q)
    return p + q_firms[i] * dp - cost_derivative(q_firms[i], i)

# Energy Function: Gradient Squared Norm + Dual Gradient 
def energy(x):
    q = x[:-1]  # firm quantities
    d = x[-1]   # dual variable

    grad_firms = np.array([firm_gradient(q, i) - d for i in range(N)])
    constraint_violation = np.sum(q) - Q
    grad_dual = -constraint_violation

    grad_full = np.append(grad_firms, grad_dual)
    return np.sum(grad_full ** 2) #returns square of the

#start point
x0 = np.array([0,0,0,0,0,1])
bounds = [(0, None)] * (N + 1)  # non-negativity for all variables

res = basinhopping(
    func=energy,
    x0=x0,
    minimizer_kwargs={
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {'ftol': 1e-9, 'maxiter': 500}
    },
    niter=200,
    stepsize=0.5,
    seed=42,
    disp=True
)

#Results 
q_opt = res.x[:-1]
d_opt = res.x[-1]

print("\n=== Final Results ===")
print("Quantities:", np.round(q_opt, 6))
print("Sum:", np.sum(q_opt))
print("Dual variable (d):", d_opt)
print("Energy (should be ~0):", energy(res.x))

# This just is to compare results and make sure its good
def compare_to_paper_results(my_results_firm, Q):
    paper_results = np.array([
        [10.403965, 13.035817, 15.407354, 17.381556, 18.771308],
        [14.050088, 17.798379, 20.907187, 23.111429, 24.132916],
        [23.588799, 28.684248, 32.021533, 33.287258, 32.418182],
        [35.785329, 40.748959, 42.802485, 41.966381, 38.696846],
        [36.912, 41.842, 43.705, 42.665, 39.182]
    ])

    Q_match_map = {
        75: 0,
        100: 1,
        150: 2,
        200: 3,
        204.306: 4
    }

    print("\n=== The paper's output ===")
    idx = Q_match_map.get(Q, None)

    if idx is None:
        print("Total output doesn't match any of the paper's.")
        return

    paper = paper_results[idx]
    total = np.sum(paper)
    percent_error = 0.0

    print("Outputs:", paper)
    for i in range(len(my_results_firm)):
        diff = abs(my_results_firm[i] - paper[i]) / paper[i] * 100
        percent_error += diff

    print("Sum:", total)
    print("\n======= Error ========")
    print("Percent Error:", round(percent_error / len(my_results_firm), 7), "%")

# After basinhopping runs and you get res.x:
my_results = np.clip(res.x, 0, None)

# exclude dual variable and compare results
q_opt = my_results[:-1]
compare_to_paper_results(q_opt, Q)

