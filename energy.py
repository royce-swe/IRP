import numpy as np
from scipy.optimize import basinhopping

N = 5 #Number of players
c = np.array([10, 8, 6, 4, 2], dtype=float) 
K = np.array([5]*N, dtype=float)
B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
Q = 100  #Total quantity   


#Price function from the paper
def price(total_q):
    return 5000**(1/1.1) * total_q**(-1/1.1)

#Derivative of the price function
def price_derivative(total_q):
    return - (1 / 1.1) * 5000 ** (1 / 1.1) * total_q ** (-1 / 1.1 - 1)

#Cost function from the paper
def cost(q, i):
    return c[i]*q + (B[i]/(B[i]+1)) * K[i]**(-1/B[i]) * q**((B[i]+1)/B[i])

def total_profit(q):
    p = price(np.sum(q))
    return np.sum(q * p - np.array([cost(q[i], i) for i in range(N)]))


# Calculates the derivative/gradient 
def gradient(x):
    q = x[:-1]
    d = x[-1]

    constraint_violation = np.sum(q) - Q
    dual_gradient = -constraint_violation

    total_q = np.sum(q)
    g = np.zeros_like(q)
    for i in range(N):
        dp_value = price(total_q) + q[i] * price_derivative(total_q)
        dc_value = c[i] + K[i]**(-1/B[i]) * q[i]**(1/B[i])
        g[i] = -(dp_value - dc_value - d)
    g = np.append(g, dual_gradient)
    return g


def energy(x):
    q = x[:-1]
    d = x[-1]
    g = gradient(x)

    E_val = 0.0
    lb = 0.0
    ub = Q

    for i in range(N):
        if g[i] <= 0:
            E_val += (ub - q[i]) * np.log1p(-g[i]) 
        else:
            E_val += (q[i] - lb) * np.log1p(g[i])
    
    dual_lb = 0.0
    dual_ub = 1000.0

    if g[N] <= 0:
        E_val += (dual_ub - d) * np.log1p(-g[N])
    else:
        E_val += (d - dual_lb) * np.log1p(g[N])

    return E_val


#Will give something like ([15, 15, 15, 15, 15]) for a consistent starting point
ip = np.array([0,0,0,0,0,1]); zero_start = True
#ip = np.append(np.full(N, Q / N), 1.0); zero_start = False

# dual player's objective function h(x) = -d*g(x)
# derivative of h =-g(x)
bounds = [(0, None)] * (N + 1)

res = basinhopping(
    func=energy,
    x0=ip,
    minimizer_kwargs={
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {'ftol': 1e-9, 'maxiter': 500}
    },
    niter=1000,
    stepsize=0.5,
    seed=42,
    disp=True
)

my_results = np.clip(res.x, 0, None)


# print results
x_opt = np.clip(res.x, 0, None)
q_opt = x_opt[:-1]
d_opt = x_opt[-1]

print("\n=== Final Results ===")
print("Quantities:", np.round(q_opt, 6))
print("Sum:", np.round(np.sum(q_opt), 6))
print("Dual variable (d):", round(d_opt, 6))
print("Profit:", round(total_profit(q_opt), 6))
energy_val = energy(x_opt)
if isinstance(energy_val, np.ndarray):
    energy_val = energy_val.item()  # safely extract scalar from 1-element array
print("Energy (should be ~0):", round(energy_val, 10))


# === Comparison with Paper ===

def compare_to_paper_results(my_results_firm, Q):
    paper_results = np.array([
        [10.403965, 13.035817, 15.407354, 17.381556, 18.771308],
        [14.050088, 17.798379, 20.907187, 23.111429, 24.132916],
        [23.588799, 28.684248, 32.021533, 33.287258, 32.418182],
        [35.785329, 40.748959, 42.802485, 41.966381, 38.696846],
        [36.912, 41.842, 43.705, 42.665, 39.182]
    ])

    Q_map = {
        75: 0,
        100: 1,
        150: 2,
        200: 3,
        204.306: 4
    }

    idx = Q_map.get(Q)
    if idx is None:
        print("Total output doesn't match any of the paper's.")
        return

    paper = paper_results[idx]
    total = np.sum(paper)
    percent_error = np.mean(np.abs((my_results_firm - paper) / paper)) * 100

    print("\n=== The paper's output ===")
    print("Outputs:", paper)
    print("Sum:", round(total, 6))
    print("\n======= Error ========")
    print("Percent Error:", round(percent_error, 5), "%")

compare_to_paper_results(q_opt, Q)
print("Optimization Started from Zero" if zero_start else f"Optimization Started from {ip}")


#75: Good, max = 500, niter = 1000, stepsize = 0.5, 0.0004% / 5.74283e-6, 0 start
#100: Good, max = 500, niter = 1000, stepsize = 0.5, 3e-5% Error/ 5.678e-7 energy, 0 start 
#150: Good, max = 500, niter = 1000, stepsize = 0.5/ max = 3000, niter = 1500, stepsize = 0.05, 0.00019% Error / 1.124e-5, 0 start
#200: Decent, max = 500, niter = 1000, stepsize = 1.5, 0.0004% Error/ 0.005 energy, non 0 start
#204.306: Decent, max = 500, niter = 1000, stepsize = 1.5, 0.02761% Error/ 0.008 energy, non 0 start