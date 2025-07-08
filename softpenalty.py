import numpy as np
from scipy.optimize import minimize

N = 5
c = np.array([10, 8, 6, 4, 2], dtype=float)
K = np.array([5]*N, dtype=float)
B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
Q = 200      
d = 1e4        # penalty weight for soft constraint gpt added this

def price(total_q):
    return 5000**(1/1.1) * total_q**(-1/1.1)

def price_derivative(total_q):
    return - (1/1.1) * 5000**(1/1.1) * total_q**(-1/1.1 - 1)

def cost(q, i):
    return c[i]*q + (B[i]/(B[i]+1)) * K[i]**(-1/B[i]) * q**((B[i]+1)/B[i])

def total_profit(q):
    p = price(np.sum(q))
    return np.sum(q * p - np.array([cost(q[i], i) for i in range(N)]))

'''
# Calculates the derivative/gradient for hard constraint
def grad_hard(q):
    total_q = np.sum(q)
    g = np.zeros_like(q)
    for i in range(N):
        dp_term = price(total_q) + q[i] * price_derivative(total_q) #partial of the price function
        dc_term = c[i] + K[i]**(-1/B[i]) * q[i]**(1/B[i]) #partial of the cost
        g[i] = -(dp_term - dc_term)
    return g
'''

# Soft constraint objective and gradient || Gpt said the hard constraint makes more sense & it was more accurate
def objective_soft(q):
    profit = total_profit(q)
    violation = max(np.sum(q) - Q, 0.0)
    penalty = d * violation**2
    return -profit + penalty

def grad_soft(q):
    total_q = np.sum(q)
    grad = np.zeros_like(q)
    for i in range(N):
        dp_term = price(total_q) + q[i] * price_derivative(total_q)
        dc_term = c[i] + K[i]**(-1/B[i]) * q[i]**(1/B[i])
        grad[i] = -(dp_term - dc_term)
    if total_q > Q:
        grad += 2 * d * (total_q - Q)
    return grad


# Initial point/guess
def random_initial():
    x0 = np.clip(np.random.rand(N) * Q, 0, None)
    return x0 * (Q / np.sum(x0))

'''
# Hard Constraint \w multiple restarts 
best_hard = None
for _ in range(10):
    x0 = random_initial()
    res = minimize(
        fun=lambda q: -total_profit(q),
        x0=x0,
        jac=grad_hard,
        bounds=[(0, None)]*N,
        constraints=[{'type': 'eq', 'fun': lambda q: np.sum(q) - Q}],
        method='SLSQP',
        options={'ftol': 1e-9, 'maxiter': 200}
    )
    if res.success:
        if best_hard is None or res.fun < best_hard.fun:
            best_hard = res

q_hard = np.clip(best_hard.x, 0, None)
'''

# Soft Constraint /w multiple restarts
best_soft = None
for _ in range(10):
    x0 = random_initial()
    res = minimize(
        fun=objective_soft,
        x0=x0,
        jac=grad_soft,
        bounds=[(0, None)]*N,
        method='SLSQP',
        options={'ftol': 1e-9, 'maxiter': 200}
    )
    if res.success:
        if best_soft is None or res.fun < best_soft.fun:
            best_soft = res

q_soft = np.clip(best_soft.x, 0, None)

'''
# print results
print("=== Hard Constraint Results ===")
print("Outputs:", np.round(q_hard, 6))
print("Sum:", np.round(np.sum(q_hard), 6))
print("Profit:", np.round(total_profit(q_hard), 6))
'''

print("\n=== Soft Constraint Results ===")
print("Outputs:", np.round(q_soft, 6))
print("Sum:", np.round(np.sum(q_soft), 6))
print("Profit:", np.round(total_profit(q_soft), 6))

paper_res = np.array([
    [10.403965, 13.035817, 15.407354, 17.381556, 18.771308],
    [14.050088, 17.798379, 20.907187, 23.111429, 24.132916],
    [23.588799, 28.684248, 32.021533, 33.287258, 32.418182],
    [35.785329, 40.748959, 42.802485, 41.966381, 38.696846],
    [36.912, 41.842, 43.705, 42.665, 39.182]])
total = 0

print("\n=== The paper's output ===")
print("Outputs:")
match Q:
    case 75:
        print(paper_res[0])
        for x in range(5):
            total += paper_res[0,x]
        print('Sum: ', total)

    case 100:
        print(paper_res[1])
        for x in range(5):
            total += paper_res[1,x]
        print('Sum: ', total)

    case 150:
        print(paper_res[2])
        for x in range(5):
            total += paper_res[2,x]
        print('Sum: ', total)

    case 200:
        print(paper_res[3])
        for x in range(5):
            total += paper_res[3,x]
        print('Sum: ', total)

    case 204.306:
        print(paper_res[4])
        for x in range(5):
            total += paper_res[4,x]
        print('Sum: ', total)

    case _:
        print("Total output doesn't match any of the paper's.")


 

