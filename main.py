import numpy as np
from scipy.optimize import minimize


N = 5 #Number of players
c = np.array([10, 8, 6, 4, 2], dtype=float) 
K = np.array([5]*N, dtype=float)
B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
Q = 204.306  #Total quantity   

#Price function from the paper
def price(total_q):
    return 5000**(1/1.1) * total_q**(-1/1.1)

#Derivative of the price function (given in the paper)
def price_derivative(total_q):
    return - (1/1.1) * 5000**(1/1.1) * total_q**(-1/1.1 - 1)

#Cost function from the paper
def cost(q, i):
    return c[i]*q + (B[i]/(B[i]+1)) * K[i]**(-1/B[i]) * q**((B[i]+1)/B[i])

def total_profit(q):
    p = price(np.sum(q))
    return np.sum(q * p - np.array([cost(q[i], i) for i in range(N)]))


# Calculates the derivative/gradient for hard constraint
def gradient(q):
    total_q = np.sum(q)
    g = np.zeros_like(q)
    for i in range(N):
        dp_term = price(total_q) + q[i] * price_derivative(total_q) #partial deriv of the price function
        dc_term = c[i] + K[i]**(-1/B[i]) * q[i]**(1/B[i]) #partial deriv of the cost
        g[i] = -(dp_term - dc_term)
    return g

#Will give something like ([15, 15, 15, 15, 15]) for a consistent starting point
ip = np.full(N, Q / N)

res = minimize(
    fun=lambda q: -total_profit(q),
    x0=ip,
    jac=gradient,
    bounds=[(0, None)] * N,
    constraints=[{'type': 'eq', 'fun': lambda q: np.sum(q) - Q}],
    method='SLSQP',
    options={'ftol': 1e-9, 'maxiter': 200}
)

if not res.success:
    raise ValueError("Optimization failed:", res.message)

q_hard = np.clip(res.x, 0, None)


# print results
print("=== My Results ===")
print("Outputs:", np.round(q_hard, 6))
print("Sum:", np.round(np.sum(q_hard), 6))
print("Profit:", np.round(total_profit(q_hard), 6))

paper_results = np.array([
    [10.403965, 13.035817, 15.407354, 17.381556, 18.771308],
    [14.050088, 17.798379, 20.907187, 23.111429, 24.132916],
    [23.588799, 28.684248, 32.021533, 33.287258, 32.418182],
    [35.785329, 40.748959, 42.802485, 41.966381, 38.696846],
    [36.912, 41.842, 43.705, 42.665, 39.182]])

total = 0
percent_error = 0

print("\n=== The paper's output ===")
print("Outputs:")
match Q:
    case 75:
        print(paper_results[0])
        for x in range(5):
            total += paper_results[0,x]

            difference = q_hard[x] - paper_results[0, x]
            difference /= paper_results[0, x]
            difference = abs(difference)
            difference *= 100
            percent_error += difference
        print('Sum: ', total)
        print("\n======= Error ========\n", "Percent Error: ", round(percent_error/5, 3), "%" )

    case 100:
        print(paper_results[1])
        for x in range(5):
            total += paper_results[1,x]
        
            difference = q_hard[x] - paper_results[1, x]
            difference /= paper_results[1, x]
            difference = abs(difference)
            difference *= 100
            percent_error += difference
        print('Sum: ', total)
        print("\n======= Error ========\n", "Percent Error: ", round(percent_error/5, 3), "%" )

    case 150:
        print(paper_results[2])
        for x in range(5):
            total += paper_results[2,x]

            difference = q_hard[x] - paper_results[2, x]
            difference /= paper_results[2, x]
            difference = abs(difference)
            difference *= 100
            percent_error += difference
        print('Sum: ', total)
        print("\n======= Error ========\n", "Percent Error: ", round(percent_error/5, 3), "%" )

    case 200:
        print(paper_results[3])
        for x in range(5):
            total += paper_results[3,x]

            difference = q_hard[x] - paper_results[3, x]
            difference /= paper_results[3,x]
            difference = abs(difference)
            difference *= 100
            percent_error += difference
        print('Sum: ', total)
        print("\n======= Error ========\n", "Percent Error: ", round(percent_error/5, 3), "%" )

    case 204.306:
        print(paper_results[4])
        for x in range(5):
            total += paper_results[4,x]

            difference = q_hard[x] - paper_results[4, x]
            difference /= paper_results[4, x]
            difference = abs(difference)
            difference *= 100
            percent_error += difference
        print('Sum: ', total)
        print("\n======= Error ========\n", "Percent Error: ", round(percent_error/5, 3), "%" )

    case _:
        print("Total output doesn't match any of the paper's.")



#print(f'\n {random_initial()}')


 
'''
if Q == 75:
    paper = [paper_results][0]
elif Q == 100:
    paper = paper_results[1]
elif Q == 150:
    paper = paper_results[2]
elif Q == 200:
    paper = paper_results[3]
elif Q == 204.306:
    paper = paper_results[4]
else:
    paper = np.zeros(5)

x = np.arange(N)

plt.figure(figsize=(9, 5))
plt.bar(x - 0.2, paper, width=0.4, label='Murphy et al. (1982)', color='lightgray', edgecolor='black')
plt.bar(x + 0.2, my_results, width=0.4, label='Your Model', color='cornflowerblue', edgecolor='black')
plt.xticks(x, [f"Firm {i+1}" for i in range(N)])
plt.ylabel("Output")
plt.title(f"Optimized Output vs Paper Reference (Q = {Q})")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
'''
