'''
-Should be putting in the energy function into basin hopping not the profit function
-energy function value should return very close to 0
-energy function should only take in the input x and it should calculate the gradient inside of it
-if i plug values into the energy function I should get 0 or very close to it like x^-5 do this because even tho I am getting the results the paper got we dont know if they have the best results we may have better
-need to factor in dual players & derivative of dual players with respect to d is h'(x) = -g(x)
'''

import numpy as np
from scipy.optimize import basinhopping

N = 5 #Number of players
c = np.array([10, 8, 6, 4, 2], dtype=float) 
K = np.array([5]*N, dtype=float)
B = np.array([1.2, 1.1, 1.0, 0.9, 0.8], dtype=float)
Q = 75  #Total quantity   

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


# Calculates the derivative/gradient 
def gradient(q):
    total_q = np.sum(q)
    g = np.zeros_like(q)
    for i in range(N):
        dp_value = price(total_q) + q[i] * price_derivative(total_q) #deriv of the price function
        dc_value = c[i] + K[i]**(-1/B[i]) * q[i]**(1/B[i]) #deriv of the cost
        g[i] = -(dp_value - dc_value) # dual player 
    return g

#Will give something like ([15, 15, 15, 15, 15]) for a consistent starting point
ip = np.full(N, Q / N)

# dual player's objective function h(x) = -d*g(x)
# derivative of h =-g(x)
bounds = [(0, None)] * N
constraints = [{'type': 'eq', 'fun': lambda q: np.sum(q) - Q}]

minimizer = {
    'method': 'SLSQP',
    'jac': gradient,
    'constraints': constraints,
    'bounds': bounds,
    'options': {'ftol': 1e-9, 'maxiter': 200}
}

res = basinhopping(
    func=lambda q: -total_profit(q),
    x0=ip,
    minimizer_kwargs=minimizer,
    niter=200,  # Number of "basin hops"
    stepsize=0.001,  # Max jump in each step
    seed=42,
    disp=True
)

my_results = np.clip(res.x, 0, None)


# print results
print("=== My Results ===")
print("Outputs:", np.round(my_results, 6))
print("Sum:", np.round(np.sum(my_results), 6))
print("Profit:", np.round(total_profit(my_results), 6))

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

            difference = my_results[x] - paper_results[0, x]
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
        
            difference = my_results[x] - paper_results[1, x]
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

            difference = my_results[x] - paper_results[2, x]
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

            difference = my_results[x] - paper_results[3, x]
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

            difference = my_results[x] - paper_results[4, x]
            difference /= paper_results[4, x]
            difference = abs(difference)
            difference *= 100
            percent_error += difference
        print('Sum: ', total)
        print("\n======= Error ========\n", "Percent Error: ", round(percent_error/5, 3), "%" )

    case _:
        print("Total output doesn't match any of the paper's.")



#print(f'\n {random_initial()}')


 

