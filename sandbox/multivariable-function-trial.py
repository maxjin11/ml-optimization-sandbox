import numpy as np

# Testing gradient descent to find the minimum of a multivariable function
# Requirements: a function h(x, y) which is differentiable for all points (x, y) in its domain, and which is also convex.

# example function: f(x, y) = 0.5x^2 + y^2 + 1
# note that this function is bivariate.
def f(r):
    return 0.5*r[0]**2 + r[1]**2 + 1

# Gradient function for the above function
# What is the gradient? It's the first derivative at a selected point. For a multivariable function, it's a vector of partial derivatives.
def gradient(r):
    return np.array([r[0], 2*r[1]])

# Gradient descent optimization
def gradient_descent(gradient, start, learn_rate, iterations, tolerance):
    vector = start

    # for each iteration:
    for _ in range(iterations):
        # calculate the size of the "step" we want to take, scale the gradient by the learn rate
        # as we approach the local minima, these steps get smaller, scaling with the gradient value.
        diff = -learn_rate * gradient(vector)

        # if the size of the step is no longer greater than the tolerance, exit the process, since the size of the step of is satisfactory.
        if np.all(np.abs(diff) <= tolerance):
            break
        
        # take the "step" and get cloesr to the local minima
        vector += diff

    # return the x-value which returns the local minimum
    return vector

# Initial point
start = np.array([4.0, 2.0])

# Learning rate
learn_rate = 0.1

# Number of iterations
iterations = 50

# Convergence tolerance
tolerance = 1e-6

# Run the algorithm to search for the minimum value
result = gradient_descent(gradient, start, learn_rate, iterations, tolerance)
print("(x, y): ", result, "f(x, y): ", f(result))