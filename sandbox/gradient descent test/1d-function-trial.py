import numpy as np

# Testing gradient descent to find the minimum of a function
# Requirements: a function h(x) which is differentiable for all x in its domain, and which is also convex.
# Convex function:
# A function where two points on its curve connected by a line segment is always greater than the function at those same points,
# or a function in which its second derivative is always strictly greater than 0.

# example function: f(x) = x^2 - 5x + 3
def f(x):
    return x**2 - 5*x + 3

# Gradient function for the above function
# What is the gradient? It's the first derivative at a selected point. For a multivariable function, it's a vector of partial derivatives.
def gradient(x):
    return 2 * x - 5

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
start = 5

# Learning rate
learn_rate = 0.1

# Number of iterations
iterations = 50

# Convergence tolerance
tolerance = 1e-6

# Run the algorithm to search for the minimum value
result = gradient_descent(gradient, start, learn_rate, iterations, tolerance)
print("x: ", result, "f(x): ", f(result))