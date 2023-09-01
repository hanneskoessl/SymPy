import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set up the symbols and equations.
t = sp.symbols('t')
x1, x2, v1, v2 = sp.symbols('x1 x2 v1 v2', cls=sp.Function)
m1, m2, k1, k2, k3 = sp.symbols('m1 m2 k1 k2 k3')

# Matrix representation
A = sp.Matrix([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [-1/m1 * (k1 + k2), k2/m1, 0, 0],
               [k2/m2, -1/m2 * (k2 + k3), 0, 0]])

state = sp.Matrix([x1(t), x2(t), v1(t), v2(t)])
system = state.diff(t) - A * state

# Step 2: Define the constants and initial conditions.
values = {
    m1: 1.,  # kg
    m2: 1.,  # kg
    k1: 4.,  # N/m
    k2: 2.,  # N/m
    k3: 4.   # N/m
}

initial_conditions = {
    x1(0): 1.,
    x2(0): 0.,
    v1(0): 0.,
    v2(0): 0.
}

# Step 3: Numerically evaluate the solutions.
solutions = sp.dsolve(system.subs(values), [x1(t), x2(t), v1(t), v2(t)], ics=initial_conditions)

print(solutions[0])
print(solutions[1])
print(solutions[2])
print(solutions[3])

# Step 4: Use lambdify to convert SymPy expressions to NumPy functions
f_x1 = sp.lambdify(t, solutions[0].rhs, 'numpy')
f_x2 = sp.lambdify(t, solutions[1].rhs, 'numpy')

# Step 5: Plot the results.
end_time = 50  # in seconds
times = np.linspace(0, end_time, 1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Plot for x1(t)
ax1.plot(times, f_x1(times), label='$x_1(t)$', color='blue')
ax1.set_ylabel("Displacement (m)")
ax1.set_title("Displacement of First Mass over Time")
ax1.legend()
ax1.grid(True)

# Plot for x2(t)
ax2.plot(times, f_x2(times), label='$x_2(t)$', color='red')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Displacement (m)")
ax2.set_title("Displacement of Second Mass over Time")
ax2.legend()
ax2.grid(True)

plt.tight_layout()  # Adjust the layout so that the plots don't overlap
plt.show()