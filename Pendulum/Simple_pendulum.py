import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols and functions
t, l, g = sp.symbols('t l g', positive=True, real=True)
theta_t = sp.Function('theta')(t)

# Define the linearized equation of motion for a pendulum
eq = sp.Eq(theta_t.diff(t, t) + g/l * theta_t, 0)

# Solve the linearized version
solution_linear = sp.dsolve(eq)
print("General solution:", solution_linear)
print(sp.latex(solution_linear))

# Substitute specific values for l and g 
l_value = 1.0 
g_value = 9.81
param_sol = solution_linear.subs({l: l_value, g: g_value}).rhs
print("Parametrized solution:", param_sol)
print(sp.latex(sp.Eq(theta_t, param_sol)))

# Determine the constants of integration based on the initial conditions
init_conds = {'theta_t(0)': 0.26, 'dtheta_dt_t(0)': 0}
constants = sp.solve((param_sol.subs(t, 0) - init_conds['theta_t(0)'], 
                      param_sol.diff(t).subs(t, 0) - init_conds['dtheta_dt_t(0)']))

result = sp.simplify(param_sol.subs(constants))
print("Specific solution with initial conditions:", result)
print(sp.latex(sp.Eq(theta_t,result)))

# Convert the symbolic solution to a numerical function
f = sp.lambdify(t, result, "numpy")

# Generate time values and corresponding pendulum positions
times = np.linspace(0, 10, 400)  # 10 seconds, 400 points
theta_values = f(times)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(times, theta_values, label='Linearized Solution')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel(r'$\theta(t)$', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('Linearized Pendulum Motion Over Time', fontsize=24)
plt.grid(True)
plt.savefig("pendulum.png")
plt.show()