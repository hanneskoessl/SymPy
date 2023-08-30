import sympy as sp

# Define symbolic variables and functions
t = sp.symbols('t')                         
m, l, g = sp.symbols('m l g', positive=True, real=True)  
theta = sp.Function('theta')(t)

# Kinetic Energy (T)
v = l * theta.diff(t)
T = 0.5 * m * v**2

# Potential Energy (V)
h = l * (1 - sp.cos(theta))
V = m * g * h

# Lagrangian (L)
L = T - V

# Euler-Lagrange Equation
# Partial derivatives
dL_dtheta = L.diff(theta)
dL_dthetadot = L.diff(theta.diff(t))
euler_lagrange = dL_dthetadot.diff(t) - dL_dtheta
print(sp.latex(euler_lagrange))

# Divide the equation by m*l**2
divided_euler_lagrange = sp.simplify(euler_lagrange/(m*l**2))

# Display the differential equation
equation_of_motion = sp.nsimplify(sp.Eq(divided_euler_lagrange, 0))
print(equation_of_motion)