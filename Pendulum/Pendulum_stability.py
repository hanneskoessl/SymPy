import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols and values
theta, omega = sp.symbols('theta omega')
g, l = sp.symbols('g l', positive=True, real=True)
values = {g: 9.82, l: 1.0}

# Equations of motion
theta_dot = omega
omega_dot = -g/l * sp.sin(theta)

# Setting up the equations for the equilibrium points and solve them 
equations = [theta_dot, omega_dot]
equilibrium_points = sp.solve(equations, [theta, omega])
print(equilibrium_points)

# Manually add the equilibrium point (-pi, 0)
equilibrium_points.append((-sp.pi, 0))

# Calculate the Jacobian matrix
J = sp.Matrix([theta_dot, omega_dot]).jacobian([theta, omega])
print(sp.latex(J))

# Analyze the stability of each equilibrium point
for point in equilibrium_points:
    J_at_point = J.subs({theta: point[0], omega: point[1]})
    eigenvalues = sp.simplify(J_at_point.eigenvals()).doit()
    print(f"Equilibrium point: {point}")
    print(f"Eigenvalues: {eigenvalues}")
    print("-----------------------------")

# Compute the derivatives of the system on a grid
Omega, Theta  = np.mgrid[-8:8:200j, -1.5*np.pi:1.5*np.pi:200j]
U = sp.lambdify((theta, omega), theta_dot.subs(values), 'numpy')(Theta, Omega)
V = sp.lambdify((theta, omega), omega_dot.subs(values), 'numpy')(Theta, Omega)

# Plot using streamplot
plt.streamplot(Theta, Omega, U, V, density=1.5, linewidth=0.5, cmap=plt.cm.inferno)

# Add the equilibrium points to the plot
equilibrium_thetas, equilibrium_omegas = zip(*equilibrium_points)  # unzip the tuple list
plt.scatter(equilibrium_thetas, equilibrium_omegas, color='red', s=80)  

plt.xlabel('$\\theta$', fontsize=14)
plt.ylabel('$\omega$', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('Phase Portrait of the Pendulum', fontsize=22)
plt.grid(True)
plt.savefig("phase_portrait.png")
plt.show()









