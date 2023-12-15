import numpy as np
import matplotlib.pyplot as plt

# Objective Function Coefficients
c = np.array([4, 7])  # Coefficients for Z = 4x + 7y (Maximize)

# Constraints coefficients
A = np.array([[1, 0], [0, 2], [-1, -1]])  # Coefficients for constraints
b = np.array([4, 6, -3])  # Right-hand side of constraints

# Feasible region plotting
x = np.linspace(0, 5, 100)

# Plot constraints
plt.plot(x, (4 - x)/2, label="x + 2y <= 4")
plt.plot(x, (6 - 2*x)/2, label="2x + y <= 6")
plt.plot(x, -x + 3, label="x + y >= 3")

# Fill feasible region
plt.fill_between(x, 0, (4 - x)/2, where=((4 - x)/2 >= 0), interpolate=True, alpha=0.3)
plt.fill_between(x, 0, (6 - 2*x)/2, where=((6 - 2*x)/2 >= 0), interpolate=True, alpha=0.3)
plt.fill_between(x, -x + 3, 5, where=(-x + 3 >= 0), interpolate=True, alpha=0.3)

# Optimal solution point
plt.scatter(2, 1, color='red', marker='*', s=100, label="Optimal Solution (2, 1)")

# Set labels and title
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Linear Programming Problem - Graphical Solution")

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
# Code
