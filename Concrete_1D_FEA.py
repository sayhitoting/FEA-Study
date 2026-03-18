import numpy as np

# -------------------------
# 1. Problem definition
# -------------------------
E = 210e9      # Young's modulus (Pa)
A = 0.01       # Cross-sectional area (m^2)
L = 2.0        # Total length (m)

num_elements = 2
num_nodes = num_elements + 1
element_length = L / num_elements

# -------------------------
# 2. Initialize matrices
# -------------------------
K_global = np.zeros((num_nodes, num_nodes))
F_global = np.zeros(num_nodes)

# Apply force at the last node
F_global[-1] = 1000  # 1000 N

# -------------------------
# 3. Element stiffness matrix
# -------------------------
k_local = (E * A / element_length) * np.array([
    [1, -1],
    [-1, 1]
])

# -------------------------
# 4. Assembly
# -------------------------
for i in range(num_elements):
    K_global[i:i+2, i:i+2] += k_local

# -------------------------
# 5. Apply boundary conditions
# -------------------------
# Node 1 is fixed → displacement = 0
K_reduced = K_global[1:, 1:]
F_reduced = F_global[1:]

# -------------------------
# 6. Solve system
# -------------------------
u_reduced = np.linalg.solve(K_reduced, F_reduced)

# Add back the fixed displacement
u = np.zeros(num_nodes)
u[1:] = u_reduced

# -------------------------
# 7. Compute element forces
# -------------------------
forces = []
for i in range(num_elements):
    u_element = u[i:i+2]
    f_element = k_local @ u_element
    forces.append(f_element)

# -------------------------
# 8. Output
# -------------------------
print("Displacements:", u)
print("Element forces:", forces)