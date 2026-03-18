import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# MATERIAL / SECTION
# -------------------------

E = 30e9
A = 0.01
A_reinf = 0.03  # reinforced area

# -------------------------
# GEOMETRY (RECTANGULAR TRUSS)
# -------------------------

nx = 6   # number of nodes along length
ny = 2   # 2 layers (bottom + top)
L = 10
H = 2

nodes = []
for j in range(ny):
    for i in range(nx):
        x = i * (L/(nx-1))
        y = j * H
        nodes.append([x, y])
nodes = np.array(nodes)

# -------------------------
# ELEMENT CONNECTIVITY
# -------------------------

elements = []

def node(i, j):
    return j*nx + i

# Horizontal members
for j in range(ny):
    for i in range(nx-1):
        elements.append([node(i,j), node(i+1,j)])

# Vertical members
for i in range(nx):
    elements.append([node(i,0), node(i,1)])

# Diagonals
for i in range(nx-1):
    elements.append([node(i,0), node(i+1,1)])
    elements.append([node(i+1,0), node(i,1)])

elements = np.array(elements)

# -------------------------
# DOF SETUP
# -------------------------

n_nodes = len(nodes)
dof = 2 * n_nodes

# -------------------------
# LOADS (TOP NODES)
# -------------------------

F = np.zeros(dof)

for i in range(nx):
    n = node(i,1)  # top row
    F[2*n + 1] = -10000  # downward load

# -------------------------
# SUPPORTS
# -------------------------

supports = [
    (node(0,0), 0), (node(0,0), 1),  # left fixed
    (node(nx-1,0), 1)                # right roller
]

# -------------------------
# TRUSS STIFFNESS
# -------------------------

def element_stiffness(n1, n2, A):
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]
    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    c = (x2-x1)/L
    s = (y2-y1)/L

    k = (E*A/L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    return k, L, c, s

# -------------------------
# SOLVER
# -------------------------

def solve_truss(A_values):
    K = np.zeros((dof, dof))

    for e, (n1, n2) in enumerate(elements):
        k, _, _, _ = element_stiffness(n1, n2, A_values[e])
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k[i,j]

    K_mod = K.copy()
    F_mod = F.copy()

    for (n, d) in supports:
        idx = 2*n + d
        K_mod[idx,:] = 0
        K_mod[:,idx] = 0
        K_mod[idx,idx] = 1
        F_mod[idx] = 0

    u = np.linalg.solve(K_mod, F_mod)

    # stresses
    stresses = []
    for e, (n1, n2) in enumerate(elements):
        k, L, c, s = element_stiffness(n1, n2, A_values[e])
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        u_e = u[dofs]

        strain = (1/L)*np.array([-c, -s, c, s]) @ u_e
        stress = E * strain
        stresses.append(stress)

    return u, np.array(stresses)

# -------------------------
# INITIAL ANALYSIS
# -------------------------

A_vals = np.full(len(elements), A)

u1, stress1 = solve_truss(A_vals)

# find first crack (max tension)
crack_idx = np.argmax(stress1)
crack_element = elements[crack_idx]

# -------------------------
# REINFORCEMENT LOGIC
# -------------------------

A_vals_reinf = A_vals.copy()

# reinforce elements connected to crack nodes
for i, (n1, n2) in enumerate(elements):
    if n1 in crack_element or n2 in crack_element:
        A_vals_reinf[i] = A_reinf

# -------------------------
# RE-ANALYSIS
# -------------------------

u2, stress2 = solve_truss(A_vals_reinf)

# -------------------------
# PLOTTING
# -------------------------

def plot(ax, u, stress, A_vals, title):

    scale = 200

    # undeformed
    for (n1, n2) in elements:
        x = [nodes[n1,0], nodes[n2,0]]
        y = [nodes[n1,1], nodes[n2,1]]
        ax.plot(x, y, 'k--', alpha=0.3)

    # deformed
    for i, (n1, n2) in enumerate(elements):
        x = [nodes[n1,0] + scale*u[2*n1],
             nodes[n2,0] + scale*u[2*n2]]
        y = [nodes[n1,1] + scale*u[2*n1+1],
             nodes[n2,1] + scale*u[2*n2+1]]

        s = stress[i]

        color = 'blue' if s > 0 else 'red'
        lw = 2 if A_vals[i] == A else 4

        ax.plot(x, y, color=color, linewidth=lw)

        # highlight critical
        if i == crack_idx:
            ax.plot(x, y, color='yellow', linewidth=5)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)

# -------------------------
# PLOTS SIDE BY SIDE
# -------------------------

fig, axs = plt.subplots(1, 2, figsize=(12,5))

plot(axs[0], u1, stress1, A_vals, "Before Reinforcement")
plot(axs[1], u2, stress2, A_vals_reinf, "After Reinforcement")

plt.tight_layout()
plt.show()