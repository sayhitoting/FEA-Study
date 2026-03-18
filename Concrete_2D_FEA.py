import numpy as np

# =========================
# USER INPUT
# =========================

# Node coordinates: [x, y]
nodes = np.array([
    [0, 0],
    [1, 1],
    [2, 0]
])

elements = [
    (0, 1),
    (1, 2),
    (0, 2)
]

supports = [
    (0, 0), (0, 1),
    (2, 1)
]

E = 30e9          # Young's modulus (Pa)
A = 0.01          # Cross-sectional area (m^2)
ft = 3e6          # Tensile strength (Pa)


# Loads: (node, dof, value)
loads = [
    (1, 1, -10000)    # Downward force at node 1
]

# Desired crack node
target_node = 1

# =========================
# FEA FUNCTIONS
# =========================

def element_stiffness(n1, n2, E, A):
    x1, y1 = n1
    x2, y2 = n2
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

def assemble(nodes, elements, E_list, A_list):
    dof = len(nodes)*2
    K = np.zeros((dof, dof))

    for i, (n1, n2) in enumerate(elements):
        k, _, _, _ = element_stiffness(nodes[n1], nodes[n2], E_list[i], A_list[i])

        dof_map = [
            2*n1, 2*n1+1,
            2*n2, 2*n2+1
        ]

        for a in range(4):
            for b in range(4):
                K[dof_map[a], dof_map[b]] += k[a, b]

    return K

def apply_bc(K, F, supports):
    for node, d in supports:
        idx = 2*node + d
        K[idx, :] = 0
        K[:, idx] = 0
        K[idx, idx] = 1
        F[idx] = 0
    return K, F

def solve(K, F):
    return np.linalg.solve(K, F)

def compute_stress(nodes, elements, u, E_list):
    stresses = []
    for i, (n1, n2) in enumerate(elements):
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]

        L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        c = (x2-x1)/L
        s = (y2-y1)/L

        u_e = np.array([
            u[2*n1], u[2*n1+1],
            u[2*n2], u[2*n2+1]
        ])

        strain = (1/L) * np.array([-c, -s, c, s]) @ u_e
        stress = E_list[i] * strain
        stresses.append(stress)

    return np.array(stresses)

# =========================
# INITIAL ANALYSIS
# =========================

E_list = [E]*len(elements)
A_list = [A]*len(elements)

dof = len(nodes)*2
F = np.zeros(dof)

for node, d, val in loads:
    F[2*node + d] = val

K = assemble(nodes, elements, E_list, A_list)
K, F = apply_bc(K, F, supports)

u = solve(K, F)
stresses = compute_stress(nodes, elements, u, E_list)

# Find first cracking element
crack_ratios = stresses / ft
first_crack_element = np.argmax(crack_ratios)

print("Initial stresses:", stresses)
print("First crack element:", first_crack_element)

# =========================
# REINFORCEMENT LOGIC
# =========================

def reinforce(elements, target_node, E_list, factor=5):
    new_E = E_list.copy()

    for i, (n1, n2) in enumerate(elements):
        # If element does NOT touch target node → reinforce it
        if target_node not in (n1, n2):
            new_E[i] *= factor

    return new_E

E_reinf = reinforce(elements, target_node, E_list)

# =========================
# RE-ANALYSIS
# =========================

K2 = assemble(nodes, elements, E_reinf, A_list)
F2 = np.zeros(dof)

for node, d, val in loads:
    F2[2*node + d] = val

K2, F2 = apply_bc(K2, F2, supports)

u2 = solve(K2, F2)
stresses2 = compute_stress(nodes, elements, u2, E_reinf)

crack_ratios2 = stresses2 / ft
new_crack_element = np.argmax(crack_ratios2)

print("\nAfter reinforcement:")
print("Stresses:", stresses2)
print("New first crack element:", new_crack_element)