import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER INPUT
# =========================

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

E = 30e9
A = 0.01
ft = 3e6

supports = [
    (0, 0), (0, 1),
    (2, 1)
]

loads = [
    (1, 1, -10000)
]

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

    return (E*A/L) * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])

def assemble(nodes, elements, E_list, A_list):
    dof = len(nodes)*2
    K = np.zeros((dof, dof))

    for i, (n1, n2) in enumerate(elements):
        k = element_stiffness(nodes[n1], nodes[n2], E_list[i], A_list[i])
        dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

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
    strains = []

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

        strains.append(strain)
        stresses.append(stress)

    return np.array(stresses), np.array(strains)

# =========================
# VISUALIZATION
# =========================

def plot_truss(nodes, elements, u=None, stresses=None,
               supports=None, loads=None, scale=1.0, title="Truss"):

    fig, ax = plt.subplots()
    nodes = np.array(nodes)

    if u is not None:
        u = u.reshape(-1, 2)
        deformed_nodes = nodes + scale * u
    else:
        deformed_nodes = nodes

    if stresses is not None and len(stresses) > 0:
        max_stress = max(abs(stresses))
        if max_stress == 0:
            max_stress = 1
    else:
        max_stress = 1

    # Elements
    for i, (n1, n2) in enumerate(elements):
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]

        xd = [deformed_nodes[n1, 0], deformed_nodes[n2, 0]]
        yd = [deformed_nodes[n1, 1], deformed_nodes[n2, 1]]

        if stresses is not None:
            s = stresses[i] / max_stress
            color = plt.cm.coolwarm((s + 1) / 2)
        else:
            color = "black"

        ax.plot(x, y, 'k--', linewidth=1)
        ax.plot(xd, yd, color=color, linewidth=3)

        # Stress label
        if stresses is not None:
            xm = (xd[0] + xd[1]) / 2
            ym = (yd[0] + yd[1]) / 2
            ax.text(xm, ym, f"{stresses[i]/1e6:.2f} MPa",
                    fontsize=8, ha='center')

    # Nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], color='black', label="Nodes")
    ax.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], color='red', label="Deformed")

    # Supports
    if supports is not None:
        for node, d in supports:
            x, y = nodes[node]
            if d == 0:
                ax.scatter(x, y, marker='>', color='green', s=100, label="X Support")
            elif d == 1:
                ax.scatter(x, y, marker='^', color='blue', s=100, label="Y Support")

    # Loads with magnitude
    if loads is not None:
        for node, d, val in loads:
            x, y = nodes[node]

            if d == 0:
                dx = np.sign(val) * 0.3
                dy = 0
            else:
                dx = 0
                dy = np.sign(val) * 0.3

            ax.arrow(x, y, dx, dy, head_width=0.1, color='purple')

            # Label load magnitude
            ax.text(x + dx, y + dy, f"{val:.0f} N",
                    color='purple', fontsize=9)

    # Colorbar
    if stresses is not None:
        sm = plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=plt.Normalize(vmin=-max_stress, vmax=max_stress)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Stress (Pa)")

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.show()

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
stresses, strains = compute_stress(nodes, elements, u, E_list)

print("\n=== INITIAL ANALYSIS ===")
print("Displacements (m):")
print(u)

print("\nStrains:")
for i, s in enumerate(strains):
    print(f"Element {i}: {s:.6e}")

print("\nStresses (Pa):")
for i, s in enumerate(stresses):
    print(f"Element {i}: {s:.3e}")

plot_truss(nodes, elements, u, stresses,
           supports=supports,
           loads=loads,
           scale=1000,
           title="Initial Structure")

# =========================
# REINFORCEMENT
# =========================

def reinforce(elements, target_node, E_list, factor=5):
    new_E = E_list.copy()

    for i, (n1, n2) in enumerate(elements):
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
stresses2, strains2 = compute_stress(nodes, elements, u2, E_reinf)

print("\n=== AFTER REINFORCEMENT ===")
print("Displacements (m):")
print(u2)

print("\nStrains:")
for i, s in enumerate(strains2):
    print(f"Element {i}: {s:.6e}")

print("\nStresses (Pa):")
for i, s in enumerate(stresses2):
    print(f"Element {i}: {s:.3e}")

plot_truss(nodes, elements, u2, stresses2,
           supports=supports,
           loads=loads,
           scale=1000,
           title="After Reinforcement")