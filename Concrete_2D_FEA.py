import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# -------------------------
# INPUT DATA
# -------------------------

nodes = np.array([
    [0, 0],
    [2, 0],
    [1, 1]
])

elements = [
    (0, 2),
    (1, 2),
    (0, 1)
]

E = 200e9
A = 0.01

supports = [
    (0, 0), (0, 1),   # node 0 fixed
    (1, 1)            # node 1 roller (y only)
]

loads = np.zeros(2 * len(nodes))
loads[2*2 + 1] = -10000  # downward load at node 2


# -------------------------
# FUNCTIONS
# -------------------------

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
    return k


def assemble(nodes, elements, E_list, A):
    dof = 2 * len(nodes)
    K = np.zeros((dof, dof))

    for i, (n1, n2) in enumerate(elements):
        k = element_stiffness(nodes[n1], nodes[n2], E_list[i], A)
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]

        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k[a, b]

    return K


def apply_bc(K, F, supports):
    for node, direction in supports:
        dof = 2*node + direction
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
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

        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        u_e = u[dofs]

        strain = (1/L) * np.array([-c, -s, c, s]) @ u_e
        stress = E_list[i] * strain
        stresses.append(stress)

    return np.array(stresses)


def find_first_crack(stresses):
    return np.argmax(stresses)  # max tension


def reinforce(elements, target_node, E_list, factor=5):
    new_E = E_list.copy()
    reinforced = []

    for i, (n1, n2) in enumerate(elements):
        if target_node not in (n1, n2):
            new_E[i] *= factor
            reinforced.append(i)

    return new_E, reinforced


# -------------------------
# PLOTTING
# -------------------------

def plot(ax, nodes, elements, u, stresses, supports, loads,
         reinforced_elements=None, scale=10000, title=""):

    deformed = nodes + scale * u.reshape(-1, 2)

    # --- Undeformed structure (ghost) ---
    for (n1, n2) in elements:
        ax.plot([nodes[n1][0], nodes[n2][0]],
                [nodes[n1][1], nodes[n2][1]],
                linestyle='--',
                color='gray',
                linewidth=2,
                alpha=0.6,
                label="Undeformed" if (n1, n2) == elements[0] else "")

    # Create line segments
    segments = []
    for (n1, n2) in elements:
        segments.append([deformed[n1], deformed[n2]])

    lc = LineCollection(segments, cmap='coolwarm')
    lc.set_array(stresses)
    lc.set_linewidth(5)
    ax.add_collection(lc)

    # Reinforcement overlay (dashed, does NOT override color)
    if reinforced_elements:
        for i in reinforced_elements:
            n1, n2 = elements[i]
            ax.plot([deformed[n1][0], deformed[n2][0]],
                    [deformed[n1][1], deformed[n2][1]],
                    'k--', linewidth=2, label="Reinforced" if i == reinforced_elements[0] else "")

    # Nodes
    ax.scatter(nodes[:,0], nodes[:,1], color='black', label="Nodes")
    ax.scatter(deformed[:,0], deformed[:,1], color='red', label="Deformed")

    # Supports
    for node, d in supports:
        x, y = nodes[node]
        if d == 0:
            ax.plot(x, y, 'g>', markersize=10, label="X Support")
        else:
            ax.plot(x, y, 'b^', markersize=10, label="Y Support")

    # Loads
    for i in range(len(nodes)):
        fx = loads[2*i]
        fy = loads[2*i+1]
        if fx != 0 or fy != 0:
            ax.arrow(nodes[i][0], nodes[i][1],
                     fx*0.00005, fy*0.00005,
                     head_width=0.05, color='purple')
            ax.text(nodes[i][0], nodes[i][1]+0.1,
                    f"{fy:.0f} N", color='purple')

    # Stress labels
    for i, (n1, n2) in enumerate(elements):
        mid = (deformed[n1] + deformed[n2]) / 2
        ax.text(mid[0], mid[1],
                f"{stresses[i]/1e6:.2f} MPa",
                fontsize=9)

    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True)


# -------------------------
# MAIN ANALYSIS
# -------------------------

E_list = np.array([E]*len(elements))

# BEFORE
K = assemble(nodes, elements, E_list, A)
F = loads.copy()
K, F = apply_bc(K, F, supports)
u = solve(K, F)
stresses = compute_stress(nodes, elements, u, E_list)
crack = find_first_crack(stresses)

# USER TARGET
target_node = 2

# REINFORCE
E_new, reinforced_elements = reinforce(elements, target_node, E_list)

# AFTER
K2 = assemble(nodes, elements, E_new, A)
F2 = loads.copy()
K2, F2 = apply_bc(K2, F2, supports)
u2 = solve(K2, F2)
stresses2 = compute_stress(nodes, elements, u2, E_new)
crack2 = find_first_crack(stresses2)


# -------------------------
# PLOTTING SIDE-BY-SIDE
# -------------------------

fig, axs = plt.subplots(1, 2, figsize=(14,6))

plot(axs[0], nodes, elements, u, stresses, supports, loads,
     reinforced_elements=None,
     title="Before Reinforcement")

plot(axs[1], nodes, elements, u2, stresses2, supports, loads,
     reinforced_elements=reinforced_elements,
     title="After Reinforcement")

# Shared colorbar
# sm = plt.cm.ScalarMappable(cmap='coolwarm')
# sm.set_array(np.concatenate([stresses, stresses2]))
# fig.colorbar(sm, ax=axs, label="Stress (Pa)")

plt.tight_layout()
plt.show()


# -------------------------
# PRINT RESULTS
# -------------------------

print("\n--- BEFORE ---")
for i, s in enumerate(stresses):
    print(f"Element {i}: {s:.2f} Pa")

print(f"First crack element: {crack}")

print("\n--- AFTER ---")
for i, s in enumerate(stresses2):
    print(f"Element {i}: {s:.2f} Pa")

print(f"First crack element: {crack2}")

print("\nReinforced elements:", reinforced_elements)