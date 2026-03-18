import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# MATERIAL
# -------------------------

E = 30e9
A = 0.005

# concrete tensile strength (Pa)
ft = 3e6  

# -------------------------
# GEOMETRY (DENSE GRID)
# -------------------------

nx, ny = 12, 4
L, H = 10, 2

dx = L / (nx - 1)
dy = H / (ny - 1)

nodes = np.array([[i*dx, j*dy] for j in range(ny) for i in range(nx)])

def node(i, j):
    return j*nx + i

# -------------------------
# ELEMENTS
# -------------------------

elements = []

# horizontal
for j in range(ny):
    for i in range(nx-1):
        elements.append([node(i,j), node(i+1,j)])

# vertical
for j in range(ny-1):
    for i in range(nx):
        elements.append([node(i,j), node(i,j+1)])

# diagonals
for j in range(ny-1):
    for i in range(nx-1):
        elements.append([node(i,j), node(i+1,j+1)])
        elements.append([node(i+1,j), node(i,j+1)])

elements = np.array(elements)

# -------------------------
# DOF
# -------------------------

n_nodes = len(nodes)
dof = 2*n_nodes

# -------------------------
# LOADS
# -------------------------

F = np.zeros(dof)
for i in range(nx):
    F[2*node(i, ny-1)+1] = -5000

# -------------------------
# SUPPORTS
# -------------------------

supports = [
    (node(0,0),0),(node(0,0),1),
    (node(nx-1,0),1)
]

# -------------------------
# ELEMENT STIFFNESS
# -------------------------

def element_stiffness(n1,n2):
    x1,y1 = nodes[n1]
    x2,y2 = nodes[n2]

    L = np.hypot(x2-x1,y2-y1)
    c = (x2-x1)/L
    s = (y2-y1)/L

    k = (E*A/L)*np.array([
        [ c*c, c*s,-c*c,-c*s],
        [ c*s, s*s,-c*s,-s*s],
        [-c*c,-c*s, c*c, c*s],
        [-c*s,-s*s, c*s, s*s]
    ])
    return k,L,c,s

# -------------------------
# SOLVE
# -------------------------

def solve():
    K = np.zeros((dof,dof))

    for (n1,n2) in elements:
        k,_,_,_ = element_stiffness(n1,n2)
        dofs = [2*n1,2*n1+1,2*n2,2*n2+1]

        for i in range(4):
            for j in range(4):
                K[dofs[i],dofs[j]] += k[i,j]

    K_mod = K.copy()
    F_mod = F.copy()

    for (n,d) in supports:
        idx = 2*n + d
        K_mod[idx,:] = 0
        K_mod[:,idx] = 0
        K_mod[idx,idx] = 1
        F_mod[idx] = 0

    u = np.linalg.solve(K_mod,F_mod)

    stress = []
    for (n1,n2) in elements:
        _,L,c,s = element_stiffness(n1,n2)
        dofs = [2*n1,2*n1+1,2*n2,2*n2+1]
        u_e = u[dofs]

        strain = (1/L)*np.array([-c,-s,c,s]) @ u_e
        stress.append(E*strain)

    return u, np.array(stress)

u, stress = solve()

# -------------------------
# CRACK DETECTION
# -------------------------

tension_stress = np.maximum(stress, 0)

first_crack_idx = np.argmax(tension_stress)

cracked_elements = np.where(tension_stress > ft)[0]

print("\n--- RESULTS ---")
print(f"Max tensile stress: {tension_stress[first_crack_idx]:.2e} Pa")
print(f"First crack element: {first_crack_idx}")
print(f"Number of cracked elements: {len(cracked_elements)}")

# -------------------------
# PLOT
# -------------------------

def plot():

    scale = 200

    fig, ax = plt.subplots(figsize=(10,5))

    # undeformed
    for (n1,n2) in elements:
        ax.plot(
            [nodes[n1,0],nodes[n2,0]],
            [nodes[n1,1],nodes[n2,1]],
            'k--',alpha=0.2
        )

    max_stress = np.max(np.abs(stress))

    for i,(n1,n2) in enumerate(elements):

        x = [
            nodes[n1,0] + scale*u[2*n1],
            nodes[n2,0] + scale*u[2*n2]
        ]
        y = [
            nodes[n1,1] + scale*u[2*n1+1],
            nodes[n2,1] + scale*u[2*n2+1]
        ]

        s = stress[i]

        # thickness based on stress magnitude
        lw = 0.5 + 4 * abs(s) / max_stress

        # default color
        color = 'blue' if s > 0 else 'red'

        # cracked members
        if i in cracked_elements:
            color = 'yellow'

        # first crack overrides everything
        if i == first_crack_idx:
            color = 'orange'
            lw = 5

        ax.plot(x, y, color=color, linewidth=lw)

    # load arrows
    for i in range(nx):
        n = node(i, ny-1)
        ax.arrow(
            nodes[n,0],
            nodes[n,1] + 0.2,
            0,
            -0.3,
            head_width=0.15,
            head_length=0.15,
            fc='black',
            ec='black'
        )

    ax.set_title("Stress-Based Cracking Visualization")
    ax.set_aspect('equal')
    ax.grid(True)

    # legend
    ax.plot([],[],color='blue',label='Tension')
    ax.plot([],[],color='red',label='Compression')
    ax.plot([],[],color='yellow',label='Cracked (σ > ft)')
    ax.plot([],[],color='orange',label='First Crack')

    ax.legend()

    plt.show()

plot()