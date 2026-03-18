import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# MATERIAL PROPERTIES
# -------------------------

E_conc = 30e9
E_steel = 200e9
A = 0.005
ft = 3e6

# -------------------------
# GEOMETRY
# -------------------------

nx, ny = 12, 4
L, H = 10, 2

dx = L/(nx-1)
dy = H/(ny-1)

nodes = np.array([[i*dx, j*dy] for j in range(ny) for i in range(nx)])

def node(i,j):
    return j*nx + i

# -------------------------
# ELEMENTS
# -------------------------

elements = []

for j in range(ny):
    for i in range(nx-1):
        elements.append([node(i,j), node(i+1,j)])

for j in range(ny-1):
    for i in range(nx):
        elements.append([node(i,j), node(i,j+1)])

for j in range(ny-1):
    for i in range(nx-1):
        elements.append([node(i,j), node(i+1,j+1)])
        elements.append([node(i+1,j), node(i,j+1)])

elements = np.array(elements)

n_nodes = len(nodes)
dof = 2*n_nodes

# -------------------------
# LOADS
# -------------------------

F = np.zeros(dof)
for i in range(nx):
    F[2*node(i,ny-1)+1] = -5000

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

def element_stiffness(n1,n2,E):
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
# SOLVER
# -------------------------

def solve(E_vals):
    K = np.zeros((dof,dof))

    for e,(n1,n2) in enumerate(elements):
        k,_,_,_ = element_stiffness(n1,n2,E_vals[e])
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
    for e,(n1,n2) in enumerate(elements):
        _,L,c,s = element_stiffness(n1,n2,E_vals[e])
        dofs = [2*n1,2*n1+1,2*n2,2*n2+1]
        u_e = u[dofs]

        strain = (1/L)*np.array([-c,-s,c,s]) @ u_e
        stress.append(E_vals[e]*strain)

    return u, np.array(stress)

# -------------------------
# INITIAL ANALYSIS
# -------------------------

E_vals_conc = np.full(len(elements), E_conc)
u1, stress1 = solve(E_vals_conc)

tension1 = np.maximum(stress1,0)
first_crack = np.argmax(tension1)
cracked = np.where(tension1 > ft)[0]

# -------------------------
# STEEL REPLACEMENT
# -------------------------

E_vals_mixed = E_vals_conc.copy()
for i in cracked:
    E_vals_mixed[i] = E_steel

u2, stress2 = solve(E_vals_mixed)

# -------------------------
# PLOTTING
# -------------------------

def plot(ax, u, stress, E_vals, title, show_cracks=False):

    scale = 200
    max_stress = np.max(np.abs(stress))

    # -------------------------
    # UNDEFORMED
    # -------------------------
    for (n1,n2) in elements:
        ax.plot([nodes[n1,0],nodes[n2,0]],
                [nodes[n1,1],nodes[n2,1]],
                'k--',alpha=0.2)

    # -------------------------
    # DEFORMED
    # -------------------------
    for i,(n1,n2) in enumerate(elements):

        x = [nodes[n1,0] + scale*u[2*n1],
             nodes[n2,0] + scale*u[2*n2]]

        y = [nodes[n1,1] + scale*u[2*n1+1],
             nodes[n2,1] + scale*u[2*n2+1]]

        s = stress[i]
        lw = 0.5 + 4*abs(s)/max_stress
        color = 'blue' if s > 0 else 'red'

        if show_cracks and i in cracked:
            color = 'yellow'
        if show_cracks and i == first_crack:
            color = 'orange'
            lw = 5

        ax.plot(x,y,color=color,linewidth=lw)

        # -------------------------
        # STEEL MIDPOINT DOT
        # -------------------------
        if E_vals[i] == E_steel:
            xm = (x[0]+x[1])/2
            ym = (y[0]+y[1])/2
            ax.plot(xm, ym, 'ko', markersize=4)

    # -------------------------
    # LOAD ARROWS + LABELS
    # -------------------------
    for i in range(nx):
        n = node(i,ny-1)

        x = nodes[n,0]
        y = nodes[n,1]

        load_val = F[2*n+1]  # vertical load

        # draw arrow
        ax.arrow(
            x, y + 0.2,
            0, -0.3,
            head_width=0.15,
            head_length=0.15,
            fc='black', ec='black'
        )

        # label magnitude (convert to kN)
        ax.text(
            x,
            y + 0.35,
            f"{abs(load_val)/1000:.1f} kN",
            ha='center',
            fontsize=8
        )

    # -------------------------
    # FORMATTING
    # -------------------------
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)

    # units
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
# -------------------------
# FIGURE WITH TEXT PANEL
# -------------------------

fig = plt.figure(figsize=(14,8))

gs = fig.add_gridspec(2,2,height_ratios=[3,1])

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax_text = fig.add_subplot(gs[1,:])

plot(ax1, u1, stress1, E_vals_conc, "Concrete (Initial)", True)
plot(ax2, u2, stress2, E_vals_mixed, "After Steel Replacement", False)

# -------------------------
# LEGEND OUTSIDE
# -------------------------

handles = [
    plt.Line2D([],[],color='blue',label='Tension'),
    plt.Line2D([],[],color='red',label='Compression'),
    plt.Line2D([],[],color='yellow',label='Cracked'),
    plt.Line2D([],[],color='orange',label='First Crack'),
    plt.Line2D([],[],marker='o',color='black',linestyle='None',label='Steel')
]

fig.legend(handles=handles, loc='upper center', ncol=5)

# -------------------------
# TEXT OUTPUT
# -------------------------

ax_text.axis('off')

text = f"""
INITIAL (CONCRETE):
First crack element: {first_crack}
Max tensile stress: {tension1[first_crack]:.2e} Pa
Number of cracked members: {len(cracked)}

AFTER STEEL REPLACEMENT:
Max tensile stress: {np.max(np.maximum(stress2,0)):.2e} Pa
"""

ax_text.text(0.01, 0.5, text, fontsize=11, va='center')

plt.tight_layout()
plt.show()