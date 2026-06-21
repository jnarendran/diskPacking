import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from collections import deque
import math, time
from scipy.spatial import KDTree, ConvexHull

# spreads grains evenly on spherical cap using approximately hexagonal packing
def hex_pack_for_N(R, h, d, N):
    cos_tmax = (R - h)/R
    tmax     = math.acos(cos_tmax)
    cap_arc  = R * tmax
    def count_for_s(s):
        dr = s*math.sqrt(3)/2
        kmax = int(cap_arc//dr)
        total = 1
        for k in range(1, kmax+1):
            arc_r = dr*k
            theta = arc_r/R
            rho   = R*math.sin(theta)
            n_k   = int(math.floor(2*math.pi*rho / s))
            total += n_k
        return total, kmax

    lo, hi = d, 2*cap_arc
    for _ in range(30):
        mid,_ = 0.5*(lo+hi), None
        cnt,_ = count_for_s(mid)
        if cnt >= N:
            lo = mid
        else:
            hi = mid
    s  = lo
    dr = s*math.sqrt(3)/2

    pts   = [(0.0,0.0,R)]
    theta = [0.0]
    tags  = [0]
    k = 1
    while True:
        arc_r = dr*k
        if arc_r > cap_arc + 1e-6:
            break
        th   = arc_r/R
        rho  = R*math.sin(th)
        z    = R*math.cos(th)
        n_k  = int(math.floor(2*math.pi*rho / s))
        if n_k < 1:
            break
        off  = (k%2)*(math.pi/n_k)
        for j in range(n_k):
            phi = off + 2*math.pi*j/n_k
            x   = rho*math.cos(phi); y = rho*math.sin(phi)
            pts.append((x,y,z))
            theta.append(th)
            tags.append(k)
        k += 1

    if len(pts) < N:
        raise RuntimeError(f"Generated only {len(pts)} < requested N={N}")

    data = list(zip(tags, theta, pts))
    data.sort(key=lambda x: x[0])
    data = data[:N]
    tags, theta, pts = zip(*data)
    return np.array(pts), np.array(theta), np.array(tags), s

# perturbs each hexagonally packed point within a wiggle radius to add 
    # randomness
def jitter_hex_pts(pts, s, d, R):
    w = max(0.0, (s - d)/2.0)
    new_pts = []
    for C in pts:
        Nrm = C/np.linalg.norm(C)
        ref = np.array([0,0,1.0]) if abs(Nrm[2])<0.9 else np.array([0,1.0,0])
        e1  = np.cross(Nrm, ref); e1 /= np.linalg.norm(e1)
        e2  = np.cross(Nrm, e1)
        r   = w * math.sqrt(np.random.rand())
        a   = 2*math.pi*np.random.rand()
        disp= r*(math.cos(a)*e1 + math.sin(a)*e2)
        C2  = C + disp
        C2  = R*(C2/np.linalg.norm(C2))
        new_pts.append(C2)
    new_theta = [math.acos(p[2]/R) for p in new_pts]
    return np.array(new_pts), np.array(new_theta)

# creates list of all point pairs within set touch distance
def build_graph_fast(pts, R, touch):
    chord = 2*R*math.sin((touch/R)/2)
    tree  = KDTree(pts)
    pairs = tree.query_pairs(chord)
    adj   = [[] for _ in pts]
    for i,j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    return adj

# finds grains at apex and edge of the spherical cap
def find_edge_apex(theta, R, h, d, tags):
    apex   = set(np.where(R*theta <= 5.0)[0].tolist())
    kmax   = tags.max()
    by_tag = set(np.where(tags == kmax)[0].tolist())
    tmax   = math.acos((R - h)/R)
    by_geo = set(np.where(R*(tmax - theta) <= d/2.0)[0].tolist())
    rim    = by_tag.union(by_geo)
    return rim, apex

# helps determine which grains are at the edge of the spherical cap
def find_edge_hull(pts):
    hull = ConvexHull(pts[:,:2])
    return set(hull.vertices)

# helper to find single shortest paths from each source to target
def _shortest_path(adj, source, target):
    visited = {source}
    q = deque([(source, [source])])
    while q:
        u, path = q.popleft()
        if u == target:
            return path
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append((v, path + [v]))
    return None

# finds shortest path from each source to target
def find_all_chains(adj, sources, target):
    raw = []
    for s in sources:
        p = _shortest_path(adj, s, target)
        if p is not None:
            raw.append(tuple(p))
    seen = set(); unique = []
    for p in raw:
        if p not in seen:
            seen.add(p)
            unique.append(list(p))
    return unique

# finds path of upward chains for force chain propagation
def find_chain_upward(adj, pts, sources, target, max_angle_deg=90.0):
    cos_thresh = math.cos(math.radians(max_angle_deg))
    visited = set(sources)
    q = deque((s, [s]) for s in sources)
    while q:
        u, path = q.popleft()
        if u == target:
            return path
        Pu = pts[u]
        for v in adj[u]:
            if v in visited:
                continue
            vec = pts[v] - Pu
            dz  = vec[2]
            normv = np.linalg.norm(vec)
            if normv == 0 or dz / normv < cos_thresh:
                continue
            visited.add(v)
            q.append((v, path + [v]))
    return None

# created 3D visualization of spherical cap and disks
def plot_scene(R, h, pts, d, chains, adj, elev=30, azim=45):
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    # draw cap
    nmesh = 50
    tmax  = math.acos((R-h)/R)
    tt    = np.linspace(0, tmax, nmesh)
    pp    = np.linspace(0, 2*math.pi, nmesh)
    TT, PP = np.meshgrid(tt, pp)
    X = R*np.sin(TT)*np.cos(PP)
    Y = R*np.sin(TT)*np.sin(PP)
    Z = R*np.cos(TT)
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3, linewidth=0)

    N = len(pts)
    # build set of chain-edges so we don’t draw them in gray
    chain_edges = set()
    for ch in chains:
        for a,b in zip(ch, ch[1:]):
            chain_edges.add((min(a,b), max(a,b)))

    # draw all other contacts
    for i in range(N):
        for j in adj[i]:
            if j <= i:
                continue
            key = (min(i,j), max(i,j))
            if key in chain_edges:
                continue
            p0, p1 = pts[i], pts[j]
            ax.plot([p0[0],p1[0]],
                    [p0[1],p1[1]],
                    [p0[2],p1[2]],
                    c='lightgray', lw=0.5, alpha=0.6)

    # colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(chains))]

    # draw chains
    for idx,ch in enumerate(chains):
        col = colors[idx]
        for a,b in zip(ch, ch[1:]):
            p0, p1 = pts[a], pts[b]
            ax.plot([p0[0],p1[0]],
                    [p0[1],p1[1]],
                    [p0[2],p1[2]],
                    c=col, lw=3)

    # map nodes to chain‐color if any
    node_chain = {}
    for idx,ch in enumerate(chains):
        for v in ch:
            node_chain[v] = idx

    # draw disks
    for i, C in enumerate(pts):
        Nrm = C/np.linalg.norm(C)
        ref = np.array([0,0,1.]) if abs(Nrm[2])<0.9 else np.array([0,1.,0])
        e1  = np.cross(Nrm, ref); e1 /= np.linalg.norm(e1)
        e2  = np.cross(Nrm, e1)
        th  = np.linspace(0, 2*math.pi, 64)
        circ = C[None,:] + (d/2.0)*(np.cos(th)[:,None]*e1 + 
                                    np.sin(th)[:,None]*e2)

        if i in node_chain:
            fc, ec, alp = colors[node_chain[i]], 'k', 1.0
        else:
            fc, ec, alp = 'gray', 'k', 0.6

        poly = art3d.Poly3DCollection([circ],
                                     facecolors=fc,
                                     edgecolors=ec,
                                     alpha=alp)
        ax.add_collection3d(poly)

    # equal-axis
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    mx = 0.5*(xlim[0]+xlim[1])
    my = 0.5*(ylim[0]+ylim[1])
    mz = 0.5*(zlim[0]+zlim[1])
    r = 0.5*max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    ax.set_xlim3d(mx-r, mx+r)
    ax.set_ylim3d(my-r, my+r)
    ax.set_zlim3d(mz-r, mz+r)

    plt.tight_layout()
    plt.show()

def main():
    #shell parameters
    cap_diam = 63.75  # mm
    h        = 24.5   # mm
    d        = 2.6    # mm
    #a_cap    = cap_diam/2.0
    R        = 63.75/2 #(a_cap*a_cap + h*h)/(2*h)

    cap_area  = 2*math.pi*R*h
    disk_area = math.pi*(d/2.0)**2
    max_N     = int(math.floor(0.861*cap_area/disk_area))
    print(f"Max disks = {max_N}")
    # sets default parameters for 100 trials at desired touch threshold
        # a single trial is plotted
    N0    = int(input(f"Desired N (1–{max_N}): "))
    N1    = N0
    step  = int(1)
    M     = int(100)
    factor= float(1.136)
    touch = factor * d

    Ns, rates, cnums = [], [], []
    t0 = time.perf_counter()

    for N in range(N0, N1+1, step):
        print(f"\n=== N = {N} ===")
        pts_base, theta_base, tags, s = hex_pack_for_N(R, h, d, N)

        succ = 0
        contact_avgs = []

        for _ in range(M):
            pts_r, theta_r = jitter_hex_pts(pts_base, s, d, R)
            adj            = build_graph_fast(pts_r, R, touch)
            rim, _      = find_edge_apex(theta_r, R, h, d, tags)
            hull_edge   = find_edge_hull(pts_r)
            edge_all    = rim.union(hull_edge)
            apex_target = int(np.argmax(pts_r[:,2]))

            # find all candidate rim→apex paths
            all_chains = find_all_chains(adj, list(edge_all), apex_target)
            # filter for <= 90 degree hops (i.e. no downward)
            valid_chains = []
            cos_thresh = math.cos(math.radians(90.0))
            for ch in all_chains:
                ok = True
                for u,v in zip(ch, ch[1:]):
                    vec = pts_r[v] - pts_r[u]
                    placeholderA = np.linalg.norm(vec)==0
                    placeholderB = vec[2]/np.linalg.norm(vec) < cos_thresh
                    if placeholderA or placeholderB:
                        ok = False
                        break
                if ok:
                    valid_chains.append(ch)

            if valid_chains:
                succ += 1

            interior = [i for i in range(N) if i not in edge_all]
            avg_c = np.mean([len(adj[i]) for i in interior]) if interior else 0.0
            contact_avgs.append(avg_c)

        rate = succ/M
        mean_c = float(np.mean(contact_avgs))
        print(f"  success‐rate = {succ}/{M} = {rate:.3f}")
        print("mean contacts = {mean_c:.3f}")

        Ns.append(N)
        rates.append(rate)
        cnums.append(mean_c)

        # save last trial data for final render
        last_pts_r      = pts_r
        last_adj        = adj
        last_edge_all   = edge_all
        last_valid_chains = valid_chains

    t1 = time.perf_counter()
    print(f"\nSweep completed in {t1-t0:.3f}s")

    # final render
    print(f"\nRendering final configuration for N={Ns[-1]}")
    print(f"Found {len(last_valid_chains)} valid force chains.")
    choice = input("Highlight 0, 1, 3, or all chains? ").strip().lower()
    if choice == 'all':
        k = len(last_valid_chains)
    else:
        try:
            k = int(choice)
        except:
            k = 0
        k = max(0, min(k, len(last_valid_chains)))
    to_draw = last_valid_chains[:k]

    plot_scene(R, h, last_pts_r, d, to_draw, last_adj)

if __name__ == '__main__':
    main()