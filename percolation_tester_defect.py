import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from scipy.spatial import KDTree, ConvexHull
from collections import deque
import math

def hex_pack_annulus(R, h, d, N, r_vac):
    """
    Pack exactly N disks on the spherical cap in the annular region
    between geodesic radius r_vac and the cap rim.
    Binary‐searches for the hex‐lattice spacing s such that the annulus
    contains at least N disks — a larger r_vac shrinks the annulus so s
    shrinks too (tighter packing).
    """
    cos_tmax = (R - h)/R
    tmax     = math.acos(cos_tmax)
    cap_arc  = R*tmax

    def count_in_annulus(s):
        dr = s*math.sqrt(3)/2
        if dr <= 0:
            return 0
        kmin = int(math.ceil(r_vac/dr)) if r_vac > 1e-9 else 0
        kmax = int(cap_arc // dr)
        tot  = 1 if kmin == 0 else 0
        for k in range(max(kmin, 1), kmax+1):
            arc_r = dr*k
            if arc_r > cap_arc: break
            rho = R*math.sin(arc_r/R)
            nk  = int(math.floor(2*math.pi*rho/s))
            tot += max(nk, 0)
        return tot

    lo, hi = d, 2*cap_arc
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if count_in_annulus(mid) >= N:
            lo = mid
        else:
            hi = mid
    s  = lo
    dr = s*math.sqrt(3)/2

    kmin = int(math.ceil(r_vac/dr)) if r_vac > 1e-9 else 0
    kmax = int(cap_arc // dr)

    pts, theta, tags = [], [], []
    if kmin == 0:
        pts.append((0.0, 0.0, R))
        theta.append(0.0)
        tags.append(0)

    for k in range(max(kmin, 1), kmax+1):
        arc_r = dr*k
        if arc_r > cap_arc: break
        th  = arc_r/R
        rho = R*math.sin(th)
        z   = R*math.cos(th)
        nk  = int(math.floor(2*math.pi*rho/s))
        if nk < 1: continue
        off = (k%2)*(math.pi/nk)
        for j in range(nk):
            phi = off + 2*math.pi*j/nk
            x,y = rho*math.cos(phi), rho*math.sin(phi)
            pts.append((x, y, z))
            theta.append(th)
            tags.append(k)

    if len(pts) < N:
        raise RuntimeError(
            f"Only generated {len(pts)} disks in annulus "
            f"(r_vac={r_vac:.2f} mm), but requested N={N}. "
            f"Reduce N or r_vac."
        )

    data = sorted(zip(tags, theta, pts), key=lambda x: x[0])[:N]
    tags, theta, pts = zip(*data)
    return np.array(pts), np.array(theta), np.array(tags), s

def jitter_hex_pts(pts, s, d, R):
    w = max(0.0, (s-d)/2)
    out = []
    for C in pts:
        Nrm = C/np.linalg.norm(C)
        ref = np.array([0,0,1.0]) if abs(Nrm[2])<0.9 else np.array([0,1.0,0])
        e1  = np.cross(Nrm, ref); e1 /= np.linalg.norm(e1)
        e2  = np.cross(Nrm, e1)
        r   = w*math.sqrt(np.random.rand())
        a   = 2*math.pi*np.random.rand()
        C2  = C + r*(math.cos(a)*e1 + math.sin(a)*e2)
        out.append(R*(C2/np.linalg.norm(C2)))
    theta = [math.acos(p[2]/R) for p in out]
    return np.array(out), np.array(theta)

def build_graph_fast(pts, R, touch):
    chord = 2*R*math.sin((touch/R)/2)
    pairs = KDTree(pts).query_pairs(chord)
    adj   = [[] for _ in pts]
    for i,j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    return adj

def find_edge_targets(pts, th_r, R, h, d):
    """
    Any disk that lies on the edge of the hemisphere counts as a target.
    We use two criteria and union them:
      1) geodesic distance within d/2 of the cap rim (geometric rim)
      2) on the 2D convex hull of the xy‐projection (outermost in any direction)
    """
    tmax    = math.acos((R - h)/R)
    # 1) geometric rim: disks whose geodesic is within d/2 of tmax*R
    by_geo  = set(np.where(R*(tmax - th_r) <= d/2)[0])
    # 2) convex hull of the xy plane projection
    if len(pts) >= 3:
        hull    = set(ConvexHull(pts[:,:2]).vertices)
    else:
        hull    = set(range(len(pts)))
    return by_geo.union(hull)

def find_upward_chains(adj, pts, sources, targets, max_deg=180):
    cos_thr = math.cos(math.radians(max_deg))
    seen, chains = set(), []
    for s in sources:
        visited = {s}
        q = deque([(s, [s])])
        while q:
            u, path = q.popleft()
            if u in targets:
                tup = tuple(path)
                if tup not in seen:
                    seen.add(tup)
                    chains.append(path[:])
                continue
            Pu = pts[u]
            for v in adj[u]:
                #if v in visited: continue
                #vec   = pts[v] - Pu
                #normv = np.linalg.norm(vec)
                #if normv > 0 and vec[2]/normv >= cos_thr:
                #    visited.add(v)
                #    q.append((v, path+[v]))
                if v in visited: continue
                vec = pts[v]-Pu
                dz  = vec[2]
                if np.linalg.norm(vec)>0 and dz/np.linalg.norm(vec) >= cos_thr:
                    visited.add(v)
                    q.append((v,path+[v]))
    return chains

def plot_chains(R, h, pts, d, chains, adj, sources, targets):
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=30, azim=45)

    # spherical cap surface
    n    = 50
    tmax = math.acos((R-h)/R)
    tt   = np.linspace(0, tmax, n)
    pp   = np.linspace(0, 2*math.pi, n)
    TT, PP = np.meshgrid(tt, pp)
    X = R*np.sin(TT)*np.cos(PP)
    Y = R*np.sin(TT)*np.sin(PP)
    Z = R*np.cos(TT)
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3, linewidth=0)

    N = len(pts)
    chain_edges = set()
    for ch in chains:
        for a,b in zip(ch, ch[1:]):
            chain_edges.add((min(a,b), max(a,b)))

    # non‐chain contacts in gray
    for i in range(N):
        for j in adj[i]:
            if j <= i: continue
            key = (min(i,j), max(i,j))
            if key in chain_edges: continue
            p0, p1 = pts[i], pts[j]
            ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
                    c='lightgray', lw=0.5, alpha=0.6)

    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i%10) for i in range(len(chains))]

    # draw force chains
    for idx, ch in enumerate(chains):
        col = colors[idx]
        for a,b in zip(ch, ch[1:]):
            p0, p1 = pts[a], pts[b]
            ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
                    c=col, lw=3)

    # disk colors: chain members colored, sources/targets marked
    node_chain = {}
    for idx, ch in enumerate(chains):
        for v in ch:
            node_chain[v] = idx

    ths = np.linspace(0, 2*math.pi, 64)
    for i, C in enumerate(pts):
        Nrm = C/np.linalg.norm(C)
        ref = np.array([0,0,1.]) if abs(Nrm[2])<0.9 else np.array([0,1.,0])
        e1  = np.cross(Nrm, ref); e1 /= np.linalg.norm(e1)
        e2  = np.cross(Nrm, e1)
        circ = C[None,:] + (d/2)*(np.cos(ths)[:,None]*e1 + np.sin(ths)[:,None]*e2)

        if i in node_chain:
            fc, ec, alp = colors[node_chain[i]], 'k', 1.0
        elif i in sources:
            # vacancy‐ring disks: dark orange outline
            fc, ec, alp = 'orange', 'darkorange', 0.9
        elif i in targets:
            # rim disks: dark green outline
            fc, ec, alp = 'lightgreen', 'darkgreen', 0.9
        else:
            fc, ec, alp = 'gray', 'k', 0.6

        poly = art3d.Poly3DCollection([circ],
                                       facecolors=fc, edgecolors=ec, alpha=alp)
        ax.add_collection3d(poly)

    # equal‐axis
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    mx, my, mz = np.mean(xl), np.mean(yl), np.mean(zl)
    r = 0.5*max(xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0])
    ax.set_xlim3d(mx-r, mx+r)
    ax.set_ylim3d(my-r, my+r)
    ax.set_zlim3d(mz-r, mz+r)
    plt.tight_layout()
    plt.show()

def mainPlot():
    #cap_diam = 63.75  # mm
    h        = 24.5   # mm
    d        = 2.6    # mm
    #a        = cap_diam/2
    R        = 63.75/2 #(a*a + h*h)/(2*h)

    maxN  = int(math.floor(0.861*2*math.pi*R*h / (math.pi*(d/2)**2)))
    print(f"Max disks on full cap: {maxN}")
    N     = int(782) #int(input("Enter N (disks in annulus): "))
    r_vac = int(0) #float(input("Vacancy radius (mm along cap): "))
    tf    = float(1.136)
    touch = tf * d

    # 1) Pack exactly N disks in the annulus between r_vac and rim
    pts_a, th_a, tags_a, s = hex_pack_annulus(R, h, d, N, r_vac)
    print(f" Packed {len(pts_a)} disks  (spacing s={s:.3f} mm)")

    # 2) Jitter & re‐filter vacancy
    pts_r, th_r = jitter_hex_pts(pts_a, s, d, R)
    mask = (R*th_r >= r_vac)
    pts_r, th_r = pts_r[mask], th_r[mask]
    print(f" After jitter & vacancy filter: {len(pts_r)} disks")

    # 3) Contact graph
    adj = build_graph_fast(pts_r, R, touch)

    # 4) Source ring = first ring of disks just outside vacancy
    geod    = R*th_r
    gd_out  = geod[geod > r_vac - 1e-9]
    if len(gd_out) == 0:
        print("No disks outside vacancy."); return
    r_inner = gd_out.min()
    sources = set(np.where(abs(geod - r_inner) <= d/2)[0])
    print(f" Source ring: {len(sources)} disks at geodesic {r_inner:.2f} mm")

    # 5) Targets = ANY disk at the edge of the hemisphere
    #    (geometric rim OR convex hull — no longer just the outermost geodesic ring)
    targets = find_edge_targets(pts_r, th_r, R, h, d)
    print(f" Target (edge) disks: {len(targets)}")

    # 6) Find force chains
    chains = find_upward_chains(adj, pts_r, sources, targets, max_deg=180)
    print(f" Found {len(chains)} force chains")
    participants = set(v for ch in chains for v in ch)
    print(f" {len(participants)}/{len(pts_r)} disks participate")

    # 7) Plot — pass sources & targets so they get special colors
    #plot_chains(R, h, pts_r, d, chains, adj, sources, targets)

def mainRange():
    #cap_diam = 63.75  # mm
    h        = 24.5   # mm
    d        = 2.6    # mm
    #a        = cap_diam/2
    R        = 63.75/2 #(a*a + h*h)/(2*h)

    maxN  = int(math.floor(0.861*2*math.pi*R*h / (math.pi*(d/2)**2)))
    print(f"Max disks on full cap: {maxN}\n")
    N     = int(695)
    
    # NEW: Get range and step size for defect radii
    r_vac_start = float(input("Start defect radius (mm along cap): "))
    r_vac_end   = float(input("End defect radius (mm along cap): "))
    r_vac_step  = float(input("Step size (mm): "))
    
    # NEW: Get number of trials per defect radius
    n_trials = int(input("Number of trials per defect radius: "))
    
    # NEW: Get source band parameters
    print(f"\nSource identification options:")
    print(f"  1. Innermost ring (original method)")
    print(f"  2. Radial band from vacancy edge")
    band_choice = int(input("Choose method (1 or 2): "))
    
    source_band_width = None
    if band_choice == 2:
        source_band_width = float(input("Source band width (mm) [e.g., {:.1f} for 1 disk-diameter]: ".format(d)))
    
    tf    = float(1.136) #float(1.136)
    touch = tf * d

    # NEW: Storage for results (now a dict of lists)
    results_by_rvac = {}
    r_vac_values = []
    r = r_vac_start
    while r <= r_vac_end + 1e-9:
        r_vac_values.append(r)
        results_by_rvac[r] = []
        r += r_vac_step

    print(f"\nTesting {len(r_vac_values)} defect radii ({n_trials} trials each)")
    print(f"Radii: {r_vac_values[0]:.2f} to {r_vac_values[-1]:.2f} mm (step {r_vac_step:.2f} mm)\n")

    # NEW: Loop over all defect radii
    for r_vac in r_vac_values:
        print(f"--- Testing r_vac = {r_vac:.2f} mm ({n_trials} trials) ---")
        
        # NEW: Loop over trials
        for trial in range(n_trials):
            try:
                # 1) Pack exactly N disks in the annulus between r_vac and rim
                pts_a, th_a, tags_a, s = hex_pack_annulus(R, h, d, N, r_vac)

                # 2) Jitter & re-filter vacancy
                pts_r, th_r = jitter_hex_pts(pts_a, s, d, R)
                mask = (R*th_r >= r_vac)
                pts_r, th_r = pts_r[mask], th_r[mask]

                # 3) Contact graph
                adj = build_graph_fast(pts_r, R, touch)

                # 4) Source identification using band or innermost ring
                geod    = R*th_r
                gd_out  = geod[geod > r_vac - 1e-9]
                if len(gd_out) == 0:
                    results_by_rvac[r_vac].append({
                        'trial': trial,
                        'n_disks': len(pts_r),
                        'n_sources': 0,
                        'n_targets': 0,
                        'n_chains': 0,
                        'n_participants': 0
                    })
                    continue
                
                # MODIFIED: Choose source selection method
                if band_choice == 1:
                    # Innermost ring (original)
                    r_inner = gd_out.min()
                    sources = set(np.where(abs(geod - r_inner) <= d/2)[0])
                else:
                    # Radial band from vacancy edge
                    r_inner = r_vac
                    r_outer = r_vac + source_band_width
                    sources = set(np.where((geod >= r_inner) & (geod <= r_outer))[0])

                # 5) Targets = ANY disk at the edge of the hemisphere
                targets = find_edge_targets(pts_r, th_r, R, h, d)

                # 6) Find force chains
                chains = find_upward_chains(adj, pts_r, sources, targets, max_deg=180)
                participants = set(v for ch in chains for v in ch)

                # Store results for this trial
                results_by_rvac[r_vac].append({
                    'trial': trial,
                    'n_disks': len(pts_r),
                    'n_sources': len(sources),
                    'n_targets': len(targets),
                    'n_chains': len(chains),
                    'n_participants': len(participants)
                })

            except Exception as e:
                print(f"  Trial {trial}: Error: {e}")
                results_by_rvac[r_vac].append({
                    'trial': trial,
                    'n_disks': 0,
                    'n_sources': 0,
                    'n_targets': 0,
                    'n_chains': 0,
                    'n_participants': 0
                })
        
        # Print summary for this r_vac after all trials
        trial_results = results_by_rvac[r_vac]
        if trial_results:
            chains_vals = [res['n_chains'] for res in trial_results]
            chains_mean = np.mean(chains_vals)
            chains_std = np.std(chains_vals)
            sources_vals = [res['n_sources'] for res in trial_results]
            sources_mean = np.mean(sources_vals)
            print(f"  Chains: {chains_mean:.2f} ± {chains_std:.2f}  |  Sources: {sources_mean:.1f}")

    # NEW: Compute statistics for each r_vac
    print("\n" + "="*100)
    print("SUMMARY (Mean ± Std Dev across trials)")
    print("="*100)
    print(f"{'r_vac (mm)':<15} {'N_disks':<20} {'Sources':<20} {'Targets':<20} {'Chains':<20} {'Participants':<20}")
    print("-"*100)
    
    stats_by_rvac = {}
    for r_vac in r_vac_values:
        trial_results = results_by_rvac[r_vac]
        if not trial_results:
            continue
        
        stats_by_rvac[r_vac] = {}
        for key in ['n_disks', 'n_sources', 'n_targets', 'n_chains', 'n_participants']:
            vals = [res[key] for res in trial_results]
            stats_by_rvac[r_vac][key] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'vals': vals
            }
        
        print(f"{r_vac:<15.2f} "
              f"{stats_by_rvac[r_vac]['n_disks']['mean']:.1f}±{stats_by_rvac[r_vac]['n_disks']['std']:.1f}        "
              f"{stats_by_rvac[r_vac]['n_sources']['mean']:.1f}±{stats_by_rvac[r_vac]['n_sources']['std']:.1f}        "
              f"{stats_by_rvac[r_vac]['n_targets']['mean']:.1f}±{stats_by_rvac[r_vac]['n_targets']['std']:.1f}        "
              f"{stats_by_rvac[r_vac]['n_chains']['mean']:.1f}±{stats_by_rvac[r_vac]['n_chains']['std']:.1f}        "
              f"{stats_by_rvac[r_vac]['n_participants']['mean']:.1f}±{stats_by_rvac[r_vac]['n_participants']['std']:.1f}")

    # NEW: Plots with error bars
    if len(stats_by_rvac) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        r_vacs_sorted = sorted(stats_by_rvac.keys())
        
        # Plot 1: Number of Disks
        disks_means = [stats_by_rvac[r]['n_disks']['mean'] for r in r_vacs_sorted]
        disks_stds = [stats_by_rvac[r]['n_disks']['std'] for r in r_vacs_sorted]
        axes[0, 0].errorbar(r_vacs_sorted, disks_means, yerr=disks_stds, 
                           fmt='o-', linewidth=2, markersize=6, capsize=5, capthick=1.5)
        axes[0, 0].set_xlabel('Defect Radius (mm)', fontsize=11)
        axes[0, 0].set_ylabel('Number of Disks', fontsize=11)
        axes[0, 0].set_title('Disk Count vs Defect Radius', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Number of Force Chains
        chains_means = [stats_by_rvac[r]['n_chains']['mean'] for r in r_vacs_sorted]
        chains_stds = [stats_by_rvac[r]['n_chains']['std'] for r in r_vacs_sorted]
        axes[0, 1].errorbar(r_vacs_sorted, chains_means, yerr=chains_stds, 
                           fmt='s-', color='red', linewidth=2, markersize=6, capsize=5, capthick=1.5)
        axes[0, 1].set_xlabel('Defect Radius (mm)', fontsize=11)
        axes[0, 1].set_ylabel('Number of Chains', fontsize=11)
        axes[0, 1].set_title('Force Chains vs Defect Radius', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Number of Participants
        part_means = [stats_by_rvac[r]['n_participants']['mean'] for r in r_vacs_sorted]
        part_stds = [stats_by_rvac[r]['n_participants']['std'] for r in r_vacs_sorted]
        axes[1, 0].errorbar(r_vacs_sorted, part_means, yerr=part_stds, 
                           fmt='^-', color='green', linewidth=2, markersize=6, capsize=5, capthick=1.5)
        axes[1, 0].set_xlabel('Defect Radius (mm)', fontsize=11)
        axes[1, 0].set_ylabel('Participant Disks', fontsize=11)
        axes[1, 0].set_title('Chain Participants vs Defect Radius', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Multiple metrics overlay
        source_means = [stats_by_rvac[r]['n_sources']['mean'] for r in r_vacs_sorted]
        source_stds = [stats_by_rvac[r]['n_sources']['std'] for r in r_vacs_sorted]
        target_means = [stats_by_rvac[r]['n_targets']['mean'] for r in r_vacs_sorted]
        target_stds = [stats_by_rvac[r]['n_targets']['std'] for r in r_vacs_sorted]
        
        axes[1, 1].errorbar(r_vacs_sorted, chains_means, yerr=chains_stds, 
                           fmt='o-', linewidth=2, markersize=6, capsize=5, capthick=1.5, label='Chains')
        axes[1, 1].errorbar(r_vacs_sorted, source_means, yerr=source_stds, 
                           fmt='s--', linewidth=2, markersize=6, capsize=5, capthick=1.5, label='Sources')
        axes[1, 1].errorbar(r_vacs_sorted, target_means, yerr=target_stds, 
                           fmt='^--', linewidth=2, markersize=6, capsize=5, capthick=1.5, label='Targets')
        axes[1, 1].set_xlabel('Vacancy Radius (mm)', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title('Chains, Sources, & Targets vs Vacancy Radius', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    #main()
    mainPlot()
    #mainRange()