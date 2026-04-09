"""
Traveling Salesman Problem (TSP) Solver
A production-ready Streamlit application for solving and visualizing TSP.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools
import time
import io
from math import factorial

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TSP Solver",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  – dark industrial dashboard
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---------- Root tokens ---------- */
:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #252a35;
    --accent:    #00e5ff;
    --accent2:   #ff6b35;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --success:   #22c55e;
    --warning:   #f59e0b;
    --error:     #ef4444;
    --radius:    8px;
}

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ---------- Main panel ---------- */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1280px;
}

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,229,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(255,107,53,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
    line-height: 1.2;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ---------- Metric cards ---------- */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 150px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-value.orange { color: var(--accent2); }
.metric-value.green  { color: var(--success); }

/* ---------- Section headers ---------- */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* ---------- Route badge ---------- */
.route-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    color: var(--text);
    word-break: break-all;
    margin-bottom: 1rem;
}
.route-box .arrow { color: var(--accent2); margin: 0 0.25rem; }

/* ---------- Info / warning banners ---------- */
.banner {
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    margin-bottom: 1rem;
    border: 1px solid;
}
.banner.info    { background: rgba(0,229,255,0.07); border-color: rgba(0,229,255,0.25); color: #a5f3fc; }
.banner.warning { background: rgba(245,158,11,0.07); border-color: rgba(245,158,11,0.3); color: #fde68a; }
.banner.success { background: rgba(34,197,94,0.07); border-color: rgba(34,197,94,0.3); color: #86efac; }
.banner.error   { background: rgba(239,68,68,0.07); border-color: rgba(239,68,68,0.3); color: #fca5a5; }

/* ---------- Streamlit widget overrides ---------- */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* secondary button */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextArea textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.stDataFrame { background: var(--surface) !important; }
.stAlert { border-radius: var(--radius) !important; }

/* file uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    padding: 0.6rem 1.25rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.1) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UTILITY / MATH HELPERS
# ─────────────────────────────────────────────

def euclidean_dist(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_distance_matrix(coords):
    """Build n×n distance matrix from list of (x, y) tuples."""
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = euclidean_dist(coords[i], coords[j])
    return D


def route_distance(route, D):
    """Total distance of a round-trip route given distance matrix D."""
    total = 0.0
    n = len(route)
    for i in range(n):
        total += D[route[i]][route[(i + 1) % n]]
    return total


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────

def validate_coords(coords):
    """Return (ok, error_msg) for a list of (x, y) tuples."""
    if len(coords) < 2:
        return False, "Need at least 2 locations."
    for i, c in enumerate(coords):
        if c is None or len(c) != 2:
            return False, f"Location {i} has invalid coordinates."
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in c):
            return False, f"Location {i} contains non-numeric or infinite values."
    return True, ""


def validate_matrix(mat):
    """Return (ok, error_msg) for a 2-D numpy array."""
    if mat.ndim != 2:
        return False, "Matrix must be 2-dimensional."
    n, m = mat.shape
    if n != m:
        return False, f"Matrix must be square (got {n}×{m})."
    if n < 2:
        return False, "Matrix must be at least 2×2."
    if not np.isfinite(mat).all():
        return False, "Matrix contains NaN or infinite values."
    if np.any(mat < 0):
        return False, "Distance values must be non-negative."
    return True, ""


# ─────────────────────────────────────────────
#  TSP SOLVERS
# ─────────────────────────────────────────────

def solve_brute_force(D):
    """
    Exact brute-force TSP (O(n!)).
    Fixes node 0 as start, permutes the rest.
    Returns (best_route, best_dist).
    """
    n = len(D)
    nodes = list(range(1, n))
    best_dist = float("inf")
    best_route = None

    for perm in itertools.permutations(nodes):
        route = [0] + list(perm)
        dist = route_distance(route, D)
        if dist < best_dist:
            best_dist = dist
            best_route = route

    return best_route, best_dist


def solve_nearest_neighbor(D, start=0):
    """
    Nearest-neighbour greedy heuristic.
    Returns (route, total_dist).
    """
    n = len(D)
    visited = [False] * n
    route = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = route[-1]
        nearest = None
        nearest_dist = float("inf")
        for j in range(n):
            if not visited[j] and D[current][j] < nearest_dist:
                nearest_dist = D[current][j]
                nearest = j
        route.append(nearest)
        visited[nearest] = True

    return route, route_distance(route, D)


def two_opt_improve(route, D, max_iter=1000):
    """
    2-opt local-search improvement of a route.
    Returns (improved_route, improved_dist).
    """
    best = route[:]
    best_dist = route_distance(best, D)
    improved = True
    iterations = 0

    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        n = len(best)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                new_dist = route_distance(new_route, D)
                if new_dist < best_dist - 1e-10:
                    best = new_route
                    best_dist = new_dist
                    improved = True

    return best, best_dist


def solve_heuristic(D, use_2opt=True):
    """
    Nearest-neighbour + optional 2-opt.
    Tries all starting nodes and keeps the best.
    Returns (route, dist, label).
    """
    n = len(D)
    best_route, best_dist = None, float("inf")

    for start in range(n):
        route, dist = solve_nearest_neighbor(D, start)
        if dist < best_dist:
            best_dist = dist
            best_route = route

    label = "Nearest Neighbour"
    if use_2opt:
        best_route, best_dist = two_opt_improve(best_route, D)
        label = "Nearest Neighbour + 2-opt"

    return best_route, best_dist, label


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def make_plotly_figure(coords, route, labels, title="TSP Route"):
    """
    Build a Plotly figure showing nodes and the TSP path.
    coords  – list of (x, y)
    route   – ordered list of node indices (without repeated start)
    labels  – list of node name strings
    """
    # Build ordered coordinates including return to start
    xs = [coords[i][0] for i in route] + [coords[route[0]][0]]
    ys = [coords[i][1] for i in route] + [coords[route[0]][1]]

    fig = go.Figure()

    # ── Edge / path ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="#00e5ff", width=2.5, dash="solid"),
        name="Route",
        hoverinfo="skip",
    ))

    # ── Regular nodes ────────────────────────────────────────────────────
    node_x = [coords[i][0] for i in route[1:]]
    node_y = [coords[i][1] for i in route[1:]]
    node_labels = [labels[i] for i in route[1:]]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=14, color="#ffffff", line=dict(color="#00e5ff", width=2)),
        text=node_labels,
        textposition="top center",
        textfont=dict(color="#e8eaf0", size=12, family="Space Mono"),
        name="Locations",
    ))

    # ── Start node ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[coords[route[0]][0]], y=[coords[route[0]][1]],
        mode="markers+text",
        marker=dict(size=20, color="#ff6b35",
                    symbol="star", line=dict(color="#ffffff", width=1.5)),
        text=[labels[route[0]]],
        textposition="top center",
        textfont=dict(color="#ff6b35", size=13, family="Space Mono"),
        name="Start / End",
    ))

    # ── Direction arrows (annotations) ───────────────────────────────────
    for k in range(len(route)):
        src = route[k]
        dst = route[(k + 1) % len(route)]
        mx = (coords[src][0] + coords[dst][0]) / 2
        my = (coords[src][1] + coords[dst][1]) / 2
        dx = coords[dst][0] - coords[src][0]
        dy = coords[dst][1] - coords[src][1]
        fig.add_annotation(
            x=mx + dx * 0.01, y=my + dy * 0.01,
            ax=mx - dx * 0.01, ay=my - dy * 0.01,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=1.8, arrowcolor="#00e5ff",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono", size=14, color="#6b7280")),
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#151820",
        font=dict(color="#e8eaf0"),
        xaxis=dict(gridcolor="#252a35", zerolinecolor="#252a35", title="X"),
        yaxis=dict(gridcolor="#252a35", zerolinecolor="#252a35", title="Y"),
        legend=dict(bgcolor="#151820", bordercolor="#252a35", borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
    )
    return fig


def make_distance_heatmap(D, labels):
    """Plotly heatmap of the distance matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=D, x=labels, y=labels,
        colorscale=[[0, "#151820"], [0.5, "#0e4b5c"], [1, "#00e5ff"]],
        text=np.round(D, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="From %{y} → To %{x}<br>Distance: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#151820",
        font=dict(color="#e8eaf0", family="DM Sans"),
        margin=dict(l=60, r=20, t=40, b=60),
        height=420,
        xaxis=dict(title="To"),
        yaxis=dict(title="From", autorange="reversed"),
    )
    return fig


# ─────────────────────────────────────────────
#  RESULTS EXPORT
# ─────────────────────────────────────────────

def build_results_csv(route, labels, coords, total_dist, method, elapsed):
    """Return a CSV bytes object summarising the solution."""
    rows = []
    for step, idx in enumerate(route):
        rows.append({
            "Step": step + 1,
            "Node Index": idx,
            "Location": labels[idx],
            "X": coords[idx][0],
            "Y": coords[idx][1],
        })
    # Return-to-start row
    rows.append({
        "Step": len(route) + 1,
        "Node Index": route[0],
        "Location": labels[route[0]] + " (return)",
        "X": coords[route[0]][0],
        "Y": coords[route[0]][1],
    })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = ["", "", "", "", ""]
    df.loc[len(df)] = ["Total Distance", f"{total_dist:.4f}", "", "", ""]
    df.loc[len(df)] = ["Method", method, "", "", ""]
    df.loc[len(df)] = ["Solve Time (s)", f"{elapsed:.4f}", "", "", ""]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "result": None,      # dict with route, dist, method, elapsed, labels, coords, D
        "theme": "dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <p class="hero-title">🗺️ TSP <span>Solver</span></p>
  <p class="hero-sub">Traveling Salesman Problem · Exact &amp; Heuristic Methods · Interactive Visualisation</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # ── Input method ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Input Method</p>', unsafe_allow_html=True)
    input_method = st.selectbox(
        "Choose input mode",
        ["Manual Coordinates", "Distance Matrix", "CSV Upload"],
        label_visibility="collapsed",
    )

    # ── Solver method ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Solver</p>', unsafe_allow_html=True)
    solver_choice = st.selectbox(
        "Algorithm",
        ["Auto-select", "Brute Force (Exact)", "Nearest Neighbour + 2-opt (Heuristic)"],
        label_visibility="collapsed",
    )

    use_2opt = True
    if solver_choice == "Nearest Neighbour + 2-opt (Heuristic)":
        use_2opt = st.checkbox("Enable 2-opt improvement", value=True)

    # ── Viz options ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Visualisation</p>', unsafe_allow_html=True)
    show_heatmap = st.checkbox("Show distance matrix heatmap", value=True)
    show_dm_table = st.checkbox("Show distance matrix table", value=False)

    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280'>Built with Streamlit · Plotly · NumPy · Pandas</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  MAIN INPUT PANEL
# ─────────────────────────────────────────────

col_input, col_results = st.columns([1, 1.6], gap="large")

coords = []
labels = []
D_input = None   # will hold distance matrix if that mode selected
error_msg = ""

with col_input:
    st.markdown('<p class="section-header">📍 Input Locations</p>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  A. MANUAL COORDINATES
    # ══════════════════════════════════════════
    if input_method == "Manual Coordinates":
        n_loc = st.number_input("Number of locations", min_value=2, max_value=20, value=5, step=1)

        st.markdown("**Enter coordinates (x, y) for each location:**")

        # Build a grid-like input table using two columns
        for i in range(int(n_loc)):
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                lbl = st.text_input(f"Name", value=f"City {i}", key=f"lbl_{i}", label_visibility="visible" if i == 0 else "collapsed")
            with c2:
                x = st.number_input("X", value=float(np.random.randint(0, 100)) if f"x_{i}" not in st.session_state else st.session_state[f"x_{i}"],
                                    key=f"x_{i}", label_visibility="visible" if i == 0 else "collapsed")
            with c3:
                y = st.number_input("Y", value=float(np.random.randint(0, 100)) if f"y_{i}" not in st.session_state else st.session_state[f"y_{i}"],
                                    key=f"y_{i}", label_visibility="visible" if i == 0 else "collapsed")
            labels.append(lbl if lbl.strip() else f"City {i}")
            coords.append((x, y))

    # ══════════════════════════════════════════
    #  B. DISTANCE MATRIX
    # ══════════════════════════════════════════
    elif input_method == "Distance Matrix":
        st.markdown(
            '<div class="banner info">Paste a space- or comma-separated square matrix. '
            'Each row on a new line. Diagonal should be 0.</div>',
            unsafe_allow_html=True,
        )
        default_mat = "0 10 15 20\n10 0 35 25\n15 35 0 30\n20 25 30 0"
        mat_text = st.text_area("Distance Matrix", value=default_mat, height=160)

        # Parse
        try:
            rows = []
            for line in mat_text.strip().splitlines():
                line = line.replace(",", " ")
                row = [float(v) for v in line.split()]
                rows.append(row)
            D_input = np.array(rows)
        except Exception as e:
            error_msg = f"Could not parse matrix: {e}"

        if D_input is not None and not error_msg:
            ok, err = validate_matrix(D_input)
            if not ok:
                error_msg = err
            else:
                n = len(D_input)
                for i in range(n):
                    lbl = st.text_input(f"Node {i} name", value=f"City {i}", key=f"mlbl_{i}")
                    labels.append(lbl.strip() or f"City {i}")
                # Fake coords for visualisation (MDS-like – use first two PCA components)
                # Simple approach: place nodes using metric MDS approximation
                from numpy.linalg import eigh
                n = len(D_input)
                D2 = D_input ** 2
                J = np.eye(n) - np.ones((n, n)) / n
                B = -0.5 * J @ D2 @ J
                eigvals, eigvecs = eigh(B)
                idx = np.argsort(eigvals)[::-1]
                eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
                pos_vals = np.maximum(eigvals[:2], 0)
                emb = eigvecs[:, :2] * np.sqrt(pos_vals)
                coords = [(float(emb[i, 0]), float(emb[i, 1])) for i in range(n)]

    # ══════════════════════════════════════════
    #  C. CSV UPLOAD
    # ══════════════════════════════════════════
    elif input_method == "CSV Upload":
        st.markdown(
            '<div class="banner info">Upload a CSV with columns: <b>Location, X, Y</b></div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                df_up.columns = df_up.columns.str.strip()
                required = {"Location", "X", "Y"}
                missing_cols = required - set(df_up.columns)
                if missing_cols:
                    error_msg = f"CSV missing columns: {missing_cols}"
                else:
                    df_up = df_up.dropna(subset=["Location", "X", "Y"])
                    labels = df_up["Location"].astype(str).tolist()
                    coords = list(zip(df_up["X"].astype(float), df_up["Y"].astype(float)))
                    st.success(f"✅ Loaded {len(labels)} locations.")
                    st.dataframe(df_up[["Location", "X", "Y"]].reset_index(drop=True), height=180)
            except Exception as e:
                error_msg = f"Error reading CSV: {e}"
        else:
            # Sample CSV preview
            sample = pd.DataFrame({
                "Location": ["Depot", "Paris", "Lyon", "Marseille", "Bordeaux"],
                "X": [2.35, 2.35, 4.83, 5.37, -0.58],
                "Y": [48.85, 48.85, 45.75, 43.30, 44.84],
            })
            st.markdown("**Sample CSV format:**")
            st.dataframe(sample, hide_index=True)
            csv_sample = sample.to_csv(index=False).encode()
            st.download_button("⬇️ Download sample CSV", csv_sample, "sample_tsp.csv", "text/csv")


# ─────────────────────────────────────────────
#  SOLVE BUTTON ROW
# ─────────────────────────────────────────────

btn_col1, btn_col2 = st.columns([1, 5])
with btn_col1:
    solve_btn = st.button("▶  Solve TSP", use_container_width=True)
with btn_col2:
    reset_btn = st.button("↺  Reset", use_container_width=False)

if reset_btn:
    st.session_state["result"] = None
    st.rerun()


# ─────────────────────────────────────────────
#  SOLVE LOGIC
# ─────────────────────────────────────────────

if solve_btn:
    # Clear previous
    st.session_state["result"] = None

    # ── Gather / validate ────────────────────────────────────────────────
    if error_msg:
        st.markdown(f'<div class="banner error">❌ {error_msg}</div>', unsafe_allow_html=True)

    elif input_method == "Distance Matrix":
        if D_input is None:
            st.markdown('<div class="banner error">❌ Please provide a valid distance matrix.</div>', unsafe_allow_html=True)
        else:
            ok, err = validate_matrix(D_input)
            if not ok:
                st.markdown(f'<div class="banner error">❌ {err}</div>', unsafe_allow_html=True)
            else:
                D = D_input
                n = len(D)
                # ── Auto-select method ─────────────────────────────────────
                if solver_choice == "Auto-select":
                    method_used = "brute_force" if n <= 10 else "heuristic"
                elif solver_choice == "Brute Force (Exact)":
                    method_used = "brute_force"
                else:
                    method_used = "heuristic"

                if method_used == "brute_force" and n > 10:
                    st.markdown(
                        '<div class="banner warning">⚠️ Brute force is disabled for n > 10. '
                        'Switching to Nearest Neighbour + 2-opt.</div>',
                        unsafe_allow_html=True,
                    )
                    method_used = "heuristic"

                with st.spinner("Solving TSP…"):
                    t0 = time.perf_counter()
                    if method_used == "brute_force":
                        route, dist = solve_brute_force(D)
                        algo_label = "Brute Force (Exact)"
                    else:
                        route, dist, algo_label = solve_heuristic(D, use_2opt)
                    elapsed = time.perf_counter() - t0

                st.session_state["result"] = {
                    "route": route, "dist": dist, "method": algo_label,
                    "elapsed": elapsed, "labels": labels,
                    "coords": coords, "D": D,
                }

    else:
        # Manual or CSV: need coords
        if not coords:
            st.markdown('<div class="banner error">❌ No coordinates found. Please provide input.</div>', unsafe_allow_html=True)
        else:
            ok, err = validate_coords(coords)
            if not ok:
                st.markdown(f'<div class="banner error">❌ {err}</div>', unsafe_allow_html=True)
            else:
                n = len(coords)
                D = build_distance_matrix(coords)

                # ── Auto-select method ─────────────────────────────────────
                if solver_choice == "Auto-select":
                    method_used = "brute_force" if n <= 10 else "heuristic"
                elif solver_choice == "Brute Force (Exact)":
                    method_used = "brute_force"
                else:
                    method_used = "heuristic"

                if method_used == "brute_force" and n > 10:
                    st.markdown(
                        '<div class="banner warning">⚠️ Brute force disabled for n > 10. '
                        'Switching to Nearest Neighbour + 2-opt automatically.</div>',
                        unsafe_allow_html=True,
                    )
                    method_used = "heuristic"

                if method_used == "brute_force" and n > 10:
                    method_used = "heuristic"

                est_perms = factorial(n - 1) if n > 1 else 1
                if method_used == "brute_force":
                    st.markdown(
                        f'<div class="banner info">🔍 Brute force: evaluating '
                        f'{est_perms:,} permutations…</div>',
                        unsafe_allow_html=True,
                    )

                with st.spinner("Solving TSP…"):
                    t0 = time.perf_counter()
                    if method_used == "brute_force":
                        route, dist = solve_brute_force(D)
                        algo_label = "Brute Force (Exact)"
                    else:
                        route, dist, algo_label = solve_heuristic(D, use_2opt)
                    elapsed = time.perf_counter() - t0

                st.session_state["result"] = {
                    "route": route, "dist": dist, "method": algo_label,
                    "elapsed": elapsed, "labels": labels if labels else [f"City {i}" for i in range(n)],
                    "coords": coords, "D": D,
                }


# ─────────────────────────────────────────────
#  RESULTS PANEL
# ─────────────────────────────────────────────

res = st.session_state.get("result")

if res is None:
    # Placeholder
    with col_results:
        st.markdown('<p class="section-header">📊 Results</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="banner info">Results will appear here after clicking <b>▶ Solve TSP</b>.</div>',
            unsafe_allow_html=True,
        )
else:
    route    = res["route"]
    dist     = res["dist"]
    method   = res["method"]
    elapsed  = res["elapsed"]
    lbls     = res["labels"]
    crds     = res["coords"]
    D        = res["D"]
    n        = len(route)

    with col_results:
        st.markdown('<p class="section-header">📊 Results</p>', unsafe_allow_html=True)

        # ── Metric cards ──────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-label">Total Distance</div>
            <div class="metric-value">{dist:.2f}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Locations</div>
            <div class="metric-value orange">{n}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Solve Time</div>
            <div class="metric-value green">{elapsed*1000:.1f}<span style="font-size:0.8rem;color:#6b7280"> ms</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Algorithm badge ───────────────────────────────────────────────
        st.markdown(
            f'<div class="banner success">✅ Solved with <b>{method}</b></div>',
            unsafe_allow_html=True,
        )

        # ── Route display ─────────────────────────────────────────────────
        route_str = ' <span class="arrow">→</span> '.join(
            [f'<b>{lbls[i]}</b>' for i in route] + [f'<b>{lbls[route[0]]}</b>']
        )
        st.markdown(f'<div class="route-box">🛣️ &nbsp;{route_str}</div>', unsafe_allow_html=True)

        # ── Step-by-step table ────────────────────────────────────────────
        with st.expander("📋 Step-by-step breakdown"):
            rows = []
            full_tour = route + [route[0]]
            for k in range(len(full_tour) - 1):
                frm, to = full_tour[k], full_tour[k + 1]
                rows.append({
                    "Step": k + 1,
                    "From": lbls[frm],
                    "To": lbls[to],
                    "Leg Distance": round(D[frm][to], 4),
                })
            df_steps = pd.DataFrame(rows)
            df_steps.loc[len(df_steps)] = {
                "Step": "─", "From": "─", "To": "Total",
                "Leg Distance": round(dist, 4)
            }
            st.dataframe(df_steps, hide_index=True, use_container_width=True)

        # ── Download button ───────────────────────────────────────────────
        csv_bytes = build_results_csv(route, lbls, crds, dist, method, elapsed)
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv_bytes,
            file_name="tsp_results.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────
#  VISUALISATION TABS (full width below)
# ─────────────────────────────────────────────

if res is not None:
    route    = res["route"]
    dist     = res["dist"]
    lbls     = res["labels"]
    crds     = res["coords"]
    D        = res["D"]

    tab_titles = ["🗺️ Route Map"]
    if show_heatmap:
        tab_titles.append("🔥 Distance Heatmap")
    if show_dm_table:
        tab_titles.append("📐 Distance Matrix")

    tabs = st.tabs(tab_titles)

    # ── Route map ─────────────────────────────────────────────────────────
    with tabs[0]:
        fig_route = make_plotly_figure(
            crds, route, lbls,
            title=f"TSP Route — Total Distance: {dist:.2f}"
        )
        st.plotly_chart(fig_route, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────
    if show_heatmap:
        with tabs[1]:
            fig_heat = make_distance_heatmap(D, lbls)
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Matrix table ──────────────────────────────────────────────────────
    if show_dm_table:
        idx = 1 + (1 if show_heatmap else 0)
        with tabs[idx]:
            df_mat = pd.DataFrame(np.round(D, 3), index=lbls, columns=lbls)
            st.dataframe(df_mat, use_container_width=True)


# ─────────────────────────────────────────────
#  ABOUT EXPANDER
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About TSP & Algorithms"):
    st.markdown("""
**The Traveling Salesman Problem (TSP)** asks: given a list of cities and distances between them,
what is the shortest possible route that visits each city exactly once and returns to the starting city?

TSP is NP-hard — no known polynomial-time exact algorithm exists for large inputs.

| Method | Complexity | Best for |
|---|---|---|
| **Brute Force** | O((n-1)!) | n ≤ 10 |
| **Nearest Neighbour** | O(n²) | n ≤ 10,000 |
| **2-opt Improvement** | O(n² · iter) | Refining heuristic solutions |

**2-opt** works by repeatedly reversing sub-segments of the route until no swap improves the total distance.
This often yields solutions within 5% of optimal.
""")