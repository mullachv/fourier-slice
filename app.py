import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core import (
    SCIPY_OK,
    make_test_image,
    fft1_centered,
    fft2_centered,
    projection_via_rotation,
    find_most_different_theta,
    slice_from_F,
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fourier Slice Theorem Visualizer", layout="wide")
st.title("Fourier Slice Theorem (Central Slice Theorem) — Interactive Visualizer")

if not SCIPY_OK:
    st.error("This app needs SciPy for image rotation-based projections. Install: pip install scipy")
    st.stop()

with st.sidebar:
    st.header("Controls")
    N = st.slider("Grid size N", 96, 256, 160, step=16)
    seed = st.slider("Random seed", 0, 50, 1, step=1)
    subtract_mean = st.checkbox("Subtract mean from f(x,y) (reduces DC dominance)", value=True)

    theta1 = st.slider("θ₁ (degrees)", 0.0, 179.0, 35.0, step=1.0)

    auto_theta2 = st.checkbox("Auto-pick θ₂ most different from θ₁ (min correlation)", value=True)

    angle_step = st.slider("θ scan step (degrees) for auto θ₂", 1, 10, 2, step=1)
    if not auto_theta2:
        theta2 = st.slider("θ₂ (degrees)", 0.0, 179.0, 105.0, step=1.0)
    else:
        theta2 = None

    surface_step = st.slider("Fourier surface sparsity (bigger = lighter)", 1, 8, 3, step=1)
    slice_point_step = st.slider("Slice point sparsity (bigger = fewer points)", 1, 12, 3, step=1)
    st.markdown(
        "**Angle convention:** `θ` is measured counterclockwise from +x and is the projection normal / Fourier-slice direction, "
        r"i.e. $(k_x,k_y)=(\omega\cos\theta,\omega\sin\theta)$. "
        "The parallel integration lines in space are perpendicular to this direction."
    )
    st.subheader("Plot 1: projection lines")
    show_theta1_lines = st.radio("Emphasize lines", ["θ₁", "θ₂"], horizontal=True, index=0) == "θ₁"

# Build f
f, x, y, X, Y = make_test_image(N=N, seed=seed)
if subtract_mean:
    f = f - f.mean()

# Choose theta2 if auto
if auto_theta2:
    angles = np.arange(0.0, 180.0, float(angle_step))
    theta2, corr = find_most_different_theta(f, theta1, angles)
    st.sidebar.caption(f"Auto θ₂ = {theta2:.1f}° (corr ≈ {corr:.3f} vs θ₁)")
else:
    corr = None

# Projections + FFTs
p1 = projection_via_rotation(f, theta1)
p2 = projection_via_rotation(f, theta2)

P1 = fft1_centered(p1)
P2 = fft1_centered(p2)

F = fft2_centered(f)
F_log = np.log1p(np.abs(F))

freq = np.fft.fftshift(np.fft.fftfreq(N, d=1.0))

# RHS slices from the SAME log surface
kx1, ky1, rhs1, valid1 = slice_from_F(F_log, theta1, freq)
kx2, ky2, rhs2, valid2 = slice_from_F(F_log, theta2, freq)

lhs1 = np.log1p(np.abs(P1))
lhs2 = np.log1p(np.abs(P2))

# ----------------------------
# Layout
# ----------------------------
colA, colB = st.columns([1, 1])

# ---- Plot 1: f(x,y) as 3D surface + projection lines for θ₁ and θ₂ (one set emphasized)
with colA:
    st.subheader("1) Spatial domain: f(x,y) as a 3D surface + projection lines (θ₁ & θ₂)")
    st.caption(
        "Convention used here: θ is measured counterclockwise from +x and is the normal to projection lines "
        "(and the Fourier-slice angle). "
        "The drawn line family corresponds to constant-t lines, perpendicular to (cosθ, sinθ)."
    )

    t_indices = np.linspace(int(0.2*N), int(0.8*N), 5).astype(int)
    base_z = float(np.min(f) - 0.35*(np.max(f)-np.min(f)))
    pc_grid = np.linspace(-(N-1)/2, (N-1)/2, N)
    # Extend lines past grid so they remain hoverable outside the surface
    margin_data = 0.45
    margin_pc = margin_data * (N - 1) / 2.0
    ys_pc = np.linspace(pc_grid[0] - margin_pc, pc_grid[-1] + margin_pc, 280)
    pc_grid_ext = np.linspace(pc_grid[0] - margin_pc, pc_grid[-1] + margin_pc, N + 2 * max(1, int(margin_pc)))
    x_ext = np.linspace(-1 - margin_data, 1 + margin_data, len(pc_grid_ext))
    y_ext = np.linspace(-1 - margin_data, 1 + margin_data, len(pc_grid_ext))

    def lines_at_angle(theta_deg, p, label, emphasized, legend_name, legend_group, color, initially_visible):
        th = np.deg2rad(theta_deg)
        c, s = np.cos(th), np.sin(th)
        line_style = dict(width=7 if emphasized else 5, color=color)
        for i, t_idx in enumerate(t_indices):
            xp_pc = (t_idx - (N-1)/2)
            # constant-t lines for x cos(theta) + y sin(theta) = t
            xpc = c * (xp_pc * np.ones_like(ys_pc)) - s * ys_pc
            ypc = s * (xp_pc * np.ones_like(ys_pc)) + c * ys_pc
            # Use extended bounds so lines run past the surface and stay clickable
            mask = (xpc >= pc_grid[0] - margin_pc) & (xpc <= pc_grid[-1] + margin_pc) & (ypc >= pc_grid[0] - margin_pc) & (ypc <= pc_grid[-1] + margin_pc)
            x_draw = np.interp(xpc[mask], pc_grid_ext, x_ext)
            y_draw = np.interp(ypc[mask], pc_grid_ext, y_ext)
            n = len(x_draw)
            hover_text = f"{label} t={t_idx}<br>Σ pθ(t) = {float(p[t_idx]):.4f}"
            fig1.add_trace(go.Scatter3d(
                x=x_draw, y=y_draw, z=np.full_like(x_draw, base_z),
                mode="lines",
                line=line_style,
                text=[hover_text] * n,
                hoverinfo="text",
                name=legend_name,
                legendgroup=legend_group,
                showlegend=(i == 0),
                visible=True if initially_visible else "legendonly",
            ))

    def add_theta_normal_guide(theta_deg, label, color, legend_group, initially_visible):
        th = np.deg2rad(theta_deg)
        c, s = np.cos(th), np.sin(th)
        # Long bidirectional guide for the normal direction (cos(theta), sin(theta)).
        # Keep at projection-line plane (tiny offset prevents z-fighting).
        z_guide = base_z + 0.01 * (np.max(f) - np.min(f))
        L = 1.0 + margin_data
        lambdas = np.linspace(-L, L, 19)
        xg = lambdas * c
        yg = lambdas * s
        fig1.add_trace(go.Scatter3d(
            x=xg,
            y=yg,
            z=np.full_like(xg, z_guide),
            mode="lines",
            line=dict(color=color, width=6, dash="longdash"),
            hoverinfo="skip",
            name=label,
            showlegend=True,
            legendgroup=legend_group,
            visible=True if initially_visible else "legendonly",
        ))

    fig1 = go.Figure()
    fig1.add_trace(go.Surface(
        x=X, y=Y, z=f,
        opacity=0.25,
        showscale=False,
        colorscale=[[0.0, "#f4f1ea"], [1.0, "#c8c2b0"]],
        lighting=dict(ambient=0.75, diffuse=0.5, specular=0.08, roughness=0.95, fresnel=0.05),
        lightposition=dict(x=120, y=-160, z=220),
        name="f(x,y)",
        hovertemplate="f(x,y)<br>x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>"
    ))

    lines_at_angle(
        theta1,
        p1,
        f"θ₁={theta1:.0f}°",
        emphasized=show_theta1_lines,
        legend_name="θ₁ lines",
        legend_group="theta1",
        color="#1f77b4",
        initially_visible=show_theta1_lines,
    )
    add_theta_normal_guide(
        theta1,
        f"θ₁ normal direction ({theta1:.0f}°)",
        color="#1f77b4",
        legend_group="theta1",
        initially_visible=show_theta1_lines,
    )
    lines_at_angle(
        theta2,
        p2,
        f"θ₂={theta2:.0f}°",
        emphasized=not show_theta1_lines,
        legend_name="θ₂ lines",
        legend_group="theta2",
        color="#d62728",
        initially_visible=not show_theta1_lines,
    )
    add_theta_normal_guide(
        theta2,
        f"θ₂ normal direction ({theta2:.0f}°)",
        color="#d62728",
        legend_group="theta2",
        initially_visible=not show_theta1_lines,
    )

    fig1.update_layout(
        height=520,
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            camera=dict(
                eye=dict(x=0.05, y=-2.2, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig1, width="stretch")

# ---- Plot 2: projections pθ(t)
with colB:
    st.subheader("2) Projections: pθ(t) for θ₁ and θ₂")
    t = np.arange(N)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=p1, mode="lines", name=f"pθ(t), θ₁={theta1:.1f}°"))
    fig2.add_trace(go.Scatter(x=t, y=p2, mode="lines", name=f"pθ(t), θ₂={theta2:.1f}°"))
    fig2.add_trace(go.Scatter(x=t_indices, y=p1[t_indices], mode="markers", name="sampled t's (θ₁)"))

    fig2.update_layout(
        height=520,
        xaxis_title="t (pixel index)",
        yaxis_title="pθ(t)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=10, t=10, b=30),
    )
    st.plotly_chart(fig2, width="stretch")

colC, colD = st.columns([1, 1])

# ---- Plot 3: LHS log magnitude
with colC:
    st.subheader("3) LHS: log(1 + |Pθ(ω)|) for θ₁ and θ₂")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=freq, y=lhs1, mode="lines", name=f"LHS θ₁={theta1:.1f}°"))
    fig3.add_trace(go.Scatter(x=freq, y=lhs2, mode="lines", name=f"LHS θ₂={theta2:.1f}°"))
    fig3.update_layout(
        height=420,
        xaxis_title="ω (cycles/pixel)",
        yaxis_title="log(1+|Pθ(ω)|)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=10, t=10, b=30),
    )
    st.plotly_chart(fig3, width="stretch")

# ---- Plot 4: RHS Fourier surface + slice lines with points
with colD:
    st.subheader("4) RHS: log(1 + |F(kx,ky)|) surface + slice lines (points match Plot 3)")
    step = int(surface_step)
    F_ds = F_log[::step, ::step]
    KX, KY = np.meshgrid(freq, freq, indexing="xy")
    KX_ds = KX[::step, ::step]
    KY_ds = KY[::step, ::step]

    fig4 = go.Figure()
    fig4.add_trace(go.Surface(
        x=KX_ds,
        y=KY_ds,
        z=F_ds,
        opacity=0.25,
        showscale=False,
        colorscale=[[0.0, "#f4f1ea"], [1.0, "#c8c2b0"]],
        lighting=dict(ambient=0.75, diffuse=0.5, specular=0.08, roughness=0.95, fresnel=0.05),
        lightposition=dict(x=120, y=-160, z=220),
        name="log(1+|F|) surface",
    ))

    # Slice lines + sparse points (same rhs arrays)
    def add_slice(fig, kx, ky, rhs, valid, name, color):
        fig.add_trace(go.Scatter3d(
            x=kx[valid], y=ky[valid], z=rhs[valid],
            mode="lines", line=dict(width=7, color=color),
            name=name
        ))
        keep = np.where(valid)[0][::int(slice_point_step)]
        fig.add_trace(go.Scatter3d(
            x=kx[keep], y=ky[keep], z=rhs[keep],
            mode="markers", marker=dict(size=3, color=color),
            name=f"{name} (pts)"
        ))

    add_slice(fig4, kx1, ky1, rhs1, valid1, f"slice θ₁={theta1:.1f}°", "#1f77b4")
    add_slice(fig4, kx2, ky2, rhs2, valid2, f"slice θ₂={theta2:.1f}°", "#d62728")

    fig4.update_layout(
        height=420,
        scene=dict(
            xaxis_title="kx", yaxis_title="ky", zaxis_title="log(1+|F|)",
            camera=dict(
                eye=dict(x=0.05, y=-2.2, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig4, width="stretch")

# ---- Bonus: Direct match overlay (this is the money plot)
st.subheader("Direct match check: LHS log(1+|Pθ(ω)|) vs RHS log(1+|F(ωcosθ, ωsinθ)|)")
fig5 = go.Figure()

# RHS as function of ω is rhs(ω) already (sampled on freq grid)
fig5.add_trace(go.Scatter(x=freq, y=lhs1, mode="lines", name=f"LHS θ₁={theta1:.1f}°"))
fig5.add_trace(go.Scatter(x=freq[valid1], y=rhs1[valid1], mode="lines", line=dict(dash="dash"),
                          name=f"RHS slice θ₁={theta1:.1f}°"))

fig5.add_trace(go.Scatter(x=freq, y=lhs2, mode="lines", name=f"LHS θ₂={theta2:.1f}°"))
fig5.add_trace(go.Scatter(x=freq[valid2], y=rhs2[valid2], mode="lines", line=dict(dash="dash"),
                          name=f"RHS slice θ₂={theta2:.1f}°"))

fig5.update_layout(
    height=420,
    xaxis_title="ω (cycles/pixel)",
    yaxis_title="log magnitude",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=30, r=10, t=10, b=30),
)
st.plotly_chart(fig5, width="stretch")

st.divider()
st.subheader("Test signal f(x,y)")
st.latex(
    r"f(x,y) = \sum_{k=1}^{3} A_k \exp\left(-\frac{(x-\mu_{x,k})^2}{2\sigma_{x,k}^2} - \frac{(y-\mu_{y,k})^2}{2\sigma_{y,k}^2}\right) + B\cos(2\pi(2.2x+0.6y)) + \varepsilon"
)
st.caption(
    "**Description:** Anisotropic mixture of three bivariate Gaussians (different centers and axis-aligned covariances), plus an oriented sinusoid to inject directional spectral structure, plus small additive Gaussian noise. "
    "This demo uses a rotation+sum approximation to the Radon projection (discrete); small discrepancies come from interpolation during rotation and discretization."
)
