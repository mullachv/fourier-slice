"""Core Fourier-slice and projection logic (testable, no Streamlit)."""
import numpy as np

try:
    from scipy.ndimage import rotate as ndi_rotate
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def make_test_image(N=160, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, N, endpoint=False)
    y = np.linspace(-1, 1, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    f = (
        1.35 * np.exp(-((X - 0.25) ** 2 / (2 * 0.10**2) + (Y + 0.10) ** 2 / (2 * 0.18**2)))
        + 0.95 * np.exp(-((X + 0.32) ** 2 / (2 * 0.24**2) + (Y - 0.27) ** 2 / (2 * 0.08**2)))
        + 0.55 * np.exp(-((X + 0.02) ** 2 / (2 * 0.06**2) + (Y + 0.38) ** 2 / (2 * 0.06**2)))
    )
    f += 0.20 * np.cos(2 * np.pi * (7.0 * X + 1.7 * Y))
    f += 0.03 * rng.normal(size=f.shape)
    return f, x, y, X, Y


def fft1_centered(x):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def fft2_centered(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


def projection_via_rotation(f, theta_deg):
    """
    Approximate Radon projection:
      rotate image by theta so lines at angle theta become vertical columns, then sum columns.
    """
    if not SCIPY_OK:
        raise RuntimeError(
            "SciPy not available: install scipy or use a different projection method."
        )
    fr = ndi_rotate(
        f, angle=theta_deg, reshape=False, order=1, mode="constant", cval=0.0
    )
    return fr.sum(axis=0)


def bilinear_sample(img, xs, ys):
    """
    Bilinear sample img at fractional indices xs, ys (index coordinates 0..N-1).
    Returns sampled values and a valid mask.
    """
    N, M = img.shape
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < M) & (y1 < N)
    out = np.zeros_like(xs, dtype=float)

    xv = xs[valid]
    yv = ys[valid]
    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]

    Ia = img[y0v, x0v]
    Ib = img[y0v, x1v]
    Ic = img[y1v, x0v]
    Id = img[y1v, x1v]

    wa = (x1v - xv) * (y1v - yv)
    wb = (xv - x0v) * (y1v - yv)
    wc = (x1v - xv) * (yv - y0v)
    wd = (xv - x0v) * (yv - y0v)

    out[valid] = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out, valid


def find_most_different_theta(f, theta1_deg, angles_deg):
    """
    Pick theta2 that makes p_theta(t) as different as possible from p_theta1(t).
    Metric: minimum correlation after centering + normalization.
    """
    p1 = projection_via_rotation(f, theta1_deg).astype(float)
    p1 = p1 - p1.mean()
    p1 = p1 / (np.linalg.norm(p1) + 1e-12)

    best_theta = None
    best_corr = 1.0

    for th in angles_deg:
        if abs(th - theta1_deg) < 1e-9:
            continue
        p2 = projection_via_rotation(f, th).astype(float)
        p2 = p2 - p2.mean()
        p2 = p2 / (np.linalg.norm(p2) + 1e-12)
        corr = float(np.dot(p1, p2))
        if corr < best_corr:
            best_corr = corr
            best_theta = th

    return best_theta, best_corr


def slice_from_F(F_logmag, theta_deg, freq):
    """
    Sample RHS slice: log(1+|F(kx,ky)|) along (kx,ky)=(w cosθ, w sinθ).
    Returns rhs(w), valid mask.
    """
    th = np.deg2rad(theta_deg)
    w = freq
    kx = w * np.cos(th)
    ky = w * np.sin(th)

    df = freq[1] - freq[0]
    fmin = freq[0]
    x_idx = (kx - fmin) / df
    y_idx = (ky - fmin) / df
    rhs, valid = bilinear_sample(F_logmag, x_idx, y_idx)
    return kx, ky, rhs, valid
