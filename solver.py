import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Physical constant
c0 = 299792458.0

# --- Constants for External Validation (Imported by app.py) ---
# These are the limits for the PRODUCTION environment
PROD_MAX_N_EXP = 16
PROD_MAX_N = 2**PROD_MAX_N_EXP
PROD_MAX_STEPS = 1000
PROD_MAX_HEATMAP = 100

DEFAULT_PARAMS = {
    "N": 2**16,
    "time_window": 50.0,
    "lambda_0": 430.0,
    "pulse_shape": "sech2",
    "fwhm_nm": 5.0,
    "gdd": 0.0,
    "tod": 0.0,
    "fod": 0.0,
    "E": 100e-12,
    "beta2": 0.0,
    "beta3": 0.0,
    "beta4": 0.0,
    "alpha_db_per_km": 0.0,
    "use_nonlinear": True,
    "gamma": 0.001,
    "use_raman": False,
    "f_R": 0.18,
    "use_self_steepening": False,
    "L": 1.0,
    "n_steps": 500,        
    "n_heatmap_steps": 50, 
    "auto_time_range": True,
    "time_min": -2.0,
    "time_max": 2.0,
    "auto_lambda_range": True,
    "lambda_min": 400.0,
    "lambda_max": 460.0,
    "mfd": 8.0, 
    "n2": 2.6e-20,
}

# --------- Helpers ----------
def fwhm(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    y = y - np.min(y)
    if y.size < 2 or y.max() <= 0:
        return np.nan
    hm = 0.5 * y.max()
    s = np.sign(y - hm)
    crossings = np.where(np.diff(s) != 0)[0]
    if crossings.size == 0:
        return np.nan
    iL, iR = crossings[0], crossings[-1]
    def x_at_half(i):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i] - hm, y[i + 1] - hm
        return x1 - y1 * (x2 - x1) / (y2 - y1)
    return float(x_at_half(iR) - x_at_half(iL))

def fmt_time_si(seconds):
    ps = seconds * 1e12
    if ps < 1:
        return f"{seconds * 1e15:.2f} fs"
    if ps < 1000:
        return f"{ps:.3g} ps"
    return f"{seconds * 1e9:.3g} ns"

def fmt_nm(val_nm):
    return f"{val_nm:.3g} nm"

def process_plot(fig, ax, labels, z_data=None, dual_axis_type=None):
    if dual_axis_type:
        fig.subplots_adjust(left=0.15, right=0.92, top=0.85, bottom=0.15)
    else:
        fig.subplots_adjust(left=0.15, right=0.92, top=0.88, bottom=0.15)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100) 
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    pos = ax.get_position()
    
    meta = {
        "img": img_b64,
        "bounds": {
            "left": pos.x0,
            "bottom": pos.y0,
            "width": pos.width,
            "height": pos.height
        },
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "labels": labels,
        "dual_axis_type": dual_axis_type
    }

    if z_data is not None:
        rows, cols = z_data.shape
        r_step = max(1, rows // 100)
        c_step = max(1, cols // 100)
        small_z = z_data[::r_step, ::c_step]
        meta["z_grid"] = np.round(small_z, 2).tolist()

    return meta

# ------------------------ SSFM NLSE Solver -----------------------------------
def ssfm_nlse(A0, dt, dz, nz, beta2, beta3, beta4, gamma, w0, alpha=0.0,
              f_R=0.0, tau1=12.2e-15, tau2=32e-15, self_steepening=False,
              return_history=False, history_every=50):
    N = A0.size
    w = 2 * np.pi * np.fft.fftfreq(N, d=dt)

    dispersion_phase = (-1j * beta2 * (w**2) / 2.0 -1j * beta3 * (w**3) / 6.0 -1j * beta4 * (w**4) / 24.0)
    linear_step = np.exp((dispersion_phase - alpha / 2.0) * dz)

    H_w = None
    if f_R > 0.0:
        t_grid = np.arange(N) * dt
        norm_factor = (tau1**2 + tau2**2) / (tau1 * tau2**2)
        h_t = norm_factor * np.exp(-t_grid / tau2) * np.sin(t_grid / tau1)
        h_t[0] = 0.0
        if np.sum(h_t) > 0: h_t /= np.sum(h_t)
        H_w = np.fft.fft(h_t)

    steep_factor = (1.0 + w / w0) if self_steepening else 1.0
    A = A0.astype(np.complex128).copy()
    hist, z_hist = ([], [])

    if return_history:
        hist.append(A.copy())
        z_hist.append(0.0)

    def compute_nonlinear_term(Field):
        Intensity = np.abs(Field)**2
        if f_R > 0.0:
            I_w = np.fft.fft(Intensity)
            I_conv = np.fft.ifft(I_w * H_w).real
            I_eff = (1.0 - f_R) * Intensity + f_R * I_conv
        else:
            I_eff = Intensity
        P_NL = Field * I_eff
        P_NL_w = np.fft.fft(P_NL)
        RHS_w = -1j * gamma * steep_factor * P_NL_w
        return np.fft.ifft(RHS_w)

    def apply_nonlinearity(Field, step_m):
        if not self_steepening:
            Intensity = np.abs(Field)**2
            if f_R > 0.0:
                I_w = np.fft.fft(Intensity)
                I_conv = np.fft.ifft(I_w * H_w).real
                I_eff = (1.0 - f_R) * Intensity + f_R * I_conv
            else:
                I_eff = Intensity
            return Field * np.exp(-1j * gamma * I_eff * step_m)
        else:
            k1 = compute_nonlinear_term(Field)
            k2 = compute_nonlinear_term(Field + 0.5 * step_m * k1)
            k3 = compute_nonlinear_term(Field + 0.5 * step_m * k2)
            k4 = compute_nonlinear_term(Field + step_m * k3)
            return Field + (step_m / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    A = apply_nonlinearity(A, dz / 2.0)
    for k in range(nz):
        A = np.fft.ifft(np.fft.fft(A) * linear_step)
        if k < nz - 1: A = apply_nonlinearity(A, dz)
        else: A = apply_nonlinearity(A, dz / 2.0)
        if return_history and ((k + 1) % history_every == 0):
            hist.append(A.copy())
            z_hist.append((k + 1) * dz)
    if return_history: return A, hist, z_hist
    return A

def run_simulation(params):
    # NOTE: Logic checks for MAX_N, n_steps, etc are now handled in app.py
    # This function assumes if it is called, the parameters are valid for the current environment.
    
    N = int(params.get("N", DEFAULT_PARAMS["N"]))
    if N <= 0: raise ValueError("N must be positive.")

    time_window_ps = float(params.get("time_window", DEFAULT_PARAMS["time_window"]))
    if time_window_ps <= 0: raise ValueError("time_window must be positive.")
    time_window = time_window_ps * 1e-12 

    lambda_nm = float(params.get("lambda_0", DEFAULT_PARAMS["lambda_0"]))
    lambda_0 = lambda_nm * 1e-9

    pulse_shape = str(params.get("pulse_shape", DEFAULT_PARAMS["pulse_shape"])).lower()

    # --- Pulse Width Calculation ---
    fwhm_nm = float(params.get("fwhm_nm", DEFAULT_PARAMS["fwhm_nm"]))
    
    c = 299792458.0
    d_lambda_m = fwhm_nm * 1e-9
    d_nu_Hz = (c / (lambda_0**2)) * d_lambda_m
    
    if pulse_shape == "gaussian":
        tbp = 0.441
    else:
        tbp = 0.315
        
    if d_nu_Hz <= 1e-20:
         fwhm_fs = 1e3
    else:
        fwhm_s = tbp / d_nu_Hz
        fwhm_fs = fwhm_s * 1e15

    FWHM = fwhm_fs * 1e-15 # seconds
    
    # Dispersion parameters for input pulse
    gdd_val = float(params.get("gdd", 0.0)) # ps^2
    tod_val = float(params.get("tod", 0.0)) # ps^3
    fod_val = float(params.get("fod", 0.0)) # ps^4
    
    gdd = gdd_val * (1e-12)**2
    tod = tod_val * (1e-12)**3
    fod = fod_val * (1e-12)**4

    beta2_ps = float(params.get("beta2", DEFAULT_PARAMS["beta2"]))
    beta3_ps = float(params.get("beta3", DEFAULT_PARAMS["beta3"]))
    beta4_ps = float(params.get("beta4", DEFAULT_PARAMS["beta4"]))
    beta2 = beta2_ps * (1e-12)**2 
    beta3 = beta3_ps * (1e-12)**3
    beta4 = beta4_ps * (1e-12)**4

    dt = time_window / N
    T = (np.arange(N) - N // 2) * dt
    f0 = c0 / lambda_0
    w0 = 2 * np.pi * f0
    
    omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)

    E = float(params.get("E", DEFAULT_PARAMS["E"]))

    if pulse_shape == "gaussian":
        T0 = FWHM / (2 * np.sqrt(np.log(2)))
        energy_factor = np.sqrt(np.pi) * T0
        def amplitude_from_P0(t, P0): return np.sqrt(P0) * np.exp(-(t**2) / (2 * T0**2))
    else:
        T0 = FWHM / (2 * np.arccosh(np.sqrt(2.0)))
        energy_factor = 2.0 * T0
        def amplitude_from_P0(t, P0): return np.sqrt(P0) / np.cosh(t / T0)

    P0 = E / energy_factor
    A0_tl = amplitude_from_P0(T, P0)
    
    # Apply Input Dispersion
    if abs(gdd) > 1e-40 or abs(tod) > 1e-40 or abs(fod) > 1e-40:
        A0_w = np.fft.fft(A0_tl)
        # Phase expansion: phi(w) = 0.5*gdd*w^2 + 1/6*tod*w^3...
        # Note: omega from fftfreq is typically centered at 0 if we consider FFT domain, 
        # but fftfreq returns [0, 1, ..., -1] which corresponds to difference from center freq 0.
        input_phase = (0.5 * gdd * (omega**2) + 
                       (1.0/6.0) * tod * (omega**3) + 
                       (1.0/24.0) * fod * (omega**4))
        A0_w *= np.exp(-1j * input_phase)
        A0 = np.fft.ifft(A0_w)
    else:
        A0 = A0_tl
    
    # Calculate actual peak power and timing
    I0 = np.abs(A0)**2
    P_actual = np.max(I0)
    
    # Calculate actual FWHM -> T0_actual
    T_ps = T * 1e12
    fwhm_in_s = fwhm(T, I0)
    
    if pulse_shape == "gaussian":
        T0_actual = fwhm_in_s / (2 * np.sqrt(np.log(2)))
    else:
        T0_actual = fwhm_in_s / (2 * np.arccosh(np.sqrt(2.0)))

    alpha_db_per_km = float(params.get("alpha_db_per_km", DEFAULT_PARAMS["alpha_db_per_km"]))
    
    # Calculate Gamma from MFD and n2
    mfd_um = float(params.get("mfd", DEFAULT_PARAMS["mfd"]))
    n2 = float(params.get("n2", DEFAULT_PARAMS["n2"]))
    
    # A_eff = pi * (w_mode)^2 where w_mode = MFD/2
    mfd_m = mfd_um * 1e-6
    a_eff = np.pi * ((mfd_m / 2.0)**2)
    gamma = (2 * np.pi * n2) / (lambda_0 * a_eff)
    alpha_db_per_km = float(params.get("alpha_db_per_km", DEFAULT_PARAMS["alpha_db_per_km"]))
    alpha = (alpha_db_per_km / 4.343) / 1000.0

    use_raman = bool(params.get("use_raman", DEFAULT_PARAMS.get("use_raman", False)))
    f_R = float(params.get("f_R", DEFAULT_PARAMS["f_R"]))
    f_R_effective = f_R if use_raman else 0.0
    use_self_steepening = bool(params.get("use_self_steepening", DEFAULT_PARAMS["use_self_steepening"]))

    L = float(params.get("L", DEFAULT_PARAMS["L"]))
    n_steps = int(params.get("n_steps", DEFAULT_PARAMS["n_steps"]))
    if n_steps <= 0: raise ValueError("Number of steps must be positive.")
    
    n_heatmap = int(params.get("n_heatmap_steps", DEFAULT_PARAMS["n_heatmap_steps"]))
    if n_heatmap <= 0: raise ValueError("Heatmap steps must be positive.")
    
    nz = n_steps
    dz = L / nz
    history_every = max(1, int(nz / n_heatmap))

    A_out, A_hist, z_hist = ssfm_nlse(
        A0, dt, dz, nz,
        beta2=beta2, beta3=beta3, beta4=beta4, gamma=gamma,
        w0=w0, alpha=alpha,
        f_R=f_R_effective,
        self_steepening=use_self_steepening,
        return_history=True, history_every=history_every,
    )

    # --- CHARACTERISTIC LENGTHS ---
    # --- CHARACTERISTIC LENGTHS ---
    # Using actual T0 and P0 from the launched pulse (which may be chirped/broadened)
    if abs(beta2) > 1e-40 and T0_actual > 0: L_D = (T0_actual**2) / abs(beta2)
    else: L_D = float('inf')
    
    if gamma > 1e-40 and P_actual > 1e-40: L_NL = 1.0 / (gamma * P_actual)
    else: L_NL = float('inf')
    
    if L_NL > 0 and L_NL != float('inf') and L_D != float('inf'): N_sol = np.sqrt(L_D / L_NL)
    else: N_sol = 0.0

    plots = {}
    # T_ps already calculated above (line 277 approx)
    # 1. Time Domain
    # I0 already calculated above
    Iout = np.abs(A_out)**2

    if params.get("auto_time_range", True): 
        # Smart auto-ranging: find where intensity > -40dB (1e-4) of peak
        I_comb = np.maximum(I0, Iout)
        peak_I = np.max(I_comb)
        if peak_I > 1e-40:
            threshold = 1e-4 * peak_I
            mask = I_comb >= threshold
            if np.any(mask):
                idx = np.where(mask)[0]
                t_start = T_ps[max(0, idx[0])]
                t_end = T_ps[min(len(T_ps)-1, idx[-1])]
                span = t_end - t_start
                # Add 50% padding on each side, but at least 0.5 ps
                pad = max(0.5 * span, 0.5)
                
                t_lim_min = max(T_ps[0], t_start - pad)
                t_lim_max = min(T_ps[-1], t_end + pad)
                t_lim = (t_lim_min, t_lim_max)
            else:
                t_lim = (T_ps[0], T_ps[-1])
        else:
            t_lim = (T_ps[0], T_ps[-1])
    else: 
        t_lim = (params.get("time_min"), params.get("time_max"))
    # fwhm_in_s already calculated above
    fwhm_out_s = fwhm(T, Iout)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(T_ps, I0, label=f"Input (FWHM={fmt_time_si(fwhm_in_s)})")
    ax.plot(T_ps, Iout, "--", label=f"Output (FWHM={fmt_time_si(fwhm_out_s)})")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Time Domain")
    ax.set_xlim(t_lim)
    ax.legend()
    plots["time_domain"] = process_plot(fig, ax, labels={"x": "Time (ps)", "y": "Intensity (W)"})

    # 2. Spectrum
    A0_w = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A0)))
    Aout_w = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_out)))
    f_base = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    f_abs = f0 + f_base
    lambda_abs = c0 / f_abs
    lambda_nm_arr = lambda_abs * 1e9
    Sf_in = np.abs(A0_w)**2
    Sf_out = np.abs(Aout_w)**2
    Jac = c0 / (lambda_abs**2)
    Sl_in = Sf_in * Jac
    Sl_out = Sf_out * Jac
    sort_idx = np.argsort(lambda_nm_arr)
    lambda_nm_sorted = lambda_nm_arr[sort_idx]
    Sl_in_sorted = Sl_in[sort_idx]
    Sl_out_sorted = Sl_out[sort_idx]
    fwhm_in_nm = fwhm(lambda_nm_sorted, Sl_in_sorted)
    fwhm_out_nm = fwhm(lambda_nm_sorted, Sl_out_sorted)

    if params.get("auto_lambda_range", True):
        s_out = Sl_out_sorted / np.maximum(Sl_out_sorted.max(), 1e-300)
        mask = 10 * np.log10(np.maximum(s_out, 1e-300)) >= -40
        if not np.any(mask): mask = s_out >= 1e-6
        pad_pts = max(5, N // 200)
        idx = np.where(mask)[0]
        i0 = max(0, idx[0] - pad_pts)
        i1 = min(len(lambda_nm_sorted) - 1, idx[-1] + pad_pts)
        x_min = lambda_nm_sorted[i0]
        x_max = lambda_nm_sorted[i1]
    else:
        x_min = float(params.get("lambda_min"))
        x_max = float(params.get("lambda_max"))

    def nm_to_THz(l_nm): return (c0 / (l_nm * 1e-9)) * 1e-12
    def THz_to_nm(f_THz): return (c0 / (f_THz * 1e12)) * 1e9

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(lambda_nm_sorted, Sl_in_sorted, label=f"Input (FWHM={fmt_nm(fwhm_in_nm)})")
    ax1.plot(lambda_nm_sorted, Sl_out_sorted, "--", label=f"Output (FWHM={fmt_nm(fwhm_out_nm)})")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Spectral Density (a.u.)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_title("Spectral Domain")
    ax1.legend()
    secax = ax1.secondary_xaxis('top', functions=(nm_to_THz, THz_to_nm))
    secax.set_xlabel("Frequency (THz)")
    plots["spectral_domain"] = process_plot(fig, ax1, labels={"x": "Wavelength (nm)", "y": "PSD (a.u.)"}, dual_axis_type="spectrum")

    # 3. Heatmaps
    evolution_matrix = np.abs(np.array(A_hist))**2
    global_peak = np.max(evolution_matrix)
    norm_matrix = evolution_matrix / global_peak + 1e-20
    db_matrix = 10 * np.log10(norm_matrix)
    db_matrix[db_matrix < -40] = -40
    extent = [T_ps[0], T_ps[-1], z_hist[0], z_hist[-1]]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(db_matrix, aspect="auto", origin="lower", extent=extent, cmap="jet", vmin=-40, vmax=0)
    plt.colorbar(im, ax=ax, label="Intensity (dB)")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Distance [m]")
    ax.set_xlim(t_lim)
    ax.set_title("Temporal Evolution")
    plots["temporal_evolution"] = process_plot(fig, ax, labels={"x": "Time (ps)", "y": "Distance (m)", "z": "Intensity (dB)"}, z_data=db_matrix)

    A_matrix = np.array(A_hist)
    A_w_matrix = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A_matrix, axes=1), axis=1), axes=1)
    Sf_matrix = np.abs(A_w_matrix)**2
    wl_grid_linear = np.linspace(x_min, x_max, 600)
    Jac_sorted = c0 / ((lambda_nm_sorted * 1e-9)**2)
    Sl_rows = []
    for i in range(len(z_hist)):
        Sf_row = Sf_matrix[i, :]
        Sf_row_sorted = Sf_row[sort_idx]
        Sl_row = Sf_row_sorted * Jac_sorted
        Sl_interp = np.interp(wl_grid_linear, lambda_nm_sorted, Sl_row, left=0, right=0)
        Sl_rows.append(Sl_interp)
    Sl_rows = np.array(Sl_rows)
    Sl_max_val = np.max(Sl_rows)
    Sl_db_map = 10 * np.log10(Sl_rows / Sl_max_val + 1e-20)
    Sl_db_map[Sl_db_map < -40] = -40
    fig, ax_spec = plt.subplots(figsize=(8, 6))
    extent_wl = [wl_grid_linear[0], wl_grid_linear[-1], z_hist[0], z_hist[-1]]
    im = ax_spec.imshow(Sl_db_map, aspect="auto", origin="lower", extent=extent_wl, cmap="jet", vmin=-40, vmax=0)
    plt.colorbar(im, ax=ax_spec, label="PSD (dB)")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Distance [m]")
    ax_spec.set_title("Spectral Evolution")
    secax = ax_spec.secondary_xaxis('top', functions=(nm_to_THz, THz_to_nm))
    secax.set_xlabel("Frequency (THz)")
    plots["spectral_evolution"] = process_plot(fig, ax_spec, labels={"x": "Wavelength (nm)", "y": "Distance (m)", "z": "PSD (dB)"}, z_data=Sl_db_map, dual_axis_type="spectrum")

    N_exp = int(round(np.log2(N)))
    info_lines = [
        f"Input: {lambda_nm:.1f} nm, {fwhm_fs:.1f} fs (TL), E={E*1e12:.2f} pJ",
        f"Peak Power: {P_actual:.2f} W (TL: {P0:.2f} W)",
        f"Lengths: L_D={L_D:.3g} m, L_NL={L_NL:.3g} m, Soliton N={N_sol:.3f}",
        f"N: {N} (2^{N_exp}), Window: {time_window_ps:.1f} ps",
        f"Fiber L: {L:.3g} m, Steps: {n_steps}, dz: {dz:.3g} m",
        f"Dispersion: β2={beta2_ps} ps²/m",
        f"Raman: {use_raman}, Steepening: {use_self_steepening}, Nonlinear: {params.get('use_nonlinear')}",
    ]
    info_text = "\n".join(info_lines)

    return plots, info_text