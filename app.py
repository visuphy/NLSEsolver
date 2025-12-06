import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from solver import run_simulation, DEFAULT_PARAMS, PROD_MAX_N, PROD_MAX_STEPS, PROD_MAX_HEATMAP
import math

app = Flask(__name__)

# Apply ProxyFix for correct URL generation behind Nginx
# x_prefix=1 tells Flask to trust the X-Script-Name header from Nginx
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)

def get_form_params(form):
    """
    Read parameters from the HTML form; fall back to DEFAULT_PARAMS
    when fields are empty.
    """
    params = {}

    def get_val(name):
        return form.get(name, "").strip()

    def get_int(name, default):
        v = get_val(name)
        return default if v == "" else int(v)

    def get_float(name, default):
        v = get_val(name)
        return default if v == "" else float(v)

    def get_str(name, default):
        v = get_val(name)
        return default if v == "" else v

    # --- Grid: exponent for N ---
    default_N_exp = int(round(math.log2(DEFAULT_PARAMS["N"])))
    N_exp = get_int("N_exp", default_N_exp)
    if N_exp <= 0: raise ValueError("Exponent for N must be positive.")
    params["N_exp"] = N_exp
    params["N"] = 2 ** N_exp

    params["time_window"] = get_float("time_window", DEFAULT_PARAMS["time_window"])
    params["lambda_0"] = get_float("lambda_0", DEFAULT_PARAMS["lambda_0"])
    params["pulse_shape"] = get_str("pulse_shape", DEFAULT_PARAMS["pulse_shape"])
    params["fwhm_nm"] = get_float("fwhm_nm", DEFAULT_PARAMS["fwhm_nm"])
    params["gdd"] = get_float("gdd", DEFAULT_PARAMS["gdd"])
    params["tod"] = get_float("tod", DEFAULT_PARAMS["tod"])
    params["fod"] = get_float("fod", DEFAULT_PARAMS["fod"])
    params["E"] = get_float("E", DEFAULT_PARAMS["E"])
    params["beta2"] = get_float("beta2", DEFAULT_PARAMS["beta2"])
    params["beta3"] = get_float("beta3", DEFAULT_PARAMS["beta3"])
    params["beta4"] = get_float("beta4", DEFAULT_PARAMS["beta4"])
    params["alpha_db_per_km"] = get_float("alpha_db_per_km", DEFAULT_PARAMS["alpha_db_per_km"])

    params["mfd"] = get_float("mfd", DEFAULT_PARAMS["mfd"])
    params["n2"] = get_float("n2", DEFAULT_PARAMS["n2"])

    params["use_nonlinear"] = bool(form.get("use_nonlinear"))
    params["use_raman"] = bool(form.get("use_raman"))
    params["f_R"] = get_float("f_R", DEFAULT_PARAMS["f_R"])
    params["use_self_steepening"] = bool(form.get("use_self_steepening"))

    if not params["use_nonlinear"]:
        # Setting n2 to 0 ensures gamma (calculated in solver) will be 0
        params["n2"] = 0.0
        params["use_raman"] = False
        params["use_self_steepening"] = False

    params["L"] = get_float("L", DEFAULT_PARAMS["L"])
    params["n_steps"] = get_int("n_steps", DEFAULT_PARAMS["n_steps"])
    params["n_heatmap_steps"] = get_int("n_heatmap_steps", DEFAULT_PARAMS["n_heatmap_steps"])

    params["auto_time_range"] = bool(form.get("auto_time_range"))
    params["time_min"] = get_float("time_min", DEFAULT_PARAMS["time_min"])
    params["time_max"] = get_float("time_max", DEFAULT_PARAMS["time_max"])

    params["auto_lambda_range"] = bool(form.get("auto_lambda_range"))
    params["lambda_min"] = get_float("lambda_min", DEFAULT_PARAMS["lambda_min"])
    params["lambda_max"] = get_float("lambda_max", DEFAULT_PARAMS["lambda_max"])

    return params


# --- ROUTE 1: The Intro Page (Root URL) ---
@app.route("/", methods=["GET"])
def intro():
    """
    Serves the introduction page (intro.html) at /NLSEsolver/
    """
    return render_template("intro.html")


# --- ROUTE 2: The Tool/Calculator (Mapped to /tool) ---
@app.route("/tool", methods=["GET"])
def tool():
    """
    Serves the actual calculator interface (index.html) at /NLSEsolver/tool
    """
    # Detect Environment
    is_development = os.environ.get('APP_ENV') == 'development'

    form_values = dict(DEFAULT_PARAMS)
    form_values["N_exp"] = int(round(math.log2(form_values["N"])))
    
    # Calculate max allowed exponent based on environment
    if is_development:
        max_n_exp = 24  # 2^24 ~ 16 million points (Local machine limit)
    else:
        max_n_exp = int(math.log2(PROD_MAX_N)) # 2^16

    return render_template(
        "index.html",
        params=form_values,
        max_n_exp=max_n_exp,
        is_development=is_development
    )


# --- ROUTE 3: The Calculation Endpoint ---
@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        is_development = os.environ.get('APP_ENV') == 'development'
        params = get_form_params(request.form)
        
        # Enforce limits ONLY if NOT in development
        if not is_development:
            if params["N"] > PROD_MAX_N:
                return jsonify({"success": False, "error": f"Server Limit: N = {params['N']} exceeds max {PROD_MAX_N} (2^16). Download source for high-res."})
            if params["n_steps"] > PROD_MAX_STEPS:
                return jsonify({"success": False, "error": f"Server Limit: Steps ({params['n_steps']}) cannot exceed {PROD_MAX_STEPS}."})
            if params["n_heatmap_steps"] > PROD_MAX_HEATMAP:
                return jsonify({"success": False, "error": f"Server Limit: Heatmap steps ({params['n_heatmap_steps']}) cannot exceed {PROD_MAX_HEATMAP}."})

        plots, info_text = run_simulation(params)
        
        return jsonify({
            "success": True,
            "plots": plots,
            "info_text": info_text
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5005)