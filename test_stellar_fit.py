"""
Quick integration test for starships.stellar_fit.
Uses only locally available PHOENIX files (teff 7400-7500, logg 4.5, metal 0.0)
and the local SPIRou reduction file. No downloads, no petitRADTRANS.
"""
import sys
import numpy as np

# ── minimal config that matches available local PHOENIX files ──────────────
LOCAL_DATA = (
    "/Users/antoinedb/VSCodeProjects/starships_cleanup/data/Reduced/"
    "sequence_5-pc_mask_wings90_day1_data_trs_.npz"
)

CFG = {
    "star_name": "test_star",
    "data_file": LOCAL_DATA,
    "bad_pix_frac": 0.5,
    "min_valid_pixels": 100,
    # PHOENIX grid (only teff 7400–7500 & logg 4.5 & metal 0.0 exist locally)
    "phoenix_resolution": 70000,
    "phoenix_oversampling": 2,          # smaller → faster
    "phoenix_n_fwhm": 5,
    "phoenix_method": "linear",
    "n_poly": 4,
    "rot_broad_samp": 70000,
    "base_dir": None,
    "walker_path": None,
    "run_name": "test_run",
    "walker_file_out": "test_results",
    "n_live_points": 50,
    "n_restarts": 2,
    "special_init": {"teff": 7450.0, "vsini": 50.0, "v_shift": 0.0},
    "custom_prior_file": None,
    "fixed_params": {
        "alpha": 0.0,
        "logg":  4.5,    # fixed: only logg=4.5 available locally
        "metal": 0.0,    # fixed: only Z-0.0 available locally
    },
    "params_prior": {
        "teff":    ["uniform", 7400, 7500],
        "vsini":   ["uniform", 10.0, 120.0],
        "epsilon": ["uniform", 0.01, 0.99],
        "v_shift": ["uniform", -50, 50],
    },
}

# ── run tests ─────────────────────────────────────────────────────────────
import starships.stellar_fit as sf

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
errors = []

def check(name, fn):
    try:
        result = fn()
        print(f"  {PASS}  {name}")
        return result
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"       {type(e).__name__}: {e}")
        errors.append(name)
        return None


print("\n── Test 1: setup_stellar_fit ─────────────────────────────────────")
check("setup_stellar_fit()", lambda: sf.setup_stellar_fit(CFG))

if errors:
    print(f"\nSetup failed — stopping here.\n")
    sys.exit(1)

print(f"  n_orders loaded : {sf.ref_wave.shape[0]}")
print(f"  wavelength range: {sf.ref_wave.min():.3f} – {sf.ref_wave.max():.3f} µm")
print(f"  free params     : {list(sf.params_prior.keys())}")

print("\n── Test 2: unpack_theta / pack_theta ────────────────────────────")
theta0 = np.array([7450.0, 50.0, 0.6, 0.0])
td = check("unpack_theta", lambda: sf.unpack_theta(theta0))
if td:
    check("pack_theta round-trip",
          lambda: np.testing.assert_allclose(sf.pack_theta(td), theta0))

print("\n── Test 3: stellar model generation (order 0) ───────────────────")
model = check(
    "_generate_stellar_model_ord(idx_ord=0)",
    lambda: sf._generate_stellar_model_ord(0, **td),
)
if model is not None:
    valid = np.isfinite(model)
    print(f"  valid pixels: {valid.sum()} / {len(model)}")
    print(f"  median flux : {np.nanmedian(model):.4f}  (should be ~1.0)")

print("\n── Test 4: profile log-likelihood (one order) ───────────────────")
logl_ord = check(
    "_profile_logl_one_order(idx_ord=0)",
    lambda: sf._profile_logl_one_order(0, model),
)
if logl_ord is not None:
    print(f"  logl order 0: {logl_ord:.2f}")

print("\n── Test 5: total profile log-likelihood ─────────────────────────")
logl_total = check(
    "profile_log_likelihood(theta_dict)",
    lambda: sf.profile_log_likelihood(td),
)
if logl_total is not None:
    print(f"  total logl  : {logl_total:.2f}")

print("\n── Test 6: lnprob (prior + logl) ───────────────────────────────")
lp = check("lnprob(theta0)", lambda: sf.lnprob(theta0))
if lp is not None:
    print(f"  lnprob      : {lp:.2f}")

print("\n── Test 7: make_model_with_best_poly ────────────────────────────")
out = check(
    "make_model_with_best_poly(theta_dict)",
    lambda: sf.make_model_with_best_poly(td, return_poly=True),
)
if out is not None:
    models, polys = out
    print(f"  output shape: {models.shape}")
    finite_ords = np.sum(np.any(np.isfinite(models), axis=1))
    print(f"  orders with finite model: {finite_ords} / {models.shape[0]}")

print("\n── Test 8: prior_transform (for dynesty) ────────────────────────")
u = np.array([0.5, 0.5, 0.5, 0.5])
theta_t = check("prior_transform(u)", lambda: sf.prior_transform(u))
if theta_t is not None:
    print(f"  transformed : {dict(zip(sf.params_prior.keys(), theta_t))}")

# ── summary ───────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
if not errors:
    print(f"{PASS}  All tests passed.")
else:
    print(f"{FAIL}  {len(errors)} test(s) failed: {errors}")
    sys.exit(1)
