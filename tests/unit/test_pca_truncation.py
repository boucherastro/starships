import numpy as np
import pytest

from starships.transpec import PCA_remove, apply_pca_truncation


def _make_synthetic_spec_trans(n_spec=20, nord=2, npix=60, seed=0):
    """A synthetic transmission spectrum shaped like a real `spec_trans`: values near 1,
    a few common systematic modes shared across exposures (what the PCA is meant to
    remove), and a couple of always-masked pixels (like real order edges)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_spec)
    x = np.linspace(0, 1, npix)

    flux = np.ones((n_spec, nord, npix))
    for iord in range(nord):
        mode1 = np.outer(np.sin(2 * np.pi * t + iord), np.cos(3 * np.pi * x))
        mode2 = np.outer(t**2, np.sin(5 * np.pi * x))
        flux[:, iord, :] += 0.01 * mode1 + 0.005 * mode2
        flux[:, iord, :] += 0.001 * rng.standard_normal((n_spec, npix))

    mask = np.zeros_like(flux, dtype=bool)
    mask[:, :, :2] = True  # order-edge pixels, masked for every exposure
    return np.ma.array(flux, mask=mask)


class TestPCARemoveGuard:
    """B3: `PCA_remove` used to silently truncate `pcs[:n_pcs, :]` past the fitted rank,
    producing a silently-wrong result instead of an error -- flagged during the B3 code
    mapping as a real footgun once `n_pc` becomes a read-time argument decoupled from
    however many components were actually fit and saved at reduction time."""

    def test_raises_when_n_pcs_exceeds_fitted_components(self):
        pcs = np.zeros((5, 10))
        coefficients = np.zeros((7, 5))
        with pytest.raises(ValueError):
            PCA_remove(np.zeros((7, 10)), pcs, coefficients, n_pcs=6)

    def test_does_not_raise_when_n_pcs_within_range(self):
        pcs = np.zeros((5, 10))
        coefficients = np.zeros((7, 5))
        PCA_remove(np.zeros((7, 10)), pcs, coefficients, n_pcs=5)


class TestApplyPCATruncationEquivalence:
    """Core B3 guarantee: fitting the PCA once (at reduction time) and truncating to a
    given `n_pca` at read time must be numerically identical to the old behaviour of
    fitting a fresh PCA for that exact `n_pca` every time (what the pipeline used to do
    once per n_pc in the sweep, per B0/B3 code mapping)."""

    def test_reusing_a_fitted_pca_matches_a_fresh_fit_at_each_n_pc(self):
        spec_trans = _make_synthetic_spec_trans()
        n_comps = 10

        # Fit once (as reduction now does), at an arbitrary n_pca -- the fit itself does
        # not depend on n_pca (only the truncation below does).
        _, _, _, _, _, pca = apply_pca_truncation(spec_trans, n_pca=3, n_comps=n_comps)

        for n_pc in (1, 2, 5, 8):
            clean_reused, _, final_reused, rebuilt_reused, _, _ = apply_pca_truncation(
                spec_trans, n_pca=n_pc, pca=pca)
            clean_fresh, _, final_fresh, rebuilt_fresh, _, _ = apply_pca_truncation(
                spec_trans, n_pca=n_pc, n_comps=n_comps)

            np.testing.assert_array_equal(clean_reused.mask, clean_fresh.mask)
            np.testing.assert_allclose(clean_reused.compressed(), clean_fresh.compressed(),
                                        rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(final_reused.compressed(), final_fresh.compressed(),
                                        rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(rebuilt_reused.compressed(), rebuilt_fresh.compressed(),
                                        rtol=1e-10, atol=1e-12)

    def test_fit_is_deterministic_across_independent_calls(self):
        # Guards the `svd_solver='full'` pin in `PCA_decompose`: with sklearn's shape-based
        # 'auto' heuristic this happens to already select 'full' (deterministic LAPACK SVD)
        # for our matrices, but B3 now relies on run-to-run reproducibility explicitly.
        spec_trans = _make_synthetic_spec_trans(seed=1)
        _, _, _, _, _, pca_a = apply_pca_truncation(spec_trans, n_pca=2, n_comps=8)
        _, _, _, _, _, pca_b = apply_pca_truncation(spec_trans, n_pca=2, n_comps=8)
        np.testing.assert_array_equal(pca_a.components_, pca_b.components_)

    def test_n_pca_larger_than_fitted_components_raises(self):
        spec_trans = _make_synthetic_spec_trans()
        _, _, _, _, _, pca = apply_pca_truncation(spec_trans, n_pca=2, n_comps=5)
        with pytest.raises(ValueError):
            apply_pca_truncation(spec_trans, n_pca=6, pca=pca)
