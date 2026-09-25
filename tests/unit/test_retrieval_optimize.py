"""``starships.retrieval_optimize`` -- optimizer plumbing tests on a toy posterior.

No petitRADTRANS and no retrieval YAML: ``lnprob`` is replaced by a small
correlated Gaussian, so this checks the burn-in reading, the whitening, the
start selection, the CMA-ES loop (feasibility, resume, ...) and the comparison
table, not the model.
"""
from pathlib import Path

import emcee
import numpy as np
import pytest

import starships.retrieval_optimize as ro

pytest.importorskip('cma')

N_DIM = 6
LNPROB_MAX = 15000.  # like the real thing: large offset, only the differences matter


@pytest.fixture
def toy():
    """A correlated Gaussian posterior with a known maximum, inside a box prior."""
    rng = np.random.default_rng(1)
    mu = np.array([1., -2., 0.5, 3., -1., 0.])
    # Very different scales and a strong correlation, like real retrieval parameters
    scales = np.array([0.1, 1., 10., 0.5, 2., 0.05])
    a = rng.normal(size=(N_DIM, N_DIM))
    corr = a @ a.T / N_DIM + np.eye(N_DIM)
    cov = corr * np.outer(scales, scales)
    prec = np.linalg.inv(cov)
    low, high = mu - 8 * scales, mu + 8 * scales
    calls = []

    def lnprob(x):
        calls.append(np.array(x))
        return LNPROB_MAX - 0.5 * (x - mu) @ prec @ (x - mu)

    def neg_lnprob_batch(x_batch):
        return np.array([-lnprob(x) for x in x_batch])

    def is_feasible(x):
        return bool(np.all(x >= low) and np.all(x <= high))

    return dict(mu=mu, cov=cov, lnprob=lnprob, neg_lnprob_batch=neg_lnprob_batch,
                is_feasible=is_feasible, low=low, high=high, calls=calls, rng=rng)


# ---------------------------------------------------------------------------
# Burn-in reading
# ---------------------------------------------------------------------------

def test_load_burnin_samples_ranks_best_first_and_drops_non_finite(tmp_path):
    n_walkers, n_dim, n_steps = 10, 3, 15
    backend_file = tmp_path / 'burnin.h5'
    backend = emcee.backends.HDFBackend(str(backend_file))
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lambda x: -0.5 * np.sum(x ** 2), backend=backend)
    sampler.run_mcmc(np.random.default_rng(0).normal(size=(n_walkers, n_dim)), n_steps)

    samples, lnprob = ro.load_burnin_samples(backend_file)

    assert samples.shape == (n_walkers * n_steps, n_dim)
    assert np.all(np.diff(lnprob) <= 0)  # best first
    # The lnprob returned really is the one of the returned sample
    np.testing.assert_allclose(lnprob, -0.5 * np.sum(samples ** 2, axis=1))


# ---------------------------------------------------------------------------
# Whitening and start selection
# ---------------------------------------------------------------------------

def test_whitening_round_trip_and_unit_covariance(toy):
    samples = toy['rng'].multivariate_normal(toy['mu'], toy['cov'], size=20000)
    L, L_inv = ro.whitening_from_samples(samples)

    np.testing.assert_allclose(L @ L_inv, np.eye(N_DIM), atol=1e-8)
    np.testing.assert_allclose(L @ L.T, np.cov(samples, rowvar=False), rtol=1e-8, atol=1e-12)
    # Whitened samples have identity covariance
    whitened = (samples - samples.mean(axis=0)) @ L_inv.T
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(N_DIM), atol=1e-8)


def test_whitening_survives_a_constant_parameter(toy):
    samples = toy['rng'].multivariate_normal(toy['mu'], toy['cov'], size=500)
    samples[:, 2] = 4.2  # zero variance: singular covariance
    L, L_inv = ro.whitening_from_samples(samples)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(L_inv))


def test_select_starts_skips_neighbours_and_is_deterministic():
    # 3 clusters of near-identical points (like successive steps of one walker),
    # ranked best first: A, A, A, B, B, C
    points = np.array([[0., 0.], [0.1, 0.], [0., 0.1], [10., 0.], [10.1, 0.], [0., 10.]])
    L_inv = np.eye(2)

    starts = ro.select_starts(points, L_inv, n_starts=3, min_separation=3.)

    np.testing.assert_array_equal(starts, [0, 3, 5])
    np.testing.assert_array_equal(starts, ro.select_starts(points, L_inv, n_starts=3, min_separation=3.))
    # Asking for more than exist returns what exists
    assert len(ro.select_starts(points, L_inv, n_starts=10, min_separation=3.)) == 3


def test_default_popsize_is_a_multiple_of_n_cpu():
    assert ro.default_popsize(27, n_cpu=1) == 4 + int(3 * np.log(27))
    for n_cpu in (1, 2, 5, 24):
        assert ro.default_popsize(27, n_cpu) % n_cpu == 0
        assert ro.default_popsize(27, n_cpu) >= 4 + int(3 * np.log(27))


# ---------------------------------------------------------------------------
# CMA-ES loop
# ---------------------------------------------------------------------------

def _start_near_optimum(toy):
    """A start about one std from the maximum, and the whitening of the posterior."""
    samples = toy['rng'].multivariate_normal(toy['mu'], toy['cov'], size=5000)
    L, _ = ro.whitening_from_samples(samples)
    x0 = toy['mu'] + L @ toy['rng'].normal(size=N_DIM)
    return x0, L


def test_optimize_cmaes_finds_the_maximum(toy):
    x0, L = _start_near_optimum(toy)

    result = ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'],
                               max_evals=5000, seed=3)

    # In lnprob units: the optimum is at LNPROB_MAX
    assert result['lnprob_best'] == pytest.approx(LNPROB_MAX, abs=0.05)
    assert result['lnprob_best'] <= LNPROB_MAX + 1e-9
    np.testing.assert_allclose(result['x_best'], toy['mu'], atol=0.2 * np.sqrt(np.diag(toy['cov'])).max())
    assert result['n_evals'] < 5000  # stopped by convergence, not by the budget
    assert result['history'].shape[1] == 4
    # The best lnprob so far never decreases
    assert np.all(np.diff(result['history'][:, 1]) >= 0)

    # Every evaluated candidate is kept, not just the best one, and matches what
    # neg_lnprob_batch actually saw (toy['calls'] records every call; snapshot it
    # before recomputing lnprob on it, since toy['lnprob'] itself appends to it).
    calls = np.array(toy['calls'])
    expected_lnprob = LNPROB_MAX - 0.5 * np.einsum('ij,jk,ik->i', calls - toy['mu'],
                                                    np.linalg.inv(toy['cov']), calls - toy['mu'])
    assert result['theta_trace'].shape == (result['n_evals'], N_DIM)
    assert result['lnprob_trace'].shape == (result['n_evals'],)
    np.testing.assert_allclose(result['theta_trace'], calls)
    np.testing.assert_allclose(result['lnprob_trace'], expected_lnprob)
    # The best of the trace is (at least as good as) the reported best
    assert result['lnprob_trace'].max() <= result['lnprob_best'] + 1e-9
    assert result['lnprob_trace'].max() == pytest.approx(result['lnprob_best'])


def test_optimize_cmaes_never_evaluates_outside_the_prior(toy):
    x0, L = _start_near_optimum(toy)
    # Narrow the box so that many candidates fall outside it
    scales = np.sqrt(np.diag(toy['cov']))
    low, high = toy['mu'] - 0.5 * scales, toy['mu'] + 0.5 * scales
    is_feasible = lambda x: bool(np.all(x >= low) and np.all(x <= high))
    x0 = np.clip(x0, low, high)

    ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, is_feasible, max_evals=400, seed=3)

    assert len(toy['calls']) > 0
    assert all(is_feasible(x) for x in toy['calls'])


def test_optimize_cmaes_ranks_failed_evaluations_last(toy):
    x0, L = _start_near_optimum(toy)

    def flaky(x_batch):
        f = toy['neg_lnprob_batch'](x_batch)
        f[::3] = np.inf  # a third of the evaluations "fail"
        return f

    result = ro.optimize_cmaes(flaky, x0, L, toy['is_feasible'], max_evals=1500, seed=3)

    assert np.isfinite(result['lnprob_best'])
    assert result['lnprob_best'] > LNPROB_MAX - 1.


def test_optimize_cmaes_resume_continues_where_it_stopped(toy, tmp_path):
    x0, L = _start_near_optimum(toy)
    state_file = tmp_path / 'state.pkl'

    first = ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'], popsize=8,
                              max_evals=40, seed=3, state_file=state_file)
    # The budget is checked between generations, so the last one can overshoot it
    assert 40 <= first['n_evals'] <= 40 + 8
    assert state_file.is_file()

    resumed = ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'],
                                max_evals=5000, state_file=state_file, resume=True)

    assert resumed['n_evals'] > first['n_evals']
    assert resumed['lnprob_best'] >= first['lnprob_best']
    assert resumed['lnprob_best'] == pytest.approx(LNPROB_MAX, abs=0.05)
    # The history of the first part is kept
    assert len(resumed['history']) > len(first['history'])
    np.testing.assert_allclose(resumed['history'][:len(first['history'])], first['history'])
    # The trace of the first part is kept too, and the resumed one is longer
    assert resumed['theta_trace'].shape[0] == resumed['n_evals'] > first['theta_trace'].shape[0] == first['n_evals']
    np.testing.assert_allclose(resumed['theta_trace'][:first['n_evals']], first['theta_trace'])
    np.testing.assert_allclose(resumed['lnprob_trace'][:first['n_evals']], first['lnprob_trace'])


def test_optimize_cmaes_resume_accepts_a_state_file_without_a_trace(toy, tmp_path):
    """A state file written before theta_trace/lnprob_trace existed (real case: a job
    already running on Narval when that feature was added) must still resume, just
    without being able to recover the trace of the evaluations already done."""
    x0, L = _start_near_optimum(toy)
    state_file = tmp_path / 'old_state.pkl'
    partial = ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'], popsize=8,
                                max_evals=40, seed=3, state_file=state_file)
    # Simulate a pre-trace pickle: drop the keys added later
    import pickle
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    del state['theta_trace'], state['lnprob_trace']
    with open(state_file, 'wb') as f:
        pickle.dump(state, f)

    resumed = ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'],
                                max_evals=5000, state_file=state_file, resume=True)

    assert resumed['n_evals'] > partial['n_evals']
    assert resumed['lnprob_best'] == pytest.approx(LNPROB_MAX, abs=0.05)
    # Only what was evaluated *after* resuming is in the trace -- the pre-trace part is gone
    assert resumed['theta_trace'].shape[0] == resumed['n_evals'] - partial['n_evals']

    # load_state must also tolerate the same missing keys
    state_read = ro.load_state(state_file)
    assert state_read['theta_trace'].shape[0] > 0
    assert state_read['n_evals'] == resumed['n_evals']


def test_optimize_cmaes_resume_without_state_file_raises(toy, tmp_path):
    x0, L = _start_near_optimum(toy)
    with pytest.raises(FileNotFoundError):
        ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'],
                          state_file=tmp_path / 'missing.pkl', resume=True)


# ---------------------------------------------------------------------------
# Inspecting a running/killed optimization from its state file alone
# ---------------------------------------------------------------------------

def test_load_state_matches_an_in_progress_run(toy, tmp_path):
    x0, L = _start_near_optimum(toy)
    state_file = tmp_path / 'state.pkl'
    partial = ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'], popsize=8,
                                max_evals=40, seed=3, state_file=state_file)

    state = ro.load_state(state_file)

    assert state['n_evals'] == partial['n_evals']
    assert state['lnprob_best'] == pytest.approx(partial['lnprob_best'])
    np.testing.assert_allclose(state['x_best'], partial['x_best'])
    np.testing.assert_allclose(state['theta_trace'], partial['theta_trace'])
    np.testing.assert_allclose(state['lnprob_trace'], partial['lnprob_trace'])
    np.testing.assert_allclose(state['history'], partial['history'])


def test_inspect_state_writes_a_readable_snapshot(toy, tmp_path, capsys):
    x0, L = _start_near_optimum(toy)
    state_file = tmp_path / 'optimum_burnin_start2_state.pkl'
    ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'], popsize=8,
                      max_evals=24, seed=3, state_file=state_file)

    out_file = ro.inspect_state(state_file)

    # Default name derived from the state file, sitting next to it
    assert out_file == tmp_path / 'optimum_burnin_start2_snapshot.npz'
    assert 'evals so far' in capsys.readouterr().out
    snap = np.load(out_file)
    state = ro.load_state(state_file)
    np.testing.assert_allclose(snap['theta_trace'], state['theta_trace'])
    np.testing.assert_allclose(snap['lnprob_trace'], state['lnprob_trace'])
    assert int(snap['n_evals']) == state['n_evals']


def test_inspect_state_custom_output_path(toy, tmp_path):
    x0, L = _start_near_optimum(toy)
    state_file = tmp_path / 'state.pkl'
    ro.optimize_cmaes(toy['neg_lnprob_batch'], x0, L, toy['is_feasible'], popsize=8,
                      max_evals=16, seed=3, state_file=state_file)

    out_file = ro.inspect_state(state_file, output_file=tmp_path / 'custom.npz')

    assert out_file == tmp_path / 'custom.npz'
    assert out_file.is_file()


# ---------------------------------------------------------------------------
# Output paths never collide with an existing run's files
# ---------------------------------------------------------------------------

def test_free_output_paths_no_collision_uses_the_plain_name(tmp_path):
    result_file, state_file = ro._free_output_paths(tmp_path, 'optimum_burnin_start1', resume=False)
    assert result_file == tmp_path / 'optimum_burnin_start1.npz'
    assert state_file == tmp_path / 'optimum_burnin_start1_state.pkl'


def test_free_output_paths_skips_an_existing_pair(tmp_path):
    (tmp_path / 'optimum_burnin_start1.npz').write_bytes(b'')
    (tmp_path / 'optimum_burnin_start1_state.pkl').write_bytes(b'')

    result_file, state_file = ro._free_output_paths(tmp_path, 'optimum_burnin_start1', resume=False)

    assert result_file == tmp_path / 'optimum_burnin_start1_0.npz'
    assert state_file == tmp_path / 'optimum_burnin_start1_0_state.pkl'
    assert not result_file.exists() and not state_file.exists()


def test_free_output_paths_state_file_alone_also_counts_as_taken(tmp_path):
    # An interrupted run: only the state file exists yet, no .npz -- still a real run,
    # must not be picked up by an unrelated fresh (non-resuming) one.
    (tmp_path / 'optimum_burnin_start1_state.pkl').write_bytes(b'')

    result_file, state_file = ro._free_output_paths(tmp_path, 'optimum_burnin_start1', resume=False)

    assert result_file == tmp_path / 'optimum_burnin_start1_0.npz'


def test_free_output_paths_keeps_incrementing_past_taken_suffixes(tmp_path):
    for name in ('optimum_burnin_start1', 'optimum_burnin_start1_0', 'optimum_burnin_start1_1'):
        (tmp_path / f'{name}.npz').write_bytes(b'')

    result_file, state_file = ro._free_output_paths(tmp_path, 'optimum_burnin_start1', resume=False)

    assert result_file == tmp_path / 'optimum_burnin_start1_2.npz'


def test_free_output_paths_resume_always_returns_the_plain_name_even_if_taken(tmp_path):
    (tmp_path / 'optimum_burnin_start1.npz').write_bytes(b'')
    (tmp_path / 'optimum_burnin_start1_state.pkl').write_bytes(b'')

    result_file, state_file = ro._free_output_paths(tmp_path, 'optimum_burnin_start1', resume=True)

    # Resuming must reuse the exact existing pair, not dodge it
    assert result_file == tmp_path / 'optimum_burnin_start1.npz'
    assert state_file == tmp_path / 'optimum_burnin_start1_state.pkl'


def test_free_output_paths_raises_after_100_taken_suffixes(tmp_path):
    (tmp_path / 'optimum_burnin_start1.npz').write_bytes(b'')
    for i in range(100):
        (tmp_path / f'optimum_burnin_start1_{i}.npz').write_bytes(b'')

    with pytest.raises(FileExistsError):
        ro._free_output_paths(tmp_path, 'optimum_burnin_start1', resume=False)


# ---------------------------------------------------------------------------
# Comparison of starts
# ---------------------------------------------------------------------------

def _fake_result(path: Path, start_index: int, lnprob_best: float) -> Path:
    np.savez(path, theta_best=np.arange(3.) + start_index, param_names=np.array(['a', 'b', 'c']),
             lnprob_best=lnprob_best, lnprob_start=lnprob_best - 30., start_index=start_index,
             n_evals=1000, stop_reason='tolfun=0.1')
    return path


def test_compare_results_flags_consistent_and_inconsistent_starts(tmp_path, capsys):
    files = [_fake_result(tmp_path / 'r1.npz', 1, 15200.1),
             _fake_result(tmp_path / 'r2.npz', 2, 15200.3)]
    out = ro.compare_results(files)
    assert out['consistent'] and out['spread'] == pytest.approx(0.2)
    assert 'consistent' in capsys.readouterr().out

    files.append(_fake_result(tmp_path / 'r3.npz', 3, 15180.))
    out = ro.compare_results(files)
    assert not out['consistent']
    assert 'NOT consistent' in capsys.readouterr().out
