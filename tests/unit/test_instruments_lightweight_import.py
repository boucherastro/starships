import subprocess
import sys


def test_instruments_module_does_not_pull_in_planet_obs():
    """`pipeline/make_model.py` (and several starships_analysis/KELT-20b/ scripts) import
    `starships.instruments` but never `starships.planet_obs` -- this must stay true, or those
    consumers would silently start pulling in all of `planet_obs.py`'s heavier dependencies
    (astropy units/exofile/the whole Observations class) for no benefit (Chantier B, B2
    follow-up). Run in a fresh subprocess (not just checking `sys.modules` in-process) since
    other tests in the same pytest run legitimately do import `planet_obs`, which would make
    an in-process check depend on test execution order.
    """
    result = subprocess.run(
        [sys.executable, '-c',
         "import starships.instruments; import sys; "
         "assert 'starships.planet_obs' not in sys.modules, 'planet_obs got imported'"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
