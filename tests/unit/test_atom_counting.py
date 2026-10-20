"""Atom counting used for the abundance ratios (C/O, [X/H], ...).

Regression test for a bug where every species counted as exactly one atom of any element it
contains (so H2O counted 1 H instead of 2, H2 too, and He was counted as hydrogen), which
biased every [X/H] by about -0.23 dex on a typical hot-Jupiter composition.
"""
import numpy as np
import pytest

import starships.retrieval_utils as ru


@pytest.mark.parametrize('species, expected', [
    ('H2O', {'H': 2, 'O': 1}),
    ('CO', {'C': 1, 'O': 1}),
    ('CO2', {'C': 1, 'O': 2}),
    ('TiO', {'Ti': 1, 'O': 1}),
    ('FeH', {'Fe': 1, 'H': 1}),
    ('H2', {'H': 2}),
    ('H', {'H': 1}),
    ('He', {'He': 1}),                      # helium is not hydrogen
    ('Si', {'Si': 1}),
    ('H-', {'H': 1}),                       # charge sign ignored
    ('e-', {}),                             # electron: no atom
    ('CO_all_iso_HITEMP', {'C': 1, 'O': 1}),  # line list name ignored
    ('H2O_HITEMP', {'H': 2, 'O': 1}),
    ('C2H2', {'C': 2, 'H': 2}),
    ('CH4', {'C': 1, 'H': 4}),
])
def test_count_atoms(species, expected):
    assert ru.count_atoms(species) == expected


class TestGetAtomAbundance:

    species = ['H2O', 'CO', 'OH', 'TiO', 'H-', 'e-', 'H2', 'H', 'He']
    vmr = dict(H2O=1e-3, CO=2e-3, OH=1e-5, TiO=1e-7, **{'H-': 1e-8, 'e-': 1e-6}, H2=0.84, H=1e-4, He=0.15)

    @property
    def abundances(self):
        return [np.array([self.vmr[s]]) for s in self.species]

    def test_hydrogen_counts_two_per_h2_and_h2o_and_excludes_helium(self):
        h = ru.get_atom_abundance('H', self.abundances, self.species)[0]
        expected = 2e-3 + 1e-5 + 1e-8 + 2 * 0.84 + 1e-4       # H2O, OH, H-, H2, H (no He)
        assert h == pytest.approx(expected)

    def test_carbon_and_oxygen(self):
        c = ru.get_atom_abundance('C', self.abundances, self.species)[0]
        o = ru.get_atom_abundance('O', self.abundances, self.species)[0]
        assert c == pytest.approx(2e-3)
        assert o == pytest.approx(1e-3 + 2e-3 + 1e-5 + 1e-7)  # H2O, CO, OH, TiO

    def test_abund0_is_added_and_shapes_are_kept(self):
        ab = [np.full((4, 3), 1e-3), np.full((4, 3), 2e-3)]
        out = ru.get_atom_abundance('O', ab, ['H2O', 'CO'], abund0=1e-4)
        assert out.shape == (4, 3)
        np.testing.assert_allclose(out, 1e-3 + 2e-3 + 1e-4)
