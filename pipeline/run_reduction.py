"""Reduction-only pipeline entry point (Chantier B, B5).

Unlike `pipeline/run_pipe.py` (which also drives CCF, model generation and injection
recovery), this entry point does exactly one thing: turn raw observations into reduced,
saved sequences (`pipeline.reduction.reduce_data`), one file per visit. CCF/model/
injection stay `run_pipe.py`'s job -- they are not needed to produce a reduction, and
keeping this entry point free of them makes it usable on its own (e.g. to prepare data
for a retrieval, which never needs `run_pipe.py`'s other stages).

Assembles pieces already implemented and tested in Chantier B (B1-B3) -- no new
reduction logic lives here, only orchestration:

- `pipeline.reduction.load_planet` / `reduce_data` (per-visit reduction, B1-B3).
- `pipeline.split_nights.split_night` (visit discovery, B2), run automatically whenever
  `config_dict['visit_name']` is left empty -- this is the default behaviour of this
  entry point, not an opt-in flag (mirrors what `run_pipe.py::run_pipe` already does).

Usage
-----
python -m pipeline.run_reduction config.yaml reduction_name
"""
import argparse
import traceback
from pathlib import Path

import yaml

import pipeline.reduction as red
import pipeline.split_nights as split


def resolve_visits(config_dict):
    """Return the list of visit names to reduce, splitting the raw observations into
    visits first (`split_nights.split_night`) if `config_dict['visit_name']` is empty.
    """
    if config_dict['visit_name'] == []:
        config_dict['visit_name'] = split.split_night(
            config_dict['obs_dir'], str(config_dict['obs_dir']),
            instrument=config_dict['instrument'],
        )
    return config_dict['visit_name']


def reduce_visit(config_dict, planet, obs, dirs_dict, visit_name):
    """Reduce one visit for every `(mask_tellu, mask_wings)` combination in the config,
    at the first requested `n_pc` (the reduction itself does not depend on `n_pc` since
    B3 -- see `pipeline.reduction.reduce_data`). Kept sequential on purpose: several
    `(mask_tellu, mask_wings)` combinations can share the same underlying reduction file,
    and `pipeline.run_pipe.pool_processing` already had to fix a real write race from
    parallelizing this (see Chantier B, B3) -- there is no CCF sweep here to justify the
    added complexity of a pool for this entry point.
    """
    n_pc0 = config_dict['n_pc'][0]
    for mask_tellu in config_dict['mask_tellu']:
        for mask_wings in config_dict['mask_wings']:
            red.reduce_data(config_dict, planet, obs, dirs_dict['scratch_dir'],
                             dirs_dict['red_steps_dir'], n_pc0, mask_tellu, mask_wings,
                             visit_name)


def run_reduction(config_filepath, reduction_name):
    """Reduce every visit named (or discovered) in `config_filepath`, for every
    `(mask_tellu, mask_wings)` combination it lists. This is the whole job -- no CCF, no
    model generation, no injection recovery (see `pipeline.run_pipe` for those).
    """
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['reduction'] = reduction_name
    config_dict['obs_dir'] = Path.home() / Path(config_dict['obs_dir'])

    visit_names = resolve_visits(config_dict)
    print('VISIT NAMES: ', visit_names)

    for visit_name in visit_names:
        print('Reducing visit:', visit_name)
        planet, obs = red.load_planet(config_dict, visit_name)
        dirs_dict = red.set_save_location(config_dict['pl_name'], visit_name,
                                           config_dict['reduction'], config_dict['instrument'])
        try:
            reduce_visit(config_dict, planet, obs, dirs_dict, visit_name)
        except Exception:
            print(f'Error reducing visit {visit_name}. Skipping...')
            traceback.print_exc()
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce observations for the given config file.')
    parser.add_argument('config_filepath', type=Path, help='Path to the config.yaml file.')
    parser.add_argument('reduction_name', type=str, help='Name of the reduction.')

    args = parser.parse_args()

    run_reduction(args.config_filepath, args.reduction_name)
