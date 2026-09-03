from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from scipy.signal import find_peaks

from datetime import timedelta

import starships.planet_obs as pl_obs

''' Note: paths must be changed on the following lines before using:
Line 62 : path to directory with observation files
Line 129 : path for night splitting image to be saved
'''

# defining functions

def find_sequence_gaps(time_stamps, time_threshold = 0.5):
    """Automatically find the index of new sequence.
    This is done by finding the delta t jumps

    time_threshold defines the difference between two times to be considered
    a different night. Ex: time_threshold = 1 means they need to differ by
    at least one julian day to be considered different nights. Units of days.

    Returns the last index of each night in the run."""

    idx_steps, _ = find_peaks(np.diff(time_stamps), height = time_threshold)

    return np.array(idx_steps)

def split_gaps(time_stamps):
    ''' Returns arrays of indices for each distinct observation night '''

    # check that input is in increasing chronological order
    if np.any(np.diff(time_stamps) < 0):
        raise ValueError('`time_stamps` must be sorted.')

    idx_steps = find_sequence_gaps(time_stamps)     # get last index of each night, except the final night
    idx_start = [None] + list(idx_steps + 1)        # shift to get starts of nights
    idx_end = list(idx_steps + 1) + [None]          # shift to get ends of nights
    indices = np.arange(len(time_stamps))           # create indices for each of the timestamps

    transit_tags = []
    for i_start, i_end in zip(idx_start, idx_end):  # iterate over each night in tuple (start, end)
        slice_sequence = slice(i_start, i_end)
        tr_tag = indices[slice_sequence]
        if tr_tag.any():
            transit_tags.append(indices[slice_sequence])

    return transit_tags


def _sibling_filenames(e2ds_filename, patterns):
    """Derive the tcorr/recon filenames for one e2ds file, from its shared prefix
    (Chantier B, B2 — replaces the old APERO-only `FILE_FRAMES`/`arcfile`-based logic,
    which relied on an `ARCFILE` header keyword absent from real SPIRou-APERO files).

    Omits 'recon' entirely when `patterns['recon_suffix']` is `None` — some datasets/DRS
    versions don't produce a telluric reconstruction spectrum as a separate file (confirmed
    real for at least one NIRPS-APERO dataset)."""
    prefix = e2ds_filename[:-len(patterns['e2ds_suffix'])]
    siblings = {'tcorr': prefix + patterns['tcorr_suffix']}
    if patterns.get('recon_suffix') is not None:
        siblings['recon'] = prefix + patterns['recon_suffix']
    return siblings


def split_night(obs_dir, path_fig, instrument='SPIRou-APERO'):
    """Group raw observation files into visits/nights, and write out the
    `list_e2ds`/`list_tcorr`/`list_recon` file lists `pipeline.reduction.load_planet` expects
    for each visit.

    Works for any instrument/DRS registered in `starships.planet_obs.instruments_drs` with a
    `list_file_patterns` entry (built in for SPIRou-APERO; see
    `starships.planet_obs.register_instrument` to add one for a custom instrument).
    """
    instrument_dict = pl_obs.instruments_drs[instrument]
    patterns = instrument_dict.get('list_file_patterns')
    if patterns is None:
        raise ValueError(
            f"split_night doesn't know the raw-file naming pattern for instrument "
            f"'{instrument}' (instruments_drs['{instrument}']['list_file_patterns'] is not "
            "set). See starships.planet_obs.register_instrument to add one."
        )
    mjd_key = instrument_dict['mjd']

    obs_jd = []
    filenames = []
    for entry in obs_dir.glob(patterns['e2ds_glob']):
        valid_file = True

        # Check if the tcorr/recon siblings needed exist
        siblings = _sibling_filenames(entry.name, patterns)
        for reduction_type, sibling_name in siblings.items():
            if not (obs_dir / sibling_name).is_file():
                print(f'{reduction_type} not found with filename: {sibling_name}')
                valid_file = False

        if valid_file:
            with fits.open(entry) as hdu:
                obs_jd.append(hdu[0].header[mjd_key])
            filenames.append(entry.name)

    # sort files in array in chronological order
    idx_sort = np.argsort(obs_jd)
    filenames = np.array(filenames)
    filenames = filenames[idx_sort]

    # add date units and index objects by sorted date
    obs_jd = obs_jd * u.d
    obs_jd = obs_jd[idx_sort]

    # splitting nights and counting total observation nights
    transit_tags = split_gaps(obs_jd)
    n_tags = len(transit_tags)

    """*****************************************************"""
    """              plotting different nights              """

    fig = plt.figure(figsize=(10,4), dpi = 200)
    ax_all = plt.subplot2grid((2, 1), (0, 0))
    ax_all.plot(obs_jd, np.ones_like(obs_jd), 'o')
    ax_all.set_xlabel('Time of observation [BJD]')
    ax_all.set_ylabel('All observations')

    ax_all = plt.subplot2grid((2, 4), (1, 0))
    ylabel = None
    for idx_tag, tr_tag in enumerate(transit_tags):
        ax_tag = plt.subplot2grid((2, n_tags), (1, idx_tag))
        # Set y label only for the leftmost panel
        if ylabel is None:
            ylabel = 'Zoom on each\nset of observations'
            ax_tag.set_ylabel(ylabel)
        ax_all.plot(obs_jd[tr_tag], np.ones_like(obs_jd[tr_tag]), '.')
        color = ax_all.get_lines()[-1].get_color()
        ax_tag.plot(obs_jd[tr_tag], np.ones_like(obs_jd[tr_tag]), '.', color=color)
        ax_tag.set_yticks([])

    fig.text(0, 0, 'PI and Observation Date:')
    y_pos = -0.1
    for idx, tr_tag in enumerate(transit_tags):
        with fits.open(obs_dir / filenames[tr_tag[0]]) as hdu:
            prog_id = hdu[0].header.get('HIERARCH ESO OBS PROG ID', 'unknown program')
            date_str = hdu[0].header['DATE']
        text = f"{filenames[tr_tag[0]]}\n{prog_id}, {date_str}"
        fig.text(0, y_pos, text, ha='left', va='bottom', fontsize=9)
        y_pos -= 0.1

    plt.tight_layout()
    # plt.show()
    if path_fig is not None:
        plt.savefig(path_fig + '/night_split.pdf', bbox_inches='tight')
        print('Saved figure')

    """****************************************************"""
    """ creating list of files for each observing sequence """
    visit_name = []

    for idx_tr, tr_tag in enumerate(transit_tags):
        filenames_tr = filenames[tr_tag]
        with fits.open(obs_dir / filenames_tr[0]) as hdu:
            date = Time(hdu[0].header['DATE']).datetime

        # modify date if hour is after midnight
        if date.hour >= 0 and date.hour < 12:
            date = date - timedelta(days=1)

        # Matches the `visit_YYYY-MM-DD` convention already used by real reduced datasets
        # (e.g. `list_e2ds_visit_2019-10-07`) — no hour suffix, since the night-splitting
        # itself (above) is what actually disambiguates separate visits, not the filename.
        date_str = 'visit_' + date.strftime('%Y-%m-%d')

        file_lists = {'e2ds': list(filenames_tr)}
        file_lists['tcorr'] = []
        file_lists['recon'] = []
        for tr_file in filenames_tr:
            siblings = _sibling_filenames(tr_file, patterns)
            file_lists['tcorr'].append(siblings['tcorr'])
            file_lists['recon'].append(siblings['recon'])

        for reduc_type, reduc_flist in file_lists.items():
            name_list_files = Path(f'list_{reduc_type}_{date_str}')
            print(f'Writing to {name_list_files}')
            with open(obs_dir / name_list_files, 'w') as f:
                output = '\n'.join(reduc_flist)
                f.write(output + '\n')
            visit_name.append(date_str)

    # keep only unique visit names
    visit_name = list(set(visit_name))

    return visit_name
