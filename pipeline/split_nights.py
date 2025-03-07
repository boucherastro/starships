import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.signal import find_peaks

from datetime import timedelta

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

def split_night(obs_dir, path_fig):
    # Frames for observation files reduced with APERO pipeline
    FILE_FRAMES = {'tcorr': '{}t.fits',
                'e2ds': '{}e.fits'
                # 'pclean': '{}_pp_tellu_pclean_AB.fits'
                }

    obs_jd = []
    filenames = []
    for entry in obs_dir.glob('*e.fits'):
        valid_file = True
        hdu = fits.open(entry)
        
        # Check if other files needed exist
        obs_id = hdu[0].header['arcfile'][:-5] # changed for NIRPS
        for reduction_type, f_format in FILE_FRAMES.items():
            other_file = f_format.format(obs_id)
            other_file = obs_dir / Path(other_file)
            if not other_file.is_file():
                print(f'{reduction_type} not found with filename: {other_file.name}')
                valid_file = False
        
        if valid_file:
            obs_jd.append(hdu[0].header['mjd-obs'])
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
        hdu = fits.open(obs_dir / filenames[tr_tag[0]])
        text = f"{filenames[tr_tag[0]]}\n{hdu[0].header['HIERARCH ESO OBS PROG ID']}, {hdu[0].header['DATE']}"
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
        hdu = fits.open(obs_dir / filenames_tr[0])
        date = Time(hdu[0].header['DATE']).datetime

        # modify date if hour is after midnight
        if date.hour >= 0 and date.hour < 12:
            date = date - timedelta(days=1)

        date = date.strftime('%Y-%m-%d-%H')

        # name_list_files = Path(f'list_e2ds_visit_{date}')
        # print(f'Writing to {name_list_files}')
        # with open(obs_dir / name_list_files, 'w') as f:
        #     output = '\n'.join(filenames_tr)
        #     f.write(output + '\n')
        
        file_lists = {key: [] for key in FILE_FRAMES}
        for tr_file in filenames_tr:
            hdu = fits.open(obs_dir / tr_file)
            obs_id = hdu[0].header['arcfile'][:-5]
            for reduction_type, f_format in FILE_FRAMES.items():
                other_file = f_format.format(obs_id)
                file_lists[reduction_type].append(other_file)
                
        for reduc_type, reduc_flist in file_lists.items():
            name_list_files = Path(f'list_{reduc_type}_{date}')
            print(f'Writing to {name_list_files}')
            with open(obs_dir / name_list_files, 'w') as f:
                output = '\n'.join(reduc_flist)
                f.write(output + '\n')
            visit_name.append(str(date))

    # keep only unique visit names
    visit_name = list(set(visit_name))

    return visit_name 

    