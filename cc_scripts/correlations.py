import os
import numpy as np
from pathlib import Path
from sys import path
# path.append('/home/adb/PycharmProjects/')
# path.append('/home/adb/PycharmProjects/starships_analysis/')

import matplotlib.pyplot as plt
from importlib import reload
from itertools import product
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as const

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import starships.plotting_fcts as pf 
import starships.planet_obs as pl_obs
from starships import homemade as hm
from starships import correlation as corr
from starships import correlation_class as cc
from starships.planet_obs import Observations,Planet

# cc=reload(cc)

# - Setting plot parameters - #
plt.rc('figure', figsize=(9, 6))

couleurs = hm.get_colors('magma', 50)[5:-2]

# from cycler import cycler
# plt.rcParams['axes.prop_cycle'] = cycler(color=hm.get_colors('magma', 50))
# ex : list_of_color = [(i,0,0) for i in np.arange(10)/10]

fanne = np.load('/home/adb/projects/def-dlafre/bouchea3/saved_correl/ccfll_seq_t2_basic_DAY_all_23aug23.npz')


pl_obs=reload(pl_obs)

pl_name = 'WASP-33 b'
planet_obj=Planet(pl_name)
planet_obj.ap = 0.0259*u.au
planet_obj.R_star = (planet_obj.ap/3.69).to(u.R_sun)
planet_obj.R_pl = (0.1143*planet_obj.R_star).to(u.R_jup)
planet_obj.M_star = 1.561*const.M_sun


try:
    base_dir = os.environ['SCRATCH']
except KeyError:
    base_dir = Path.home()

reduc_dir = base_dir / Path('DataAnalysis/SPIRou/Reductions/WASP-33')

filename_list = ['v07254_wasp33_2-pc_mask_wings90_data_trs_night1.npz',
                 'v07254_wasp33_3-pc_mask_wings90_data_trs_night1.npz',
                 'v07254_wasp33_4-pc_mask_wings90_data_trs_night1.npz',
                 'v07254_wasp33_5-pc_mask_wings90_data_trs_night1.npz',
                 'v07254_wasp33_6-pc_mask_wings90_data_trs_night1.npz'
                ]
# filename = 'v07254_wasp33_2-pc_mask_wings97_data_trs_night1.npz'
obs_list = []
for filename in filename_list:
    obs = pl_obs.load_single_sequences(filename, pl_name, path=reduc_dir,
                              load_all=False, filename_end='', plot=False, planet=planet_obj)
    obs_list.append(obs)
    

masked_ratio_total = [obs.final.mask.sum() / obs.final.size for obs in obs_list]

masked_ratio = [obs.final.mask.sum(axis=0).sum(axis=-1) / obs.final.shape[0] / obs.final.shape[-1]
                for obs in obs_list]

masked_ratio = np.array(masked_ratio)

n_pc_list = [obs.params[5] for obs in obs_list]


# plt.plot(n_pc_list, masked_ratio_total, ".-", label='total')
for order in [42,]:
    y_plot = masked_ratio[:, order]
    plt.plot(n_pc_list, y_plot, ".-",label=order)
plt.legend()

_  = plt.plot(obs_list[1].wave[:, 46, :].T, obs_list[1].tellu[:, 46, :].T)

pf.plot_order(obs_list[1], 47)

## read pre-generated model
# model_file = np.load('test_mod_WASP-33_inverted_CO_only.npz')
# wave_mod, model_spec = model_file['wave_mod'], model_file['model_spec']

model_file = np.load('best_logl_model_HR.npz')
wave_mod, model_spec = model_file['wave'], model_file['spec']
plt.plot(wave_mod, model_spec)


# doing the correlation
# def one_corr():


n_RV_inj=151
corrRV0 = np.linspace(-150, 150, n_RV_inj)
# Kp_array = np.array([obs.Kp.value]) 

kind_trans = 'emission'

output_dir = base_dir / Path('DataAnalysis/SPIRou/DetectionMaps/WASP-33')
# Create output directory if it does not exist
output_dir.parent.mkdir(parents=True, exist_ok=True)

model_name = 'best_fit_JR'

visit_name = 'night2'
filename_list = [f'v07254_wasp33_{n_pc}-pc_mask_wings{mask_w}_data_trs_{visit_name}.npz'
                 for mask_w, n_pc in product([90, 95, 97], range(2, 6))]
print('\n'.join(filename_list))
# #     f'v07254_wasp33_2-pc_mask_wings90_data_trs_{visit_name}.npz',
# #     f'v07254_wasp33_3-pc_mask_wings90_data_trs_{visit_name}.npz',
# #     f'v07254_wasp33_4-pc_mask_wings90_data_trs_{visit_name}.npz',
# #     f'v07254_wasp33_5-pc_mask_wings90_data_trs_{visit_name}.npz',
# #     f'v07254_wasp33_6-pc_mask_wings90_data_trs_{visit_name}.npz',

#     f'v07254_wasp33_2-pc_mask_wings95_data_trs_{visit_name}.npz',
#     f'v07254_wasp33_3-pc_mask_wings95_data_trs_{visit_name}.npz',
#     f'v07254_wasp33_4-pc_mask_wings95_data_trs_{visit_name}.npz',
#     f'v07254_wasp33_5-pc_mask_wings95_data_trs_{visit_name}.npz',
#     f'v07254_wasp33_6-pc_mask_wings95_data_trs_{visit_name}.npz'
# ]

n_pc_list = []
mask_wings_list = []
all_obs = dict()
all_ccf_map = dict()
all_logl_map = dict()
for filename in filename_list:
    obs = pl_obs.load_single_sequences(filename, pl_name, path=reduc_dir,
                              load_all=False, filename_end='', plot=False, planet=planet_obj)
    
    # Generate Kp 
    Kp_array = np.array([obs.Kp.value]) 
        
    n_pc = int(obs.params[5])
    n_pc_list.append(n_pc)
    
    mask_wings = int(obs.params[1] * 100)  # in percent
    mask_wings_list.append(mask_wings)
    
    out_filename = f'{Path(filename).stem}_ccf_logl_seq_{model_name}'
    
    try:
        # Check if already generated
        saved_values = np.load(output_dir / Path(out_filename).with_suffix('.npz'))
        ccf_map = saved_values['corr']
        logl_map = saved_values['logl']
    except FileNotFoundError:
        # Generate 1d correlations
        # ccf_map shape: (n_exposures, n_order, n_Kp, n_vsys, n_pc, n_model) 
        ccf_map, logl_map = corr.calc_logl_injred(
            obs,'seq', planet_obj, Kp_array, corrRV0, [n_pc], wave_mod, model_spec,  kind_trans
        )
    
    
        corr.save_logl_seq(output_dir / Path(out_filename), ccf_map, logl_map,
                           wave_mod, model_spec, n_pc, Kp_array, corrRV0, kind_trans)

    
    all_obs[(n_pc, mask_wings)] = obs
    all_ccf_map[(n_pc, mask_wings)] = ccf_map
    all_logl_map[(n_pc, mask_wings)] = logl_map
    

n_pc_list = np.unique(n_pc_list)
mask_wings_list = np.unique(mask_wings_list)

# out_filename = f'v{apero_version}_wasp33_{n_pc}-pc_mask_wings{mask_wings*100:n}'
order_indices = np.arange(49)  #np.array([46])
id_pc0 = 0

# Plot all ccf and logl as a function of pc
for mask_wings in mask_wings_list:
    ccf_maps_in = [all_ccf_map[(n_pc, mask_wings)] for n_pc in n_pc_list]
    ccf_maps_in = np.concatenate(ccf_maps_in, axis=-2)
    logl_maps_in = [all_logl_map[(n_pc, mask_wings)] for n_pc in n_pc_list]
    logl_maps_in = np.concatenate(logl_maps_in, axis=-2)
    obs = all_obs[(n_pc_list[0], mask_wings)]
    print('mask_wings: ', mask_wings)
    ccf_obj, logl_obj = cc.plot_ccflogl(obs, ccf_maps_in, logl_maps_in, corrRV0,
                                        Kp_array, n_pc_list, id_pc0=id_pc0, orders=order_indices)
    
mask_wings_list, n_pc_list

[print(key, ccf_map.shape) for key, ccf_map in all_ccf_map.items()]

# Plot all ccf and logl as a function of pc
for n_pc in n_pc_list:
    ccf_maps_in = [all_ccf_map[(n_pc, mask_wings)] for mask_wings in mask_wings_list]
    ccf_maps_in = np.concatenate(ccf_maps_in, axis=-1)
    logl_maps_in = [all_logl_map[(n_pc, mask_wings)] for mask_wings in mask_wings_list]
    logl_maps_in = np.concatenate(logl_maps_in, axis=-1)
    obs = all_obs[(n_pc, mask_wings_list[0])]
    print('n_pc: ', n_pc)
    ccf_obj, logl_obj = cc.plot_ccflogl(obs, ccf_maps_in, logl_maps_in, corrRV0,
                                        Kp_array, mask_wings_list, swapaxes=(-2, -1))
    

# Plot single ccf and logl
n_pc, mask_wings = 3, 95

filename = f'v07254_wasp33_{n_pc}-pc_mask_wings{mask_wings}_data_trs_{visit_name}.npz'
obs = pl_obs.load_single_sequences(filename, pl_name, path=reduc_dir,
                              load_all=False, filename_end='', plot=False, planet=planet_obj)

args = [all_something[(n_pc, mask_wings)]
        for all_something in [all_obs, all_ccf_map, all_logl_map]]
ccf_obj, logl_obj = cc.plot_ccflogl(*args, corrRV0, Kp_array, [n_pc], orders=order_indices)

# Generate the ttest map
ccf_obj.ttest_map(all_obs[(n_pc, mask_wings)], kind='logl', vrp=np.zeros_like(obs.vrp), orders=order_indices, 
                  kp0=0, RV_limit=75, kp_step=5, rv_step=2, RV=None, speed_limit=3, icorr=obs.iIn, equal_var=False
                  )


tr=all_obs[(n_pc, mask_wings)]
cobj = ccf_obj
t_value = cobj.ttest_map_tval

t_value_scaled = t_value*(-3)/t_value.min()

(t_in, p_in) = pf.plot_ttest_map_hist(tr, cobj.rv_grid, cobj.map_prf.copy(), 
                                           cobj.ttest_map_kp, cobj.ttest_map_rv, 
                                    t_value_scaled, cobj.ttest_map_params, 
                              plot_trail=True, masked=True, ccf=cobj.map_prf.copy(),
                              vrp=np.zeros_like(tr.vrp), RV=cobj.pos, hist=False,
                                      show_max=False, show_rest_frame=False,
                                    fig_name='', path_fig=None, orders=order_indices)
# fig.get_axes()[0].text(-67.0, 300, r'H$_2$O', fontsize=16, bbox ={'facecolor':'white', 'alpha':0.8})
fig = plt.gcf()
ax = fig.axes

pos_max = -3.2
# - Lines enclosing the maximum -
ax[0].axhline(tr.Kp.to(u.km / u.s).value, linestyle='--', alpha=0.7, color='indigo', 
              xmin=0,xmax=hm.nearest(cobj.interp_grid, pos_max-20)/cobj.interp_grid.size )
ax[0].axhline(tr.Kp.to(u.km / u.s).value, linestyle='--', alpha=0.7, color='indigo',
              xmin=hm.nearest(cobj.interp_grid, pos_max+20)/cobj.interp_grid.size ,xmax=1) 
ax[0].axvline(pos_max, linestyle='--', alpha=0.7, color='indigo', 
                  ymin=0, ymax=(tr.Kp.value - 60)/cobj.ttest_map_kp[-1] )
ax[0].axvline(pos_max, linestyle='--', alpha=0.7, color='indigo',
                  ymin=(tr.Kp.value + 60)/cobj.ttest_map_kp[-1] , ymax=1)
fig.set_size_inches(4.5,3)
plt.tight_layout()
#fig.savefig(data_path+'Figures/'+'t3_H2O_night.pdf', rasterize=True)


_ = pf.plot_all_orders_correl(corrRV0,ccf_obj.data.squeeze(), all_obs[(n_pc, mask_wings)],
                                      icorr=None, logl=False, sharey=True,
                                      vrp=np.zeros_like(all_obs[(n_pc, mask_wings)].vrp),
                              RV_sys=0.0, vmin=None, vmax=None,
                                      vline=None, hline=2, kind='snr', return_snr=True)


fct_model = interp1d(wave_mod, model_spec, kind='cubic')


order_plot = [15, 16, 17, 33, 39, 46, 47]

for i_ord in order_plot:

    wv_ord = all_obs[(n_pc, mask_wings)].wv[i_ord]
    mask_ord = all_obs[(n_pc, mask_wings)].final.mask[0, i_ord]
    
    coeff = np.polyfit(wv_ord, fct_model(wv_ord), 1)
    model_ord = np.ma.array(fct_model(wv_ord), mask=mask_ord)
    
    plt.figure()
#     plt.plot(wave_mod[cond], model_spec[cond])
    poly_fct = np.poly1d(coeff)
    plt.plot(wv_ord, model_ord / poly_fct(wv_ord))
    plt.title(i_ord)
    
#     plt.gca().twinx().plot(wv_ord, mask_ord, 'r')


all_obs[(n_pc, mask_wings)].wv, all_obs[(n_pc, mask_wings)].flux.mask.shape


## combine visits

reload(pl_obs)

model_name = 'best_fit_JR'
filename_dict = {
    '1': 'v07254_wasp33_2-pc_mask_wings95_data_trs_night1.npz',
    '2': 'v07254_wasp33_3-pc_mask_wings95_data_trs_night2.npz',
}

coeffs = [0.532]
ld_model = 'linear'
kind_trans='emission'
combined_ccf = []
combined_logl = []
combined_obs = []
visit_dict = dict()

for key, fname in filename_dict.items():
    obs = pl_obs.load_single_sequences(fname, pl_name, path=reduc_dir,
                              load_all=False, filename_end='', plot=False, planet=planet_obj)
    print(obs.vrp.shape, obs.n_spec)
    
    out_filename = f'{Path(fname).stem}_ccf_logl_seq_{model_name}'
    
    try:
        # Check if already generated
        saved_values = np.load(output_dir / Path(out_filename).with_suffix('.npz'))
        ccf_map = saved_values['corr']
        logl_map = saved_values['logl']
    except FileNotFoundError:
        # Generate 1d correlations
        # ccf_map shape: (n_exposures, n_order, n_Kp, n_vsys, n_pc, n_model) 
        ccf_map, logl_map = corr.calc_logl_injred(
            obs,'seq', planet_obj, Kp_array, corrRV0, [n_pc], wave_mod, model_spec,  kind_trans
        )
    
    
        corr.save_logl_seq(output_dir / Path(out_filename), ccf_map, logl_map,
                           wave_mod, model_spec, n_pc, Kp_array, corrRV0, kind_trans)

    visit_dict[key] = obs
    combined_obs.append(obs)
    combined_ccf.append(ccf_map)
    combined_logl.append(logl_map)


transit_tags = [np.arange(obs.n_spec) for obs in visit_dict.values()]
all_visits = pl_obs.gen_merge_obs_sequence(obs, visit_dict, [1, 2], None, 
                                    coeffs, ld_model, kind_trans, light=True)


for key, val in vars(obs).items():
    try:
        print(key, type(val), len(val))
    except TypeError:
        pass

all_visits.dt = np.concatenate([vst.dt for vst in visit_dict.values()])


# Generate Kp 
Kp_array = np.array([obs.Kp.value]) 

ccf_map = np.concatenate(combined_ccf)
logl_map = np.concatenate(combined_logl)

t1 = combined_obs[0]
t2 = combined_obs[1]

idx_orders = np.array([46])  # np.arange(49)

ccf_obj, logl_obj = cc.plot_ccflogl(all_visits, 
                                    ccf_map,
                                    logl_map,
                                    corrRV0, Kp_array, [1],
                                    split_fig = [0,t1.n_spec,t1.n_spec+t2.n_spec],
                                    orders=idx_orders
                                   )

t=all_visits
ccf_obj.ttest_map(t, kind='logl', vrp=np.zeros_like(t.vrp), orders=idx_orders, 
                  kp0=0, RV_limit=75, kp_step=5, rv_step=2, RV=None, speed_limit=3, icorr=t.iIn, equal_var=False, 
                  )

tr=all_visits
cobj = ccf_obj
t_value = cobj.ttest_map_tval

(t_in, p_in), fig = pf.plot_ttest_map_hist(tr, cobj.rv_grid, cobj.map_prf.copy(), 
                                           cobj.ttest_map_kp, cobj.ttest_map_rv, 
                                    t_value*(-3)/t_value.min(), cobj.ttest_map_params, 
                              plot_trail=True, masked=True, ccf=cobj.map_prf.copy(),
                              vrp=np.zeros_like(tr.vrp), RV=cobj.pos, hist=False,
                                    fig_name='', path_fig=None, cmap='viridis')
fig.get_axes()[0].text(-67.0, 300, r'H$_2$O', fontsize=16, bbox ={'facecolor':'white', 'alpha':0.8})
plt.gcf().set_size_inches(4.5,3)
plt.tight_layout()
# fig.savefig(data_path+'Figures/fig_'+'all_visits_H2O_day.pdf', rasterize=True)

visit_dict['12'] = all_visits


args_filename = [(key, str(visit_dict[key].params[5]), str(int(visit_dict[key].params[1] * 100)))
                 for key in filename_dict.keys()]
args_filename = ['-'.join(values) for values in zip(*args_filename)]
filename = 'v07254_wasp33_nights{}_pc{}_mask_wings{}'.format(*args_filename)

pl_obs.save_sequences(filename, visit_dict, [1, 2, 12], path=reduc_dir)

pl_obs.save_sequences('v07254_wasp33_nights1_pc2_mask_wings95', visit_dict, [1], path=reduc_dir)