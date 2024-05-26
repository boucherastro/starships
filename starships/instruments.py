# Dictionaries for different instruments and/or DRS

# spirou (apero)
spirou = dict()
spirou['name'] = 'SPIRou-APERO'
spirou['airmass'] = 'AIRMASS'
spirou['telaz'] = 'TELAZ'
spirou['adc1'] = 'SBADC1_P'
spirou['adc2'] = 'SBADC2_P'
spirou['mjd'] = 'MJD-OBS'
spirou['bjd'] = 'BJD'
spirou['exptime'] = 'EXPTIME'
spirou['berv'] = 'BERV'

# nirps, apero DRS
nirps_apero = dict()
nirps_apero['name'] = 'NIRPS-APERO'
nirps_apero['airmass'] = 'HIERARCH ESO TEL AIRM START'
nirps_apero['telaz'] = 'HIERARCH ESO TEL AZ'
nirps_apero['adc1'] = 'HIERARCH ESO INS ADC1 START'
nirps_apero['adc2'] = 'HIERARCH ESO INS ADC2 START'
nirps_apero['mjd'] = 'MJD-OBS'
nirps_apero['bjd'] = 'BJD'
nirps_apero['exptime'] = 'EXPTIME'
nirps_apero['berv'] = 'BERV'
nirps_apero['resol'] = 75200
nirps_apero['high_res_wv_lim'] = [0.9, 1.8]
# nirps, geneva/ESPRESSO DRS
# implementing
nirps_geneva = dict()
nirps_geneva['name'] = 'NIRPS-GENEVA'
nirps_geneva['airmass'] = 'HIERARCH ESO TEL AIRM START'
nirps_geneva['telaz'] = 'HIERARCH ESO TEL AZ'
nirps_geneva['adc1'] = 'HIERARCH ESO INS ADC1 START'
nirps_geneva['adc2'] = 'HIERARCH ESO INS ADC2 START'
nirps_geneva['mjd'] = 'MJD-OBS'
nirps_geneva['bjd'] = 'HIERARCH ESO QC BJD'
nirps_geneva['exptime'] = 'EXPTIME'
nirps_geneva['berv'] = 'HIERARCH ESO QC BERV'

igrins_zoe = dict()
igrins_zoe['name'] = 'IGRINS'
igrins_zoe['airmass'] = 'AMSTART'
# igrins_zoe['telaz'] = 'TELRA'
igrins_zoe['adc1'] = 'NADCS'
igrins_zoe['adc2'] = 'NADCS'
igrins_zoe['bjd'] = 'JD-OBS'
igrins_zoe['mjd'] = 'MJD-OBS'
igrins_zoe['exptime'] = 'EXPTIMET'

# dictionary with instrument-DRS names
instruments_drs = {
    'SPIRou-APERO': spirou,
    'NIRPS-APERO': nirps_apero,
    'NIRPS-GENEVA': nirps_geneva,
    'IGRINS': igrins_zoe
}

# instrum = {}

# instrum['spirou'] = {
# 'name': 'spirou',
# 'nord': 49,
# 'npix': 4088,
# 'resol': 64000,
# 'high_res_wv_lim': [0.9, 2.55] 

# }

# instrum['nirps_he'] = {
# 'name': 'nirps_he',
# 'nord': 49,
# 'npix': 4088,
# 'resol': 90000,
# 'high_res_wv_lim': [0.9, 1.8] 

# }

# instrum['nirps_hr'] = {
# 'name': 'nirps_hr',
# 'nord': 49,
# 'npix': 4088,
# 'resol': 100000,
# 'high_res_wv_lim': [0.9, 1.8] 

# }


def load_instrum(instrum_name):
    return instruments_drs[instrum_name]
