#!/usr/bin/env python
# calculate BIC of DESJ2038 models

from astropy.io import fits
import pandas
import numpy
from scipy import special
import bigfloat
import os
import time

basedir = os.path.expanduser('~/lensing/h0licow/desj2038/models/esmod/')
output = 'desj2038_bic.dat'

vd = numpy.array([296.,303.])
vderr_stat = numpy.array([19.,24.])
vderr_sys = 17.
vdcov = numpy.array([[vderr_stat[0]**2,0],[0,vderr_stat[1]**2]])+vderr_sys**2

# calculate number of pixels in arcmask outside of AGN mask
arc_uvis = fits.open(basedir+'arcmask_desj2038_uvis.fits')[0].data.copy()
agn_uvis = fits.open(basedir+'agnmask_desj2038_uvis.fits')[0].data.copy()
lensmask_uvis = fits.open(basedir+'lensmask_desj2038_uvis.fits')[0].data.copy()
arc_wfc = fits.open(basedir+'arcmask_desj2038_wfc.fits')[0].data.copy()
agn_wfc = fits.open(basedir+'agnmask_desj2038_wfc.fits')[0].data.copy()
lensmask_wfc = fits.open(basedir+'lensmask_desj2038_wfc.fits')[0].data.copy()
arc_uvis = arc_uvis[4:-4,4:-4]
agn_uvis = agn_uvis[4:-4,4:-4]
lensmask_uvis = lensmask_uvis[4:-4,4:-4]
arc_wfc = arc_wfc[2:-2,2:-2]
agn_wfc = agn_wfc[2:-2,2:-2]
lensmask_wfc = lensmask_wfc[2:-2,2:-2]
inuvis = numpy.array(numpy.where((lensmask_uvis > 0) & (arc_uvis == 0)))
inwfc = numpy.array(numpy.where((lensmask_wfc > 0) & (arc_wfc == 0)))
npix = len(numpy.where((lensmask_uvis > 0) & (agn_uvis == 0))[0]) + len(numpy.where((lensmask_wfc > 0) & (agn_wfc == 0))[0])
nn = npix + 8 + 3 + 1

# list models, basefile, chi2_impos, gamma, logL_evid, logL_lenslightonly (values from running "glee -f2 -v3")
plmodels = [('esmod_3band_pl_fid11_best_src47',17.44682107,1579.2483000000002,61385.846999999994,-8121.2699999999995),\
    ('esmod_3band_pl_fid11_best_src48',8.632504566,1610.3905,61427.259999999995,-8135.3589999999995),\
    ('esmod_3band_pl_fid11_best_src49',5.825165606,1566.7778999999998,61392.646,-8119.942000000001),\
    ('esmod_3band_pl_fid11_best_src51',12.96890264,1575.8449,61432.009,-8132.299),\
    ('esmod_3band_pl_fid11_best_src52',7.786163845,1555.1454,61421.763,-8131.853999999999),\
    ('esmod_3band_pl_fid11_best_src53',11.10403383,1567.844,61434.609,-8118.273),\
    ('esmod_3band_pl_fid11_best_src54',8.428244392,1539.3811,61397.138999999996,-8113.945),\
    ('esmod_3band_pl_fid11_best_src56',5.33559961,1550.0596,61409.983,-8119.597),\
    ('esmod_3band_pl_fid11_best_src58',6.851822903,1530.9807,61434.009,-8120.959000000001),\
    ('esmod_3band_pl_fid11_best_src60',7.07736124,1529.1327,61429.174,-8118.064),\
    ('esmod_3band_pl_fid11',7.891421438,1590.83,61423.034,-8121.325000000001),\
    ('esmod_3band_pl_noagnwht10',35.82283233,2225.1984,55138.6721,-7933.559),\
    ('esmod_3band_pl_agn1p8',7.714752967,1571.3903,61411.979999999996,-8123.389),\
    ('esmod_3band_pl_src10m9',12.46186636,1599.5339000000001,61332.834,-8067.576),\
    ('esmod_3band_pl_src10p8',8.476596949,1537.0257,61403.604,-8193.420999999998),\
    ('esmod_3band_pl_arcmask1p7',8.110291883,1522.8424,61411.252,-8260.063)]
compmodels = [('esmod_3band_comp_fid_G15_best_src47',43.87311158,1582.268,60923.23,-8491.9),\
    ('esmod_3band_comp_fid_G15_best_src48',50.17793785,1580.4671999999998,60912.311,-8445.658),\
    ('esmod_3band_comp_fid_G15_best_src49',48.27900177,1569.3084999999999,60887.401999999995,-8440.337),\
    ('esmod_3band_comp_fid_G15_best_src51',37.11495205,1560.5587,60915.504,-8454.038),\
    ('esmod_3band_comp_fid_G15_best_src52',47.89032732,1565.0978,60921.501000000004,-8445.643),\
    ('esmod_3band_comp_fid_G15_best_src53',43.08946215,1537.7237,60927.032,-8467.098),\
    ('esmod_3band_comp_fid_G15_best_src54',39.06749082,1545.1433,60935.706,-8461.91),\
    ('esmod_3band_comp_fid_G15_best_src56',42.75114226,1528.6513,60941.038,-8473.826000000001),\
    ('esmod_3band_comp_fid_G15_best_src58',42.3943053,1528.4035,60934.971,-8463.048999999999),\
    ('esmod_3band_comp_fid_G15_best_src60',45.60476695,1528.2394,60940.405,-8457.096000000001),\
    ('esmod_3band_comp_fid_G15',48.46469962,1566.5074999999997,60959.273,-8498.473),\
    ('esmod_3band_comp_noagnwht6',67.52330634,1605.0691000000002,59871.648,-8362.891),\
    ('esmod_3band_comp_agn1p4',41.61197771,1590.3704,60936.215000000004,-8474.204),\
    ('esmod_3band_comp_src10m6',39.25825797,1569.0366000000001,60743.56,-8474.428),\
    ('esmod_3band_comp_src10p3',40.28843286,1536.7778999999998,60975.225000000006,-8498.221),\
    ('esmod_3band_comp_arcmask1p6',26.61890493,1569.2862999999998,61000.78,-8599.666000000001)]

plbicdata = pandas.DataFrame(columns=['basename','model','k','kin_logL','im_logL','esr_logL','logL','bic','sigma_bic','bic_wht'])
compbicdata = pandas.DataFrame(columns=['basename','model','k','kin_logL','im_logL','esr_logL','logL','bic','sigma_bic','bic_wht'])

# Power-law models
for basefile,chi2_im,gamma,evidsr_logL,ll_logL in plmodels:
    looptime = time.time()
    data = pandas.read_csv(basedir+basefile+'.mc',header=0,skiprows=[1],delim_whitespace=True)
    ltd = pandas.read_csv(basedir+'ImpSamp_LTD_'+basefile+'.dat',header=0,skiprows=[],delim_whitespace=True)
    f475 = fits.open(basedir+basefile+'_best_es001_im.fits')[0].data.copy()[9]
    f814 = fits.open(basedir+basefile+'_best_es002_im.fits')[0].data.copy()[9]
    f160 = fits.open(basedir+basefile+'_best_es003_im.fits')[0].data.copy()[9]
    if len(data) > len(ltd): data = data[0:len(ltd)]
    best = data['logP0018'].idxmax()
    ngaussprior = int(os.popen('grep -o "gaussian" '+basedir+basefile+'_best | wc -l').read())-1
    kk =  len(data.columns) - 7 - ngaussprior + 2 + 1
    kin_logL = -0.5*numpy.matmul(numpy.matmul(vd-ltd.loc[best,'vd_pred[km/s]'],numpy.linalg.inv(vdcov)),vd[:,numpy.newaxis]-ltd.loc[best,'vd_pred[km/s]']) - 0.5*numpy.log(numpy.linalg.det(vdcov)) - numpy.log(2*numpy.pi)
    im_logL = -chi2_im/2.
    esr_logL = evidsr_logL+ll_logL
    logL = esr_logL + kin_logL + im_logL
    bic = numpy.log(nn)*kk - 2*logL
    plbicdata = plbicdata.append({'basename':basefile,'model':'pl','k':kk,'kin_logL':kin_logL[0],'im_logL':im_logL,'esr_logL':esr_logL,'logL':logL[0],'bic':bic[0],'sigma_bic':-1,'bic_wht':-1},ignore_index=True)
    print('Time this loop='+str('{:.3f}'.format(time.time()-looptime))+' s; '+basefile)
sigmabic = numpy.std(plbicdata.loc[0:10,'bic'])
plbicdata['sigma_bic'] = sigmabic
for ii in range(len(plbicdata)):
    plbicdata.loc[ii,'bic_wht'] = 0.5 + 0.5*special.erf(-(plbicdata.loc[ii,'bic']-numpy.min(plbicdata.loc[plbicdata.index[10:],'bic']))/(numpy.sqrt(2.)*plbicdata.loc[ii,'sigma_bic'])) + 0.5*numpy.exp(-(plbicdata.loc[ii,'bic']-numpy.min(plbicdata.loc[plbicdata.index[10:],'bic'])-plbicdata.loc[ii,'sigma_bic']**2/4.)/2.)*special.erfc(-(plbicdata.loc[ii,'bic']-numpy.min(plbicdata.loc[plbicdata.index[10:],'bic'])-plbicdata.loc[ii,'sigma_bic']**2/2.)/(numpy.sqrt(2)*plbicdata.loc[ii,'sigma_bic']))
plbicdata['bic_wht'] /= numpy.sum(plbicdata.loc[10:,'bic_wht'])

# Composite models
for basefile,chi2_im,gamma,evidsr_logL,ll_logL in compmodels:
    looptime = time.time()
    data = pandas.read_csv(basedir+basefile+'.mc',header=0,skiprows=[1],delim_whitespace=True)
    ltd = pandas.read_csv(basedir+'ImpSamp_LTD_'+basefile+'.dat',header=0,skiprows=[],delim_whitespace=True)
    f475 = fits.open(basedir+basefile+'_best_es001_im.fits')[0].data.copy()[9]
    f814 = fits.open(basedir+basefile+'_best_es002_im.fits')[0].data.copy()[9]
    f160 = fits.open(basedir+basefile+'_best_es003_im.fits')[0].data.copy()[9]
    if len(data) > len(ltd): data = data[0:len(ltd)]
    best = data['logP0018'].idxmax()
    ngaussprior = int(os.popen('grep -o "gaussian" '+basedir+basefile+'_best | wc -l').read())-1
    kk =  len(data.columns) - 7 - ngaussprior + 2 + 1
    kin_logL = -0.5*numpy.matmul(numpy.matmul(vd-ltd.loc[best,'vd_pred[km/s]'],numpy.linalg.inv(vdcov)),vd[:,numpy.newaxis]-ltd.loc[best,'vd_pred[km/s]']) - 0.5*numpy.log(numpy.linalg.det(vdcov)) - numpy.log(2*numpy.pi)
    im_logL = -chi2_im/2.
    esr_logL = evidsr_logL+ll_logL
    logL = esr_logL + kin_logL + im_logL
    bic = numpy.log(nn)*kk - 2*logL
    compbicdata = compbicdata.append({'basename':basefile,'model':'comp','k':kk,'kin_logL':kin_logL[0],'im_logL':im_logL,'esr_logL':esr_logL,'logL':logL[0],'bic':bic[0],'sigma_bic':-1,'bic_wht':-1},ignore_index=True)
    print('Time this loop='+str('{:.3f}'.format(time.time()-looptime))+' s; '+basefile)
sigmabic = numpy.std(compbicdata.loc[0:10,'bic'])
compbicdata['sigma_bic'] = sigmabic
for ii in range(len(compbicdata)):
    compbicdata.loc[ii,'bic_wht'] = 0.5 + 0.5*special.erf(-(compbicdata.loc[ii,'bic']-numpy.min(compbicdata.loc[compbicdata.index[10:],'bic']))/(numpy.sqrt(2.)*compbicdata.loc[ii,'sigma_bic'])) + 0.5*numpy.exp(-(compbicdata.loc[ii,'bic']-numpy.min(compbicdata.loc[compbicdata.index[10:],'bic'])-compbicdata.loc[ii,'sigma_bic']**2/4.)/2.)*special.erfc(-(compbicdata.loc[ii,'bic']-numpy.min(compbicdata.loc[compbicdata.index[10:],'bic'])-compbicdata.loc[ii,'sigma_bic']**2/2.)/(numpy.sqrt(2)*compbicdata.loc[ii,'sigma_bic']))
compbicdata['bic_wht'] /= numpy.sum(compbicdata.loc[10:,'bic_wht'])

# combine and output
bicdata = plbicdata.append(compbicdata,ignore_index=True)
bicdata.to_csv(output,index=False)
os.system('mv '+output+' '+basedir)