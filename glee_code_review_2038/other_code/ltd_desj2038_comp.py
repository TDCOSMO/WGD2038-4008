#!/usr/bin/env python
# Modified by KCW 7/31/2019 - DESJ2038 chains
#
# Note: only runs on python 2.x for now
# Note: to run the interpolation packages, need to reset the LD_LIBRARY_PATH (the ones in my lib are in conflict with the default ones)
# do:
# setenv LD_LIBRARY_PATH

from scipy import interpolate
import numpy
import sys, distances, math, os
from math import log10,pi
import datetime
from numpy.random import random as R, randn as RN
from vdmodel import sigma_model as sm
import time
import cPickle
from vdmodel_2013 import sigma_model,profiles
from numpy import *
from cgsconstants import *
import NFW,tPIEMD
import ndinterp
import pandas
from astropy.convolution import convolve, Box1DKernel

starttime = time.time()

# output dir:
# ===========
basedir = os.path.expanduser('~/code_review_2038/')
kextdir = os.path.expanduser('~/code_review_2038/redesj2038k_extdistributions/')

# For manual parallelization:
# set seed, number of parts for the wmap chain.  These will be reflected in output file.
iseed = 1
ipart = 1
npart = 1 # keep fixed

numpy.random.seed(iseed)

# composite models
def ltd_comp_desj2038(lensid,amp1,amp2,amp3,burnin=0):
    lensfile = basedir + lensid + '.mc'
    if os.path.isfile(lensfile) != True:
        lensfile = basedir + lensid + '.mcmc'
    
    # additional cosmo parameters fixed
    om = 0.3
    ol = 0.7
    w = -1.
    h0 = 70.
    
    LTsamp = pandas.read_csv(lensfile,header=0,skiprows=[1],delim_whitespace=True)
    LT_wht = LTsamp.loc[burnin:,'weight'].values
    if 'H0' in LTsamp.columns:
        LT_H0 = LTsamp.loc[burnin:,'H0'].values
    else:
        LT_H0 = numpy.ones_like(LT_wht)*h0
    if 'Om' in LTsamp.columns:
        LT_Om = LTsamp.loc[burnin:,'Om'].values
    else:
        LT_Om = numpy.ones_like(LT_wht)*om
    if 'w' in LTsamp.columns:
        LT_w = LTsamp.loc[burnin:,'w'].values
    else:
        LT_w = numpy.ones_like(LT_wht)*w
    LT_MtoL = LTsamp.loc[burnin:,'l0000p01shear'].values

    LT_q1 = LTsamp.loc[burnin:,'l0001p02piemd'].values
    LT_wc1 = LTsamp.loc[burnin:,'l0001p05piemd'].values
    LT_wt1 = LTsamp.loc[burnin:,'l0002p05piemd'].values

    LT_q2 = LTsamp.loc[burnin:,'l0003p02piemd'].values
    LT_wc2 = LTsamp.loc[burnin:,'l0003p05piemd'].values
    LT_wt2 = LTsamp.loc[burnin:,'l0004p05piemd'].values
    
    LT_q3 = LTsamp.loc[burnin:,'l0005p02piemd'].values
    LT_wc3 = LTsamp.loc[burnin:,'l0005p05piemd'].values
    LT_wt3 = LTsamp.loc[burnin:,'l0006p05piemd'].values

    LT_nfw_q = LTsamp.loc[burnin:,'l0007p02nfw'].values
    LT_nfw_mlpthE = LTsamp.loc[burnin:,'l0007p04nfw'].values
    LT_nfw_rs = LTsamp.loc[burnin:,'l0007p05nfw'].values
    
    LT_shear = LTsamp.filter(like='shear',axis=1).iloc[:,-2].values
    nsamp = len(LT_wht)

    # prior for rani  
    reff = 1.50
    rEin_tot = 1.53 #thE from PL model (accounting for Dds/Ds^(1/(2*gam)) scaling)
    rani_min = 0.5*reff
    rani_max = 5.*reff
    rani_samp = numpy.random.random(nsamp)*(rani_max - rani_min) + rani_min

    # compute the cosmological distances to rescale the mlp_thetaE to the appropriate ones for the lens/sr redshift
    zl1 = 0.2283
    zs = 0.777
    Dt = numpy.empty(nsamp)
    Dds = numpy.empty(nsamp)
    Ds = numpy.empty(nsamp)
    Dd = numpy.empty(nsamp)

    # use Matt's distances model to compute ang diam distances given cosmo params
    Dist = distances.Distance()

    for i in range(0,nsamp):
        # calculate the time-delay distance:
        Dist.OMEGA_M = LT_Om[i]
        Dist.OMEGA_L = 1.-LT_Om[i]
        Dist.w = LT_w[i]
        Dist.h = LT_H0[i]/100.
    
        Dds[i] = Dist.Da(zl1, zs)
        Ds[i] = Dist.Da(zs)
        Dd[i] = Dist.Da(zl1)
    
        Dt[i] = (1+zl1)*Dd[i]*Ds[i]/Dds[i]

    LT_shear = LT_shear * (Dds/Ds)

    # concatenate if needed:
    nconcat = 1
    H0_samp=numpy.tile(LT_H0,nconcat)
    om_samp=numpy.tile(LT_Om,nconcat)
    ol_samp=numpy.tile(1.-LT_Om,nconcat)
    w_samp=numpy.tile(LT_w,nconcat)
    wht_samp=numpy.tile(LT_wht,nconcat)
    Dt_samp=numpy.tile(Dt,nconcat)

    # for mass within theta_E:
    Dds_samp = numpy.tile(Dds, nconcat)
    Ds_samp = numpy.tile(Ds, nconcat)
    Dd_samp = numpy.tile(Dd, nconcat)

    # for lens light profile pars
    wc1_samp = numpy.tile(LT_wc1, nconcat)
    wt1_samp = numpy.tile(LT_wt1, nconcat)
    q1_samp = numpy.tile(LT_q1, nconcat)

    wc2_samp = numpy.tile(LT_wc2, nconcat)
    wt2_samp = numpy.tile(LT_wt2, nconcat)
    q2_samp = numpy.tile(LT_q2, nconcat)
    
    wc3_samp = numpy.tile(LT_wc3, nconcat)
    wt3_samp = numpy.tile(LT_wt3, nconcat)
    q3_samp = numpy.tile(LT_q3, nconcat)
    
    LT_shear_samp = numpy.tile(LT_shear, nconcat)

    nsamp = nsamp*nconcat

    # convert the Einstein radii from mlp to one pertaining to strong lens/sr redshifts
    # ========================================================================

    nfw_rE_3comp_samp = numpy.tile(LT_nfw_mlpthE, nconcat)
    nfw_rE_3comp_samp = nfw_rE_3comp_samp * Dds_samp/Ds_samp

    bary_MtoL_3comp_samp = numpy.tile(LT_MtoL, nconcat)
    bary_MtoL_3comp_samp = bary_MtoL_3comp_samp * Dds_samp/Ds_samp


    # get nfw scale radius range for dynamical modeling interpolation
    nfw_rs_3comp_samp = numpy.tile(LT_nfw_rs, nconcat)

    # kappa external
    # ==============
    medshear = numpy.round(numpy.median(LT_shear),decimals=2)
    medshearstr = '{:.2f}'.format(medshear).replace('.','')
    # use the kext from EDI
    kextfile = kextdir+'gamma'+medshearstr+'kappahist_2038_measured_3innermask_nobeta_removehandpicked_zgap-1.0_-1.0_powerlaw_120_gal_120_gamma_120_oneoverr_45_gal_45_oneoverr_22.5_med_increments2_2_2_2_2_emptymsk_shearwithoutprior.cat'
    kext_cdf_table = numpy.loadtxt(kextfile,skiprows=1)  
    kext_cdf_table = convolve(kext_cdf_table,Box1DKernel(13))  # smooth the distribution
    # check if cumulative
    if kext_cdf_table[-1] != max(kext_cdf_table):
        cdf = numpy.cumsum(kext_cdf_table)
        kext_cdf_table = cdf / max(cdf)

    # extract only the entries with unique cdf value (for interpolation with cdf as x coordinate)
    kext_val_uniq = numpy.empty(0)
    kext_cdf_uniq = numpy.empty(0)
    icount = 0
    kvals = numpy.linspace(-0.2,1,2001)
    
    for i in range (0,len(kext_cdf_table)):
        if (kext_cdf_table[icount] != kext_cdf_table[i]):
            # next entry is uniq, so add to list
            kext_val_uniq=numpy.append(kext_val_uniq, kvals[i])
            kext_cdf_uniq=numpy.append(kext_cdf_uniq, kext_cdf_table[i])
            icount = i
    kext_cdf_model = interpolate.splrep(kext_cdf_uniq, kext_val_uniq, s=0, k=1) # note that need k=1 (interpolation polynomial degree, otherwise the new kext with gamma' gaussian whts gives strange kext fits and bad pdf reproduction.

    # get kext samples (array of size nsamp)
    kext_3comp_samp = interpolate.splev(numpy.random.random(nsamp),kext_cdf_model)

    # get the dynamics likelihood weights (see Ale's bulget+halo_test.py or RXJ1131's 14-0221CombinePL_3comp_LTD_grid_unifH0.py)
    currenttime = datetime.datetime.now().time()
    print 'current time before Ale dyn like comp ', currenttime

    vd_3comp_samp = numpy.zeros(nsamp)

    MpcToCm = 3.08567758e24
    Dds_samp = MpcToCm * Dds_samp
    Ds_samp = MpcToCm * Ds_samp
    Dd_samp = MpcToCm * Dd_samp

    S_cr_samp = c**2/(4*numpy.pi*G)*Ds_samp/Dd_samp/Dds_samp/M_Sun*(Dd_samp*arcsec2rad)**2 

    menc_NFW_samp = S_cr_samp*numpy.pi* nfw_rE_3comp_samp**2
    norm_samp = numpy.ones(nsamp)
    for i in range(0,nsamp):
        norm_samp[i] = menc_NFW_samp[i]/NFW.M2d(nfw_rE_3comp_samp[i],nfw_rs_3comp_samp[i])
        # norm_samp relates the 2d (cylindrical) mass to 3d (spherical) mass.  3D mass is norm_samp*NFW.M3d(r,nfw_rs_3comp_samp)


    # the two piemds describing the baryons
    rein1_samp = bary_MtoL_3comp_samp * amp1 
    rc1_samp = wc1_samp*2*q1_samp**0.5/(1.+q1_samp)
    rt1_samp = wt1_samp*2*q1_samp**0.5/(1.+q1_samp)

    sigma01_samp = S_cr_samp*rein1_samp/2.*(1./wc1_samp - 1./wt1_samp) #central sigma at r=0
    Mstar1_samp = numpy.zeros(nsamp)
    for i in range(nsamp):
        Mstar1_samp[i] = sigma01_samp[i]/tPIEMD.Sigma(0.,[rc1_samp[i],rt1_samp[i]]) #tPIEMD.Sigma calculates Sigma at r=0.  tPIEMD has units that integrates to 1.  So Mstar1_samp allows to convert mass to the units of tPIEMD.  Mstar1 is actually the total mass of stars (in units of sigmacrit*r^2). --> for now, think of this as a mass normalization.

    rein2_samp = bary_MtoL_3comp_samp * amp2
    rc2_samp = wc2_samp*2*q2_samp**0.5/(1.+q2_samp)
    rt2_samp = wt2_samp*2*q2_samp**0.5/(1.+q2_samp)

    sigma02_samp = S_cr_samp*rein2_samp/2.*(1./wc2_samp - 1./wt2_samp)
    Mstar2_samp = numpy.zeros(nsamp)
    for i in range(nsamp):
        Mstar2_samp[i] = sigma02_samp[i]/tPIEMD.Sigma(0.,[rc2_samp[i],rt2_samp[i]])
        
    rein3_samp = bary_MtoL_3comp_samp * amp3
    rc3_samp = wc3_samp*2*q3_samp**0.5/(1.+q3_samp)
    rt3_samp = wt3_samp*2*q3_samp**0.5/(1.+q3_samp)

    sigma03_samp = S_cr_samp*rein3_samp/2.*(1./wc3_samp - 1./wt3_samp)
    Mstar3_samp = numpy.zeros(nsamp)
    for i in range(nsamp):
        Mstar3_samp[i] = sigma03_samp[i]/tPIEMD.Sigma(0.,[rc3_samp[i],rt3_samp[i]])

    Mstar_samp = Mstar1_samp + Mstar2_samp + Mstar3_samp

    a1_samp = Mstar1_samp/Mstar_samp
    a2_samp = Mstar2_samp/Mstar_samp
    a3_samp = Mstar3_samp/Mstar_samp

    # for removing kext contribution
    # see rEin_tot above (currently from Courbin et al. 2011)
    m2d_tpiemd1 = numpy.zeros(nsamp)
    m2d_tpiemd2 = numpy.zeros(nsamp)
    m2d_tpiemd3 = numpy.zeros(nsamp)
    for i in range(nsamp):
        m2d_tpiemd1[i] = tPIEMD.M2d(rEin_tot,[rc1_samp[i],rt1_samp[i]]) #fraction of total mass within rEin_tot (since tPIEMD is normalized to 1)
        m2d_tpiemd2[i] = tPIEMD.M2d(rEin_tot,[rc2_samp[i],rt2_samp[i]])
        m2d_tpiemd3[i] = tPIEMD.M2d(rEin_tot,[rc3_samp[i],rt3_samp[i]])

    #defines the vertices of the rectangular aperture to calculate vdisp in.
    aperture = [0.,0.9,0.,0.9]
    seeing = 0.9
    vd = numpy.array([296.,303.])
    vderr_stat = numpy.array([19.,24.])
    vderr_sys = 17.
    vdcov = numpy.array([[vderr_stat[0]**2,0],[0,vderr_stat[1]**2]])+vderr_sys**2

    # stars: 
    # =====
    currenttime = datetime.datetime.now().time()
    print 'current time before computing s2 of stars', currenttime

    stars_s2_samp = numpy.zeros(nsamp)
    for i in range(nsamp):
        lp_pars = [a1_samp[i],rc1_samp[i],rt1_samp[i],a2_samp[i],rc2_samp[i],rt2_samp[i],a3_samp[i],rc3_samp[i],rt3_samp[i]]
        stars_s2_samp[i] = sigma_model.sigma2general(lambda r: a1_samp[i]*tPIEMD.M3d(r,[rc1_samp[i],rt1_samp[i]]) + a2_samp[i]*tPIEMD.M3d(r,[rc2_samp[i],rt2_samp[i]]) + a3_samp[i]*tPIEMD.M3d(r,[rc3_samp[i],rt3_samp[i]]),aperture,lp_pars,seeing=seeing,light_profile=profiles.twotPIEMD,reval0=0.5*(rt1_samp[i] + rt2_samp[i] + rt3_samp[i]), anisotropy='OM', anis_par=rani_samp[i])
        print '\r Progress: '+str(i)+'/'+str(nsamp),
        sys.stdout.flush()

    # halo:
    # =====
    currenttime = datetime.datetime.now().time()
    print 'current time before computing s2 of halo', currenttime

    halo_s2_samp = numpy.zeros(nsamp)
    for i in range(nsamp):
        lp_pars = [a1_samp[i], rc1_samp[i], rt1_samp[i], a2_samp[i], rc2_samp[i], rt2_samp[i], a3_samp[i], rc3_samp[i], rt3_samp[i]]
        halo_s2_samp[i] = sigma_model.sigma2general(lambda r: NFW.M3d(r,nfw_rs_3comp_samp[i]),aperture,lp_pars,seeing=seeing,light_profile=profiles.twotPIEMD,reval0=0.5*(rt1_samp[i] + rt2_samp[i] + rt3_samp[i]), anisotropy='OM', anis_par=rani_samp[i])
        print '\r Progress: '+str(i)+'/'+str(nsamp),
        sys.stdout.flush()

    ## evaluation of predicted velocity dispersion of samples
    # ------------------------------------------------------
    currenttime = datetime.datetime.now().time()
    print 'current time before computing s2 of samples', currenttime

    # now let's remove the external convergence
    Mstar_samp *= 1. - kext_3comp_samp
    norm_samp *= 1. - kext_3comp_samp


    ## get into km/s units
    s2_stars_samp = stars_s2_samp * G*Mstar_samp*M_Sun/10.**10/(Dd_samp*arcsec2rad)
    s2_halo_samp = halo_s2_samp* G*norm_samp*M_Sun/10.**10/(Dd_samp*arcsec2rad)
    vd_3comp_samp = (s2_stars_samp + s2_halo_samp)**0.5

    logp = numpy.array([])
    logp = numpy.append(logp,[-0.5*numpy.matmul(numpy.matmul(vd-ii,numpy.linalg.inv(vdcov)),vd[:,numpy.newaxis]-ii) - 0.5*numpy.log(numpy.linalg.det(vdcov)) - numpy.log(2*pi) for ii in vd_3comp_samp])
    wht_samp_D_3comp = numpy.exp(logp)
    
    # scale back Dist to Mpc
    Dds_samp = Dds_samp / MpcToCm
    Ds_samp = Ds_samp / MpcToCm
    Dd_samp = Dd_samp / MpcToCm

    currenttime = datetime.datetime.now().time()
    print 'current time after Ale dyn like comp ', currenttime
    return wht_samp,H0_samp,om_samp,ol_samp,w_samp,Dd_samp,Dt_samp,nfw_rE_3comp_samp,nfw_rs_3comp_samp,bary_MtoL_3comp_samp,kext_3comp_samp,vd_3comp_samp,LT_shear_samp,wht_samp_D_3comp

# list input files and relative amplitudes of Chameleon components, index-matched to input composite models
infiles_compos = ['esmod_3band_comp_fid']
amp1_comp = [261.885]
amp2_comp = [11.3588]
amp3_comp = [225.52]
instuff_compos = zip(infiles_compos,amp1_comp,amp2_comp,amp3_comp)

# composite models
for infile,amp1,amp2,amp3 in instuff_compos:
    if infile == '': break
    wht_samp,H0_samp,om_samp,ol_samp,w_samp,Dd_samp,Dt_samp,nfw_rE_3comp_samp,nfw_rs_3comp_samp,bary_MtoL_3comp_samp,kext_3comp_samp,vd_3comp_samp,LT_shear_samp,wht_samp_D_3comp = ltd_comp_desj2038(infile,amp1,amp2,amp3,burnin=0)
    wht_samp[numpy.isnan(wht_samp)] = 0.
    wht_samp_D_3comp[numpy.isnan(wht_samp_D_3comp)] = 0.

    outfile = basedir + "ImpSamp_LTD_"+infile+".dat"
    fout = open(outfile, 'w')
    fout.write("weight H0 OmegaM OmegaL w Ddmod Dtmod Dtmod/(1-kext) nfw_re nfw_rs bary_MtoL kext vd_pred[km/s] shear wht_model wht_dyn\n")
    if (blind == 1):
        fout.write("#Note: cosmo param values are shifted/masked in python imp samp script\n")
    for i in range(0,len(wht_samp)):
        fout.write("%le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le\n"%(wht_samp[i]*wht_samp_D_3comp[i], H0_samp[i], om_samp[i], ol_samp[i], w_samp[i], Dd_samp[i], Dt_samp[i], Dt_samp[i]/(1.-kext_3comp_samp[i]), nfw_rE_3comp_samp[i], nfw_rs_3comp_samp[i], bary_MtoL_3comp_samp[i], kext_3comp_samp[i], vd_3comp_samp[i], LT_shear_samp[i], wht_samp[i], wht_samp_D_3comp[i]))
    fout.close()
    print(outfile+'\n')

print('Total runtime: '+str('{:.1f}'.format(time.time()-starttime))+' s')