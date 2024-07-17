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

# power-law models
def ltd_pl_desj2038(lensid,burnin=0):
    looptime = time.time()
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
    LT_qP = LTsamp.loc[burnin:,'l0000p02spemd'].values   
    LT_reP = LTsamp.loc[burnin:,'l0000p04spemd'].values
    LT_gam = LTsamp.loc[burnin:,'l0000p06spemd'].values
    LT_shear = LTsamp.filter(like='shear',axis=1).iloc[:,-2].values
    nsamp = len(LT_wht)
    
    # prior for rani
    reff = 1.50
    rani_min = 0.5*reff
    rani_max = 5.*reff
    rani_samp = numpy.random.random(nsamp)*(rani_max - rani_min) + rani_min
    
    # compute the cosmological distances to rescale the mlp_thetaE to the appropriate ones for the lens/sr redshift
    zl = 0.2283
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
        Dds[i] = Dist.Da(zl,zs)
        Ds[i] = Dist.Da(zs)
        Dd[i] = Dist.Da(zl)
        Dt[i] = (1+zl)*Dd[i]*Ds[i]/Dds[i]
    
    
    # convert the Einstein radii to the spherical equivalent one, also convert from mlp to single-plane equivalent
    # since "scale:" is used in reP (5th par) of spemd, the scaling of thE now depends on both reP and Dds/Ds to some power
    LT_thE = pow((2./(1.+LT_qP)),(1./(2.*LT_gam))) * numpy.sqrt(LT_qP) * LT_reP
    LT_thE = LT_thE * pow ( (Dds/Ds), (1./(2.*LT_gam)) )
    LT_reP = LT_reP * pow ( (Dds/Ds), (1./(2.*LT_gam)) )
    
    LT_shear = LT_shear * (Dds/Ds)
    
    # concatenate if needed:
    nconcat = 1  ## cannot change this without breaking code for the lens/dyna parameters that are not concatenated fully (to be checked)
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
    
    LT_thE_samp = numpy.tile(LT_thE, nconcat)
    LT_reP_samp = numpy.tile(LT_reP, nconcat)
    LT_qP_samp = numpy.tile(LT_qP, nconcat)
    LT_gam_samp = numpy.tile(LT_gam, nconcat)
    LT_shear_samp = numpy.tile(LT_shear, nconcat)
    
    nsamp = nsamp*nconcat
    
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
    kext_PL_samp = interpolate.splev(numpy.random.random(nsamp),kext_cdf_model)
    
    # factor to convert thE in arcsec and Dist in Mpc into m in M_solar
    mfact = pow(2.9979e8,2)*0.25/6.6738e-11 * 1.e6 * 3.08568e16 * pow((pi/(180.*3600.)),2) / 1.98892e30
    mE_samp = mfact * Dd_samp * Ds_samp / Dds_samp * LT_thE_samp * LT_thE_samp
    
    # to get the likelihood of the dynamics data
    # ==========================================
    # convert slope from lensing model to gamma' (gamma' = 2*gam_clens + 1)
    gammaP_samp = LT_gam_samp * 2. + 1
    h0_samp = H0_samp/100.
    
    # inputs needed to dynamics code:
    aperture = [0.0,0.9,0.0,0.9]
    seeing = 0.9
    vd = numpy.array([296.,303.])
    vderr_stat = numpy.array([19.,24.])
    vderr_sys = 17.
    vdcov = numpy.array([[vderr_stat[0]**2,0],[0,vderr_stat[1]**2]])+vderr_sys**2
    
    mass_samp = (1-kext_PL_samp)*mE_samp # mass of galaxy
    
    # if mass is negative, set to zero
    for i in range(0,nsamp):
        if (mass_samp[i] < 0.):
            print("mass < 0 at i=",i,"  Will set to zero so dyn code doesn't break")
            mass_samp[i] = 0.
    
    # Try to load a pre-made gamma/riso grid
    try:
        model = numpy.load(basedir+'DESJ2038_PL_sigv_grid_v1.dat',allow_pickle=True)
    # otherwise define the grid. 121x121 grid takes
    except:
        gamma = numpy.linspace(1.,3.,121) 
        riso = numpy.logspace(log10(0.5*reff),log10(5*reff),121)
    
        t = time.time()
        model = sm.modelGrids(gamma,riso,aperture,reff,seeing)
        print("Time to create grid: ",time.time()-t)
    
        f = open('DESJ2038_PL_sigv_grid_v1.dat','wb')
        cPickle.dump(model,f,2)
        f.close()
        os.system('mv DESJ2038_PL_sigv_grid_v1.dat '+basedir)
    
        #pylab.imshow(model.z)
        #pylab.show()
    
    # Compute model vd and logp
    vd_PL_samp = sm.getSigmaFromGrid2(mass_samp, LT_thE_samp, gammaP_samp, rani_samp, model,Dd_samp)
    logp = numpy.array([])
    logp = numpy.append(logp,[-0.5*numpy.matmul(numpy.matmul(vd-ii,numpy.linalg.inv(vdcov)),vd[:,numpy.newaxis]-ii) - 0.5*numpy.log(numpy.linalg.det(vdcov)) - numpy.log(2*pi) for ii in vd_PL_samp])


    wht_samp_D = numpy.exp(logp)
    print('Time this loop='+str('{:.3f}'.format(time.time()-looptime))+' s',)
    return wht_samp,H0_samp,om_samp,ol_samp,w_samp,Dd_samp,Dt_samp,LT_reP_samp,LT_thE_samp,gammaP_samp,kext_PL_samp,vd_PL_samp,LT_shear_samp,wht_samp_D

# list input files
infiles_pl = ['esmod_3band_pl_fid']

# PL models
for infile in infiles_pl:
    if infile == '': break
    wht_samp,H0_samp,om_samp,ol_samp,w_samp,Dd_samp,Dt_samp,LT_reP_samp,LT_thE_samp,gammaP_samp,kext_PL_samp,vd_PL_samp,LT_shear_samp,wht_samp_D = ltd_pl_desj2038(infile,burnin=0)
    wht_samp[numpy.isnan(wht_samp)] = 0.
    wht_samp_D[numpy.isnan(wht_samp_D)] = 0.
    
    outfile = basedir + "ImpSamp_LTD_"+infile+".dat"
    fout = open(outfile, 'w')
    fout.write("weight H0 OmegaM OmegaL w Ddmod Dtmod Dtmod/(1-kext) reP reP_sph[reP_for_zl,zs] gammaP kext vd_pred[km/s] shear wht_model wht_dyn\n")
    if (blind == 1):
        fout.write("# Note: cosmo param values are shifted/masked in python imp samp script\n")
    for i in range(0,len(wht_samp)):
        fout.write("%le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le\n"%(wht_samp[i]*wht_samp_D[i], H0_samp[i], om_samp[i], ol_samp[i], w_samp[i], Dd_samp[i], Dt_samp[i], Dt_samp[i]/(1.-kext_PL_samp[i]), LT_reP_samp[i], LT_thE_samp[i], gammaP_samp[i], kext_PL_samp[i], vd_PL_samp[i], LT_shear_samp[i], wht_samp[i], wht_samp_D[i]))
    fout.close()
    print(outfile+'\n')

print('Total runtime: '+str('{:.1f}'.format(time.time()-starttime))+' s')