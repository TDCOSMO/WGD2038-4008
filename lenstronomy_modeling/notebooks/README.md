# Lenstronomy modeling notebooks

This directory contains several notebooks used for modeling WGD2038-4008
 with the software lenstronomy.
  
Some of these notebooks use the following codes for plotting purposes:
- [paperfig](https://github.com/ajshajib/paperfig)
- [Coloripy](https://github.com/ajshajib/coloripy)

This is the list of notebooks in the general order of workflow, and grouped
 accordingly.

**Modeling:**
- These notebooks creates cutouts for modeling and makes initial PSF
 estimates:
    - `Image preprocessing F160W.ipynb`
    - `Image preprocessing F814W.ipynb`
    - `Image preprocessing F475X.ipynb`
- `2038 Multiband Image Modeling.ipynb` setups the model with multiple
 settings. These models are sent to a cluster for optimization. Test models
  can be run with shorter optimization routines, and the model outputs (both
   from test runs and full cluster runs) can be visually inspected.
- `Convert Sersic to Chameleon for initial parameters.ipynb` provides the
 initial Chameleon parameters for composite models from an optimized triple-Sersic profile int
  the F160W band. 
- `Fit deflector light using photutils.ipynb` obtains the MGE of the
 deflector's light to use in the kinematic computation, and to obtain the IR photometry.
- `Fit deflector light using photutils UVIS.ipynb` obtains the UVIS bands' photometry for stellar mass estimation.
- `Estimate stellar mass.ipynb` estimates the deflector's stellar mass from 3-band HST photometry using stellar population synthesis method.
  
**Inspecting model qualities and making figures:**
- `Make model decomposition plots.ipynb` creates the lenstronomy model
 summary figures for the paper.
- `Check models and MCMC chain convergence.ipynb` plots model summaries and MCMC trace
 plots for multiple model runs to visually inspect them. 
- `Shorten MCMC chains.ipynb` thins MCMC chains for all the models for
 kinematic compuation with a tractable number of samples from the lens model
  posterior.
- `Fermat potentials and lens model comparisons.ipynb` computes the Fermat
 potential differences for the lens models, combined them using BIC
  weighting, and compare between model settings.
- `Check NFW properties.ipynb` checks the M-c relation for the NFW halos
 in the composite models.
  
**Post-processing of the models to add kinematics, external convergence:**
- The kinematics is computed on cluster using the `output_class.ModelOutput
.compute_model_velocity_dispersion()` method. This module is provided in the
 "process_output" folder. The other files in that folder are used to
  distribute the computing job over multiple cores in the cluster. 
- `Distance ratio prior.ipynb` obtains the D_s/D_ds distribution from the Pantheon dataset.
- `Combine kinematics and external convergence.ipynb` combines the computed
 velocity dispersion with the lens models to provide final Fermat potential
  differences, or the time-delay predictions.
