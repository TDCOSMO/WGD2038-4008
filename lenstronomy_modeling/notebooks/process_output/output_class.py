import numpy as np
import os
from tqdm import tqdm_notebook, tnrange
import joblib
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.Sampling.parameters import Param
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Analysis.td_cosmography import TDCosmography


cwd = os.getcwd()
base_path, _ = os.path.split(cwd)


class ModelOutput(object):
    """
    Class to load lens model posterior chain and other model setups for WGD
    2038-4008.
    """

    TIME_DELAYS = np.array([])
    SIGMA_DELAYS = np.array([])

    # measured velocity dispersion from Buckley-Geer et al. (2020)
    VEL_DIS = np.array([296.])
    SIG_VEL_DIS = np.array([19.])
    PSF_FWHM = np.array([0.9])
    MOFFAT_BETA = np.array([1.74])
    SLIT_WIDTH = np.array([0.75])
    APERTURE_LENGTH = np.array([1.])

    Z_L = 0.230 # deflector redshift from Agnello et al. (2018)
    Z_S = 0.777 # source redshift

    def __init__(self, model_name, model_type, dir_prefix,
                 dir_suffix, is_test=False, microlensing=False):
        """
        Load the model output file and load the posterior chain and other model
        speification objects.
        """
        self.model_id = model_name
        self.model_type = model_type
        self.is_test = is_test
        self.file_path = dir_prefix + model_name + dir_suffix

        f = open(self.file_path, 'rb')
        [input_, output_] = joblib.load(f)
        f.close()
        fitting_kwargs_list, kwargs_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, init_samples = input_
        kwargs_result, multi_band_list_out, fit_output, _ = output_

        self.kwargs_joint = kwargs_joint
        self.kwargs_likelihood = kwargs_likelihood

        self.kwargs_result = kwargs_result
        self.multi_band_list = multi_band_list_out

        self.samples_mcmc = fit_output[-1][1]
        self.chain_likelihoods = fit_output[-1][-1]
        # random.shuffle()
        self.num_param_mcmc = len(self.samples_mcmc)
        self.param_mcmc = fit_output[-1][2]

        self.kwargs_model = kwargs_model
        # if self.is_test:
        #    self.kwargs_model['cosmo'] = None
        self.kwargs_constraints = kwargs_constraints

        # if 'special' in kwargs_params:
        #     special = kwargs_params['special'][2]
        # else:
        #     special = []
        #
        # if 'extinction_model' in kwargs_params:
        #     extinction_model = kwargs_params['extinction_model'][2]
        # else:
        #     extinction_model = []

        self.param_class = Param(kwargs_model,
                                 kwargs_params['lens_model'][2],
                                 kwargs_params['source_model'][2],
                                 kwargs_params['lens_light_model'][2],
                                 kwargs_params['point_source_model'][2],
                                 #kwargs_params['special'][2],
                                 #kwargs_params['extinction_model'][2],
                                 kwargs_lens_init=kwargs_params['lens_model'][
                                     0],
                                 **kwargs_constraints
                                 )

        kwargs_result = self.param_class.args2kwargs(self.samples_mcmc[-1])

        self.lens_result = kwargs_result['kwargs_lens']
        self.lens_light_result = kwargs_result['kwargs_lens_light']
        self.ps_result = kwargs_result['kwargs_ps']
        self.source_result = kwargs_result['kwargs_source']

        self.r_eff = [] #self.get_r_eff()
        self.a_ani = []

        # declare following variables to populate later
        self.model_time_delays = None
        self.model_velocity_dispersion = None

        # numerical options to perform the numerical integrals
        self.kwargs_galkin_numerics = {#'sampling_number': 1000,
                                       'interpol_grid_num': 1000,
                                       'log_integration': True,
                                       'max_integrate': 100,
                                       'min_integrate': 0.001}

        self.td_cosmography = TDCosmography(z_lens=self.Z_L, z_source=self.Z_S,
                                            kwargs_model=self.kwargs_model)

        self.bic = self.get_bic_value()

    def get_bic_value(self):
        """
        Compute BIC value for this model.
        """
        num_data = np.sum([np.sum(m) for m in
                           self.kwargs_likelihood['image_likelihood_mask_list']])
        num_param = self.param_class.num_param()[0] + \
                    self.param_class.num_param_linear()
        max_logL = np.max(self.chain_likelihoods)

        return compute_BIC(num_data, num_param, max_logL)

    def get_num_samples(self):
        """
        Get the number of samples.
        :return:
        :rtype:
        """
        return len(self.samples_mcmc)

    def get_r_eff(self, i=-1):
        """
        Compute effective radius of the light distribution in F160W band.
        """
        # use 3.2 arcsec computed from isophote fitted model, instead of using
        # the lens model
        raise NotImplementedError

        if i == -1:
            kwargs_result = self.kwargs_result
        else:
            kwargs_result = self.param_class.args2kwargs(self.samples_mcmc[i])

        self._imageModel = class_creator.create_im_sim(self.multi_band_list,
                                                       self.kwargs_joint[
                                                           'multi_band_type'],
                                                       self.kwargs_model,
                                                       bands_compute=
                                                       self.kwargs_likelihood[
                                                           'bands_compute'],
                                                       likelihood_mask_list=
                                                       self.kwargs_likelihood[
                                                           'image_likelihood_mask_list'],
                                                       band_index=0)

        model, error_map, cov_param, param = self._imageModel.image_linear_solve(
            inv_bool=True,
            **kwargs_result)

        lens_light_bool_list = [False] * len(
            self.kwargs_model['lens_light_model_list'])

        if self.is_test:
            lens_light_bool_list[0] = True
        else:
            if self.model_type == "powerlaw":
                # indices: 6, 7, 8
                lens_light_bool_list[6] = True  # F160W Sersic profile
                lens_light_bool_list[7] = True
                lens_light_bool_list[8] = True
            elif self.model_type == "composite":
                lens_light_bool_list[6] = True  # Chameleon profile index: 6
            else:
                raise NotImplementedError

        r_eff = self.lens_analysis.half_light_radius_lens(
            kwargs_result['kwargs_lens_light'],
            center_x=self.lens_light_result[0]['center_x'],
            center_y=self.lens_light_result[0]['center_x'],
            model_bool_list=lens_light_bool_list,
            deltaPix=0.01, numPix=1000)
        return r_eff

    def compute_model_time_delays(self):
        """
        Compute time delays from the lens model and store it in class variable `model_time_delays`.
        """
        num_samples = self.get_num_samples()

        self.model_time_delays = []

        for i in tnrange(num_samples,
                         desc="{} model delays".format(self.model_id)):
            param_array = self.samples_mcmc[i]

            kwargs_result = self.param_class.args2kwargs(param_array)

            model_arrival_times = self.td_cosmography.time_delays(
                    kwargs_result['kwargs_lens'],
                    kwargs_result['kwargs_ps'],
                    original_ps_position=True
            )
            # print(model_arrival_times)
            dt_AB = model_arrival_times[1] - model_arrival_times[3]
            dt_AC = model_arrival_times[1] - model_arrival_times[2]
            dt_AD = model_arrival_times[1] - model_arrival_times[0]

            self.model_time_delays.append([dt_AB, dt_AC, dt_AD])

        self.model_time_delays = np.array(self.model_time_delays)

    def compute_model_velocity_dispersion(self,
                                          cGD_light=True, cGD_mass=True,
                                          a_ani_min=0.5,
                                          a_ani_max=5,
                                          start_index=0,
                                          num_compute=None,
                                          print_step=None,
                                          r_eff_uncertainty=0.02
                                          ):
        """
        Compute velocity dispersion from the lens model for different measurement setups.
        :param num_samples: default `None` to compute for all models in the
        chain, use lower number only for testing and keep it same between
        `compute_model_time_delays` and this method.
        :param start_index: compute velocity dispersion from this index
        :param num_compute: compute for this many samples
        :param print_step: print a notification after this many step
        """
        num_samples = self.get_num_samples()

        self.model_velocity_dispersion = []

        anisotropy_model = 'OM'  # anisotropy model applied
        aperture_type = 'slit'  # type of aperture used

        if num_compute is None:
            num_compute = num_samples - start_index

        n = 0

        # source_result = kwargs_result['kwargs_source']
        # ps_result = kwargs_result['kwargs_ps']

        if self.is_test:
            band_index = 0
        else:
            band_index = 2

        kwargs_aperture = {'aperture_type': aperture_type,
                           'length': self.APERTURE_LENGTH[n],
                           'width': self.SLIT_WIDTH[n],
                           'center_ra': 0., #lens_light_result[0]['center_x'],
                           'center_dec': 0., #lens_light_result[0]['center_y'],
                           'angle': 0
                           }

        kwargs_seeing = {'psf_type': 'MOFFAT',
                         'fwhm': self.PSF_FWHM[n],
                         'moffat_beta': self.MOFFAT_BETA[n]}

        light_model_bool = [False] * len(
            self.kwargs_model['lens_light_model_list'])

        if self.is_test:
            light_model_bool[0] = True
        else:
            if self.model_type == "powerlaw":
                # indices: 8, 9, 10
                light_model_bool[6] = True  # F160W Sersic profile
                light_model_bool[7] = True
                light_model_bool[8] = True
            elif self.model_type == "composite":
                light_model_bool[6] = True  # Chameleon profile F160W index: 6
            else:
                raise NotImplementedError

        self.kwargs_model['lens_light_model_list'] = ['MULTI_GAUSSIAN',
                                                      'MULTI_GAUSSIAN',
                                                      'MULTI_GAUSSIAN']
        light_model_bool = [False, False, True]

        lens_model_bool = [False] * len(
            self.kwargs_model['lens_model_list'])

        if self.model_type == 'powerlaw':  # self.kwargs_model['lens_model_list'][0] == 'SPEMD':
            lens_model_bool[0] = True

            cGD_light = False
            cGD_mass = False
        else:
            lens_model_bool[0] = True
            lens_model_bool[2] = True

            cGD_light = False
            cGD_mass = True

        kinematics_api = KinematicsAPI(z_lens=self.Z_L, z_source=self.Z_S,
                                       kwargs_model=self.kwargs_model,
                                       kwargs_aperture=kwargs_aperture,
                                       kwargs_seeing=kwargs_seeing,
                                       anisotropy_model=anisotropy_model,
                                       cosmo=None,
                                       lens_model_kinematics_bool=lens_model_bool,
                                       light_model_kinematics_bool=light_model_bool,
                                       multi_observations=False,
                                       kwargs_numerics_galkin=self.kwargs_galkin_numerics,
                                       analytic_kinematics=False,
                                       Hernquist_approx=False,
                                       MGE_light=cGD_light,
                                       MGE_mass=cGD_mass,
                                       kwargs_mge_light=None,
                                       kwargs_mge_mass=None,
                                       sampling_number=1000,
                                       num_kin_sampling=2000,
                                       num_psf_sampling=500)

        for i in range(start_index, start_index+num_compute):
            if print_step is not None:
                if (i-start_index)%print_step == 0:
                    print('Computing step: {}'.format(i-start_index))

            sample = self.samples_mcmc[i]

            #vel_dis_array = []

            r_eff_uncertainty_factor = np.random.normal(loc=1., scale=0.02)

            if self.is_test:
                a_ani = 2
            else:
                r_eff = 3.2 * r_eff_uncertainty_factor #self.get_r_eff(i)
                # Jeffrey's prior for a_ani
                a_ani = 10**np.random.uniform(np.log10(a_ani_min),
                                              np.log10(a_ani_max))

            self.a_ani.append(a_ani)
            self.r_eff.append(r_eff)

            kwargs_result = self.param_class.args2kwargs(sample)

            # set the anisotropy radius. r_eff is pre-computed half-light
            # radius of the lens light
            kwargs_anisotropy = {'r_ani': a_ani * r_eff}

            kwargs_gaussian_light = [{
                'center_x': self.lens_light_result[0]['center_x'],
                'center_y': self.lens_light_result[0]['center_y'],
                # these values are from the isophote model -> MGE fit in F160W
                'amp': np.array([108.69100868,  799.04561399, 1536.81696056,
                                 2510.46534315, 2425.14639225, 2188.20896608,
                                 3075.68274026, 1808.28311979]),
                'sigma': np. array([0.10700896,  0.24475083,  0.52492347,
                                    1.03770399,  2.24402017, 3.74713874,
                                    8.58773362,
                                    10.27207989]) * r_eff_uncertainty_factor
            }] * 3

            # compute the velocity disperson in a pre-specified cosmology
            # (see lenstronomy function)
            vel_dis = kinematics_api.velocity_dispersion(
                kwargs_result['kwargs_lens'],
                kwargs_gaussian_light,
                #kwargs_result['kwargs_lens_light'],
                kwargs_anisotropy
            )

            self.model_velocity_dispersion.append(vel_dis)

        self.model_velocity_dispersion = np.array(
            self.model_velocity_dispersion)

        return self.model_velocity_dispersion

    def load_velocity_dispersion(self, model_name, dir_prefix, dir_suffix,
                                 compute_chunk, total_samples=None):
        """
        Load saved model velocity dispersions.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :param compute_chunk: number of samples in a computed chunk to
        combine output files with computed velocity dispersions
        :type compute_chunk: int
        :return:
        :rtype:
        """
        loaded_vel_dis = []
        if total_samples is None:
            total_samples = self.get_num_samples()

        for i in range(int(total_samples / compute_chunk)):
            start_index = i * compute_chunk
            file_path = dir_prefix + '{}_'.format(start_index) + \
                        model_name + dir_suffix

            if loaded_vel_dis == []:
                loaded_vel_dis = np.loadtxt(file_path)
            else:
                loaded_vel_dis = np.append(loaded_vel_dis,
                                           np.loadtxt(file_path),
                                           axis=0)

        assert len(loaded_vel_dis) == self.get_num_samples()

        self.model_velocity_dispersion = loaded_vel_dis

    def load_a_ani(self, model_name, dir_prefix, dir_suffix,
                   compute_chunk):
        """
        Load saved r_ani from files.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :param compute_chunk: number of samples in a computed chunk to
        combine output files with computed a_ani
        :type compute_chunk: int
        :return:
        :rtype:
        """
        loaded_a_ani = []

        for i in range(int(self.get_num_samples() / compute_chunk)):
            start_index = i * compute_chunk
            file_path = dir_prefix + '{}_'.format(start_index) + \
                        model_name + dir_suffix

            if loaded_a_ani == []:
                loaded_a_ani = np.loadtxt(file_path)
            else:
                loaded_a_ani = np.append(loaded_a_ani, np.loadtxt(file_path),
                                         axis=0)

        assert len(loaded_a_ani) == self.get_num_samples()

        self.a_ani = loaded_a_ani

    def load_r_eff(self, model_name, dir_prefix, dir_suffix,
                   compute_chunk):
        """
        Load saved r_eff from files.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix: should include the slash at the end
        :param dir_suffix: example '_mod_out.txt'
        :type dir_suffix:
        :param compute_chunk: number of samples in a computed chunk to
        combine output files with R_eff
        :type compute_chunk: int
        :return:
        :rtype:
        """
        loaded_r_eff = []

        for i in range(int(self.get_num_samples() / compute_chunk)):
            start_index = i * compute_chunk
            file_path = dir_prefix + '{}_'.format(start_index) + \
                        model_name + dir_suffix

            if loaded_r_eff == []:
                loaded_r_eff = np.loadtxt(file_path)
            else:
                loaded_r_eff = np.append(loaded_r_eff, np.loadtxt(file_path),
                                         axis=0)

        assert len(loaded_r_eff) == self.get_num_samples()

        self.r_eff = loaded_r_eff

    def save_time_delays(self, model_name, dir_prefix, dir_suffix):
        """
        Save computed time delays.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix:
        :param dir_suffix:
        :type dir_suffix:
        :return:
        :rtype:
        """
        file_path = dir_prefix + model_name + dir_suffix
        np.savetxt(file_path, self.model_time_delays)

    def load_time_delays(self, model_name, dir_prefix, dir_suffix):
        """
        Load saved time delays.
        :param model_name:
        :type model_name:
        :param dir_prefix:
        :type dir_prefix:
        :param dir_suffix:
        :type dir_suffix:
        :return:
        :rtype:
        """
        file_path = dir_prefix + model_name + dir_suffix

        loaded_time_delays = np.loadtxt(file_path)

        assert len(loaded_time_delays) == self.get_num_samples()

        self.model_time_delays = loaded_time_delays

    def get_Ds_Dds(self, cosmo=None, blind=True):
        """
        """
        if not blind:
            if cosmo is None:
                cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

            D_s_fiducial = cosmo.angular_diameter_distance(self.Z_S)
            D_ds_fiducial = cosmo.angular_diameter_distance_z1z2(self.Z_L,
                                                                 self.Z_S)

            Ds_Dds_fiducial = D_s_fiducial / D_ds_fiducial
        else:
            Ds_Dds_fiducial = 1.

        sampled_observed_vel_dis = np.random.normal(
            loc=self.VEL_DIS[0],
            scale=self.SIG_VEL_DIS[0],
            size=len(self.model_velocity_dispersion)
            )

        return Ds_Dds_fiducial * sampled_observed_vel_dis**2 \
                                    / self.model_velocity_dispersion**2


from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.lens_model import LensModel

lens_cosmo = LensCosmo(z_lens=0.230, z_source=0.777)
lens_analysis = LensProfileAnalysis(
    LensModel(lens_model_list=['NFW_ELLIPSE', 'SHEAR', 'TRIPLE_CHAMELEON'],
              z_lens=0.230, z_source=0.777,
              multi_plane=False,  # True,
              ))


def custom_loglikelihood_addition(kwargs_lens=None, kwargs_source=None,
                                  kwargs_lens_light=None, kwargs_ps=None,
                                  kwargs_special=None, kwargs_extinction=None):
    """
    Impose a Gaussian prior on the NFW scale radius R_s based on Gavazzi et al. (2007).
    """
    # imports inside function to avoid pickling
    from colossus.halo import concentration
    from colossus.halo import mass_defs
    from colossus.cosmology import cosmology

    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
    from lenstronomy.LensModel.lens_model import LensModel

    lens_cosmo = LensCosmo(z_lens=0.230, z_source=0.777)
    lens_analysis = LensProfileAnalysis(
        LensModel(lens_model_list=['NFW_ELLIPSE', 'SHEAR', 'TRIPLE_CHAMELEON'],
                  z_lens=0.230, z_source=0.777,
                  multi_plane=False,  # True,
                  ))

    if kwargs_lens[0]['alpha_Rs'] < 0.:
        return -np.inf

    if not -0.014271818911080656 - 0.2 < kwargs_lens[0][
        'center_x'] < -0.014271818911080656 + 0.2:
        return -np.inf
    if not -0.020882886550870693 - 0.2 < kwargs_lens[0][
        'center_y'] < -0.020882886550870693 + 0.2:
        return -np.inf

    if not -0.5 < kwargs_lens[0]['e1'] < 0.5:
        return -np.inf
    if not -0.5 < kwargs_lens[0]['e2'] < 0.5:
        return -np.inf

    log_L = 0.

    # integrate upto 3.2 arcsec, which is half-light radius (~half-mass radius)
    mean_convergence = lens_analysis.mass_fraction_within_radius(kwargs_lens,
                                                                 kwargs_lens[
                                                                     2][
                                                                     'center_x'],
                                                                 kwargs_lens[
                                                                     2][
                                                                     'center_y'],
                                                                 3.2,
                                                                 numPix=320)

    stellar_mass = np.log10(
        mean_convergence[2] * np.pi * (3.2 / 206265 * lens_cosmo.dd) ** 2
        * lens_cosmo.sigma_crit * 2)  # multiplying by 2 to convert half-mass to full mass

    # log_L += - 0.5 * (stellar_mass - 11.40)**2 / (0.08**2 + 0.1**2)
    # adding 0.07 uncertainty in quadrature to account for 15% uncertainty in H_0, Om_0 ~ U(0.05, 0.5)
    high_sm = 11.57 + 0.25 + 0.06  # +0.06 is to add H_0 uncertainty
    low_sm = 11.57 - 0.06  # -0.06 is to add H_0 uncertainty
    if stellar_mass > high_sm:
        log_L += -0.5 * (high_sm - stellar_mass) ** 2 / (0.16 ** 2)
    elif stellar_mass < low_sm:
        log_L += -0.5 * (low_sm - stellar_mass) ** 2 / (0.13 ** 2)
    else:
        log_L += 0.

    _, _, c, r, halo_mass = lens_cosmo.nfw_angle2physical(kwargs_lens[0]['Rs'],
                                                          kwargs_lens[0][
                                                              'alpha_Rs'])
    log_L += -0.5 * (np.log10(halo_mass) - 13.5) ** 2 / 0.3 ** 2

    my_cosmo = {'flat': True, 'H0': 70., 'Om0': 0.3, 'Ob0': 0.05,
                'sigma8': 0.823, 'ns': 0.96}  # fiducial cosmo
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)

    c200 = concentration.concentration(halo_mass * cosmo.h, '200c',
                                       # input halo mass needs to be in M_sun/h unit
                                       0.23, model='diemer19')

    log_L += -0.5 * (np.log10(c) - np.log10(c200)) ** 2 / (0.11 ** 2)

    return log_L


def compute_BIC(num_data, num_model, max_logL):
    """
    Bayesian information criterion
    """
    return np.log(num_data) * num_model - 2 * max_logL


def get_relative_weights(bics, sigma_numeric, neighbor_indices=None,
                         take_std=False, sigma_model=None):
    """
    Get relative weights for model chains given their BIC values.

    :param logZs:
    :type logZs:
    :param sigma_numeric:
    :type sigma_numeric:
    :param neighbor_indices:
    :type neighbor_indices:
    :return:
    :rtype:
    """

    if sigma_model is not None:
        pass
    elif take_std:
        sigma_model = np.std(bics)
    elif neighbor_indices is not None:
        delta_bic_neighbors = []
        for ni in neighbor_indices:
            delta_bic_neighbors.append(abs(bics[ni[0] - 1] - bics[ni[1] - 1]))

        sigma_model = np.std(delta_bic_neighbors)
    else:
        raise ValueError("Don't know what to do with sigma_model")

    # print("sigma_mmodel:", sigma_model)
    min_bic = np.min(bics)
    # print(max_logZ)
    del_bics = bics - min_bic

    sigma = np.sqrt(sigma_model ** 2 + sigma_numeric ** 2)

    effective_nums = []

    for del_bic in del_bics:
        distribution = np.random.normal(loc=-del_bic / 2., scale=sigma,
                                        size=1000000)
        effective_num = float(len(distribution[distribution >= 0.]))
        effective_num += np.sum(np.exp(distribution[distribution < 0.]))
        effective_nums.append(effective_num / 1000000.)

    return effective_nums / np.max(effective_nums)