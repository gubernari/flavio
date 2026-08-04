from functools import lru_cache
from math import sqrt
import pkgutil

import flavio
import numpy as np
import yaml

from flavio.classes import Parameter
from flavio.config import config
from flavio.physics.bdecays.formfactors.b_p.isgurwise import isgur_wise
from flavio.statistics.probability import MultivariateNormalDistribution


PROCESS = 'Bs->K'

# The FLAG 2024 B_s -> K BCL parametrization uses a non-standard branch point:
# t_+ = (m_B + m_pi)^2 rather than (m_Bs + m_K)^2.
M_B_FLAG = 5.27931
M_PI_FLAG = 0.1349766
M_BS_FLAG = 5.36688
M_K_FLAG = 0.493677


def pole(mres, q2):
    if mres == 0 or mres is None:
        return 1
    return 1 / (1 - q2 / mres**2)


@lru_cache(maxsize=config['settings']['cache size'])
def z_flag(q2, t0=None):
    tm = (M_BS_FLAG - M_K_FLAG) ** 2
    tp = (M_B_FLAG + M_PI_FLAG) ** 2
    if t0 is None:
        t0 = tp - sqrt(tp * (tp - tm))
    elif t0 == 'tm':
        t0 = tm
    sq2 = sqrt(tp - q2)
    st0 = sqrt(tp - t0)
    return (sq2 - st0) / (sq2 + st0)


def param_fplus(a_i, q2, t0=None):
    z = z_flag(q2, t0=t0)
    n = len(a_i)
    k = np.arange(n)
    return (a_i * (z**k - (-1.0) ** (k - n) * k / n * z**n)).sum()


def param_f0(a_i, q2, t0=None):
    z = z_flag(q2, t0=t0)
    k = np.arange(len(a_i))
    return (a_i * z**k).sum()


def ff(process, q2, par, t0=None):
    r"""B_s->K form factors using the FLAG 2024 BCL coefficients.

    This implementation follows arXiv:2411.04268 and therefore uses the
    process-specific definition of the expansion parameter z with
    t_+ = (m_B + m_pi)^2.
    """
    if process != PROCESS:
        raise ValueError(f"{PROCESS} is the only process supported by this implementation.")
    flavio.citations.register("Bourrely:2008za")
    mpl = par[PROCESS + ' BCL m+']
    m0 = par[PROCESS + ' BCL m0']
    a_plus = np.array([par[f'{PROCESS} BCL a{i}_f+'] for i in range(4)])
    a0_free = [par[f'{PROCESS} BCL a{i}_f0'] for i in range(3)]
    fplus_q20 = pole(mpl, 0) * param_fplus(a_plus, 0, t0=t0)
    f0_q20 = pole(m0, 0) * param_f0(a0_free, 0, t0=t0)
    a3_f0 = (fplus_q20 - f0_q20) / z_flag(0, t0=t0) ** 3
    a0 = np.array(a0_free + [a3_f0])
    return {
        'f+': pole(mpl, q2) * param_fplus(a_plus, q2, t0=t0),
        'f0': pole(m0, q2) * param_f0(a0, q2, t0=t0),
    }


def ff_isgurwise(process, q2, par, scale, t0=None):
    r"""B_s->K form factors with f_T from the improved Isgur-Wise relation."""
    _ff = ff(process, q2, par, t0=t0)
    return isgur_wise(process, q2, _ff, par, scale=scale)


def _set_parameter_metadata(parameter_name):
    p = Parameter[parameter_name]
    if parameter_name == PROCESS + ' IW a_T':
        p.tex = r'$a_T^{B_s\to K}$'
        p.description = r'Relative power correction to the $B_s\to K$ tensor form factor from the improved Isgur-Wise relation'
        return
    coeff = parameter_name.split(' BCL ')[1]
    a_name, ff_name = coeff.split('_')
    tex_ff = {'f+': r'f_+', 'f0': r'f_0'}[ff_name]
    tex_a = a_name.replace('a', r'a_')
    p.tex = rf'${tex_a}^{{({tex_ff[2:]})}}$'
    p.description = rf'BCL $B_s\to K$ form factor parametrization coefficient ${tex_a}$ of ${tex_ff}$'


def load_parameters(filename, constraints):
    f = pkgutil.get_data('flavio.physics', filename)
    ff_dict = yaml.safe_load(f)
    for parameter_name in ff_dict['parameters'] + [PROCESS + ' IW a_T']:
        try:
            Parameter[parameter_name]
        except KeyError:
            Parameter(parameter_name)
        _set_parameter_metadata(parameter_name)
    constraints.set_constraint(PROCESS + ' BCL m0', ff_dict['m0'])
    constraints.set_constraint(PROCESS + ' BCL m+', ff_dict['m+'])
    constraints.set_constraint(PROCESS + ' IW a_T', ff_dict['iw_aT'])
    covariance = np.outer(ff_dict['uncertainties'], ff_dict['uncertainties']) * np.asarray(ff_dict['correlation'])
    constraints.add_constraint(
        ff_dict['parameters'],
        MultivariateNormalDistribution(central_value=ff_dict['central_values'], covariance=covariance),
        is_parameter_constraint=True,
    )
