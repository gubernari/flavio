r"""Functions for $e^+ e^-\to l^+ l^- of various flavours"""
# Written by Ben Allanach

from math import pi, sqrt
from flavio.physics.zdecays.smeftew import gV_SM, gA_SM, _QN
from flavio.physics.common import add_dict
from flavio.classes import Observable, Prediction
from flavio.physics import ckm as ckm_flavio
import numpy as np
import flavio.physics.zdecays.smeftew as smeftew
import flavio

# Switch on to output SM computations for checks
TEST_SM = False

# Kronecker delta
def delta(a, b):
    return int(a == b)

# Calculates helicity amplitudes for X, Y in 1,2 meaning L, R: from arXiv:2306.08669 (4.2)
def calc_Ns(C, par, s, fam, l_type):
    n    = np.empty((2, 2), dtype=np.complex_)
    n_sm = np.empty((2, 2), dtype=np.complex_)
    mz = par['m_Z']
    gammaZ = 1 / par['tau_Z']
    GF = par['GF']
    alpha = par['alpha_e']
    if (TEST_SM):
        alpha = 1 / 132.507 # Use madgraph default values for check
        GF    = 1.66390e-05
        MZ    = 91.188
        gammaZ = 2.441404e+00
    s2w   = par['s2w']
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * pi * alpha
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    g_cw  = sqrt(gLsq + gYsq)
    # Need to define complex matrix element first, then take mod squared and multiply by the form factor of each piece.
    if (fam != 2 and fam != 3):
        raise ValueError(f'ee_ll called with incorrect family {fam} - should be 2 or 3')
    # WARNING: flavio's gAe and gAei have opposite sign to normal gA 
    gVe  = g_cw * gV_SM('e', par) 
    gAe  = g_cw * gA_SM('e', par)
    gVei = g_cw * gV_SM(l_type, par)
    gAei = g_cw * gA_SM(l_type, par)
    # My ordering for X and Y is (1, 2):=(L, R). Bug-fixed the range - previously it was only calculating for L L combination: 29/4/25
    for X in range(1, 3):
        for Y in range(1 ,3):
            # Changed the relative sign of gAe to agree with ppll from flavio on 29/4/25: bug-fixed
            # 25/5/25: bug-fixed SMEFT Z couplings gex and gey - they were missing a factor of g_cw before. Thanks to E Loisa
            gexSM = (gVe - gAe) * delta(X, 2) + (gVe + gAe) * delta(X, 1)
            geySM = (gVei - gAei) * delta(Y, 2) + (gVei + gAei) * delta(Y, 1)
            gex   = gexSM + (smeftew.d_gVl('e', 'e', par, C) - smeftew.d_gAl('e', 'e', par, C)) * g_cw * delta(X, 2) + (smeftew.d_gVl('e', 'e', par, C) + smeftew.d_gAl('e', 'e', par, C)) * g_cw * delta(X, 1)
            gey   = geySM + (smeftew.d_gVl(l_type, l_type, par, C) - smeftew.d_gAl(l_type, l_type, par, C)) * g_cw * delta(Y, 2) + (smeftew.d_gVl(l_type, l_type, par, C) + smeftew.d_gAl(l_type, l_type, par, C)) * g_cw * delta(Y, 1)
            n_sm[X-1, Y-1] = eSq / s + gexSM * geySM / (s - mz**2 + 1j * mz * gammaZ)
            n[X-1, Y-1]    = eSq / s + gex * gey     / (s - mz**2 + 1j * mz * gammaZ) + (C[f'll_11{fam}{fam}'] + C[f'll_1{fam}{fam}1']) * delta(X, 1) * delta(Y, 1) + C[f'ee_11{fam}{fam}'] * delta(X, 2) * delta(Y, 2) + C[f'le_11{fam}{fam}'] * delta(X, 1) * delta(Y, 2) + C[f'le_{fam}{fam}11'] * delta(X, 2) * delta(Y, 1)
    return n_sm, n
    
# predicted ratio to SM with SMEFT operators of e+e-->mu+mu- or tau+tau- for LEP2 energy E and family fam=2,3 total cross-section. Programmed by BCA 22/2/23, checked and corrected 27/2/23
def ee_ll(C, par, E, fam):
    # Check energy E is correct
    if (E != 130.3 and E != 136.3 and E != 161.3 and E != 172.1 and E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and E != 199.5 and E!= 201.8 and E!= 204.8 and E!= 206.5):
        raise ValueError('ee_ll called with incorrect LEP2 energy {} GeV.'.format(E))
    s = E * E
    l_type = 'none'
    if (fam == 2):
        l_type = 'mu'
    elif (fam == 3):
        l_type = 'tau'
    NSM, N = calc_Ns(C, par, s, fam, l_type)
    # Need to define complex matrix element first, then take mod squared and multiply by the form factor of each piece.
    sigma_tot_NP = 0; sigma_tot_SM = 0
    CONV_FACTOR = 2.569e-9 # To get from GeV^-2 to pb.
    for X in range(1, 3):
        for Y in range(1 ,3):
            sigma_tot_NP += abs(N[X-1, Y-1])**2
            sigma_tot_SM += abs(NSM[X-1, Y-1])**2
    # This next contribution has no helicity structure in SM and therefore doesn't interfere with it
    sigma_tot_NP += 3 * abs(C[f'le_{fam}{fam}11'])**2
    if (TEST_SM and l_type == 'mu'):
        print(str(E) + " " + str(sigma_tot_SM * s / (48 * np.pi * CONV_FACTOR)) + "  # process=e+ e- > " + l_type + "+ " + l_type + "-")
    return (sigma_tot_NP / sigma_tot_SM)


def ee_ll_obs(wc_obj, par, E, fam):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ll(C, par, E, fam)

_process_tex = r"e^+e^- \to l^+l^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to l^+l^-$ :: $' + _process_tex + r"$"

_obs_name = "R_sigma(ee->ll)"
_obs = Observable(_obs_name)
_obs.arguments = ['E', 'fam']
Prediction(_obs_name, ee_ll_obs)
_obs.set_description(r"Ratio of cross section of $" + _process_tex + r"$ at energy $E$ to that of the SM")
_obs.tex = r"$R_\sigma(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)

# predicted ratio of AFB to SM from SMEFT operators of e+e-->mumu/tautau for LEP2 energy E and family fam total cross-section. AfbSM is the SM prediction for AFB. fam=2 or 3 is the lepton family. Programmed by BCA 27/2/23
def ee_ll_afb(C, par, E, fam):
    # Check energy E is correct
    if (E != 130.3 and E != 136.3 and E != 161.3 and E != 172.1 and E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and E != 199.5 and E!= 201.8 and E!= 204.8 and E!= 206.5):
        raise ValueError('ee_ll called with incorrect LEP2 energy {} GeV.'.format(E))
    s = E * E
    l_type = 'none'
    if (fam == 2):
        l_type = 'mu'
    elif (fam == 3):
        l_type = 'tau'
    NSM, N = calc_Ns(C, par, s, fam, l_type)
    # Need to define complex matrix element first, then take mod squared and multiply by the form factor of each piece.
    afb_tot    = 0;    afb_tot_SM = 0
    CONV_FACTOR = 2.569e-9 # To get from GeV^-2 to pb.
    for X in range(1, 3):
        for Y in range(1 ,3):
            afb_tot += s / (64 * pi * CONV_FACTOR) * abs(N[X-1, Y-1])**2 * (2 * delta(X, Y) - 1)
            afb_tot_SM += s / (64 * pi * CONV_FACTOR) * abs(NSM[X-1, Y-1])**2 * (2 * delta(X, Y) - 1)
    if (TEST_SM and l_type == 'mu'):
        print(str(E) + " " + str(afb_tot_SM) + "  # AFB e+ e- > " + l_type + "+ " + l_type + "-")
    return afb_tot / afb_tot_SM

def ee_ll_afb_obs(wc_obj, par, E, fam):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ll_afb(C, par, E, fam)

_process_tex = r"e^+e^- \to l^+l^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to l^+l^-$ :: $' + _process_tex + r"$"

_obs_name = "R_Afb(ee->ll)"
_obs = Observable(_obs_name)
_obs.arguments = ['E', 'fam']
Prediction(_obs_name, ee_ll_afb_obs)
_obs.set_description(r"$A_{FB}/A_{FB}(SM) of $" + _process_tex + r"$ at energy $E$")
_obs.tex = r"$A_{FB}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
