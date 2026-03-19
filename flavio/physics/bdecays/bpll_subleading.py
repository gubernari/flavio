import flavio
import numpy as np
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.physics import ckm
from flavio.config import config

# auxiliary function to construct helicity_amps_deltaC7, helicity_amps_deltaC9
def _helicity_amps_deltaC(q2, deltaC, C_name, par, B, P):
    mB = par['m_'+B]
    mP = par['m_'+P]
    scale = config['renormalization scale']['bpll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    N = flavio.physics.bdecays.bpll.prefactor(q2, par, B, P)
    ff = flavio.physics.bdecays.bpll.get_ff(q2, par, B, P)
    wc  = {'7': 0, '7p': 0, 'v': 0, 'a': 0, 's': 0, 'p': 0, 't': 0,'vp': 0, 'ap': 0, 'sp': 0, 'pp': 0, 'tp': 0, }
    wc[C_name] = deltaC
    return flavio.physics.bdecays.angular.helicity_amps_p(q2, mB, mP, mb, 0, 0, 0, ff, wc, N) # ml=0 as this only affects scalar contributions

# parametrization of subleading hadronic corrections
def helicity_amps_deltaC7(q2, deltaC7, par, B, P):
    r"""A function returning a contribution to the helicity amplitudes in
    $B\to P\ell^+\ell^-$ coming from an effective shift of
    the Wilson coefficient $C_7(\mu_b)$. This can be used to parametrize
    residual uncertainties due to subleading non-factorizable hadronic effects.
    """
    return _helicity_amps_deltaC(q2, deltaC7, '7', par, B, P)

def helicity_amps_deltaC9(q2, deltaC9, par, B, P):
    r"""A function returning a contribution to the helicity amplitudes in
    $B\to P\ell^+\ell^-$ coming from an effective shift of
    the Wilson coefficient $C_9(\mu_b)$. This can be used to parametrize
    residual uncertainties due to subleading non-factorizable hadronic effects.
    """
    return _helicity_amps_deltaC(q2, deltaC9, 'v', par, B, P)


# lambda_u/lambda_t = xi_u/xi_t, where xi_i = V_ib V_is* for b->s and V_id V_ib* for b->d transitions
def _eps_u_over_t(par, B, P):
    qiqj = meson_quark[(B, P)]
    xi_u = ckm.xi('u', qiqj)(par)
    xi_t = ckm.xi('t', qiqj)(par)
    return xi_u/xi_t


def _deltaC9_polynomial_par(q2, par, B, P, pname='deltaC9'):
    process = B + '->' + P
    return (par[process + ' ' + pname + ' a Re'] + par[process + ' ' + pname + ' b Re'] * q2
            + 1j * (par[process + ' ' + pname + ' a Im'] + par[process + ' ' + pname + ' b Im'] * q2))


def _deltaC9_constant_par(par, B, P, pname='deltaC9'):
    process = B + '->' + P
    return par[process + ' ' + pname + ' c Re'] + 1j * par[process + ' ' + pname + ' c Im']


# One possibility is to parametrize the effective shift in C7 or C9 as a simple
# polynomial in q2.

def helicity_amps_deltaC7_polynomial(q2, par, B, P):
    deltaC7   =( par[B+'->'+P+' deltaC7 a Re']  + par[B+'->'+P+' deltaC7 b Re'] *q2
             +1j*( par[B+'->'+P+' deltaC7 a Im']  + par[B+'->'+P+' deltaC7 b Im'] *q2 ))
    return helicity_amps_deltaC7(q2, deltaC7, par, B, P)

# note that, when parametrizing the correction as a shift to C9 rather than C7,
# the contribution to the transverse (+ and -) amplitudes has to start with
# 1/q2, otherwise one effectively sets corrections to B->Pgamma to zero.
def helicity_amps_deltaC9_polynomial(q2, par, B, P):
    deltaC9 = _deltaC9_polynomial_par(q2, par, B, P, pname='deltaC9')
    return helicity_amps_deltaC9(q2, deltaC9, par, B, P)

# a constant shift, e.g. for high q^2
def helicity_amps_deltaC9_constant(q2, par, B, P):
    deltaC9 = _deltaC9_constant_par(par, B, P, pname='deltaC9')
    return helicity_amps_deltaC9(q2, deltaC9, par, B, P)


def helicity_amps_deltaC9_hk_polynomial(q2, par, B, P):
    r"""Subleading hadronic correction for $B\to\pi\ell^+\ell^-$ with explicit
    CKM structure:
    $(1+\lambda_u/\lambda_t)H-(\lambda_u/\lambda_t)K$."""
    eps_u = _eps_u_over_t(par, B, P)
    deltaC9_H = _deltaC9_polynomial_par(q2, par, B, P, pname='deltaC9')
    deltaC9_K = _deltaC9_polynomial_par(q2, par, B, P, pname='deltaC9K')
    deltaC9 = (1 + eps_u) * deltaC9_H - eps_u * deltaC9_K
    return helicity_amps_deltaC9(q2, deltaC9, par, B, P)


def helicity_amps_deltaC9_hk_constant(q2, par, B, P):
    r"""Constant high-$q^2$ analogue of `helicity_amps_deltaC9_hk_polynomial`."""
    eps_u = _eps_u_over_t(par, B, P)
    deltaC9_H = _deltaC9_constant_par(par, B, P, pname='deltaC9')
    deltaC9_K = _deltaC9_constant_par(par, B, P, pname='deltaC9K')
    deltaC9 = (1 + eps_u) * deltaC9_H - eps_u * deltaC9_K
    return helicity_amps_deltaC9(q2, deltaC9, par, B, P)


# Functions returning functions needed for Implementation
def fct_deltaC7_polynomial(B, P):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return helicity_amps_deltaC7_polynomial(q2, par, B, P)
    return fct

def fct_deltaC9_polynomial(B, P):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return helicity_amps_deltaC9_polynomial(q2, par, B, P)
    return fct

def fct_deltaC9_constant(B, P):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return helicity_amps_deltaC9_constant(q2, par, B, P)
    return fct


def fct_deltaC9_hk_polynomial(B, P):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return helicity_amps_deltaC9_hk_polynomial(q2, par, B, P)
    return fct


def fct_deltaC9_hk_constant(B, P):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return helicity_amps_deltaC9_hk_constant(q2, par, B, P)
    return fct

# AuxiliaryQuantity & Implementatation: subleading effects at LOW q^2

for had in [('B0','K0'), ('B+','K+'),]:
    process = had[0] + '->' + had[1] + 'll' # e.g. B0->K*0ll
    quantity = process + ' subleading effects at low q2'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' subleading hadronic effects (i.e. all effects not included'
                    r'elsewhere) at $q^2$ below the charmonium resonances')

    # Implementation: C7-polynomial
    iname = process + ' deltaC7 polynomial'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC7_polynomial(B=had[0], P=had[1]))
    i.set_description(r"Effective shift in the Wilson coefficient $C_7(\mu_b)$"
                      r" as a first-order polynomial in $q^2$.")

    # Implementation: C9-polynomial
    iname = process + ' deltaC9 polynomial'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC9_polynomial(B=had[0], P=had[1]))
    i.set_description(r"Effective shift in the Wilson coefficient $C_9(\mu_b)$"
                      r" as a first-order polynomial in $q^2$.")


# AuxiliaryQuantity & Implementatation: subleading effects at HIGH q^2

for had in [('B0','K0'), ('B+','K+'), ]:
    for l in ['e', 'mu', 'tau']:
        process = had[0] + '->' + had[1] + 'll' # e.g. B0->K0ll
        quantity = process + ' subleading effects at high q2'
        a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
        a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                        ' subleading hadronic effects (i.e. all effects not included'
                        r'elsewhere) at $q^2$ above the charmonium resonances')

        # Implementation: C9 constant shift
        iname = process + ' deltaC9 shift'
        i = Implementation(name=iname, quantity=quantity,
                       function=fct_deltaC9_constant(B=had[0], P=had[1]))
        i.set_description(r"Effective constant shift in the Wilson coefficient $C_9(\mu_b)$.")


# AuxiliaryQuantity & Implementation: CKM-resolved subleading effects in B->pill
for had in [('B0', 'pi0'), ('B+', 'pi+')]:
    process = had[0] + '->' + had[1] + 'll'

    quantity = process + ' subleading effects at low q2'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' subleading hadronic effects (i.e. all effects not included'
                    r' elsewhere) at $q^2$ below the charmonium resonances')
    iname = process + ' deltaC9 H+K polynomial'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC9_hk_polynomial(B=had[0], P=had[1]))
    i.set_description(r"Effective shift in $C_9(\mu_b)$ with CKM structure "
                      r"$(1+\lambda_u/\lambda_t)H-(\lambda_u/\lambda_t)K$"
                      r" as a first-order polynomial in $q^2$.")

    quantity = process + ' subleading effects at high q2'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' subleading hadronic effects (i.e. all effects not included'
                    r' elsewhere) at $q^2$ above the charmonium resonances')
    iname = process + ' deltaC9 H+K shift'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC9_hk_constant(B=had[0], P=had[1]))
    i.set_description(r"Effective constant shift in $C_9(\mu_b)$ with CKM "
                      r"structure $(1+\lambda_u/\lambda_t)H-(\lambda_u/\lambda_t)K$.")
