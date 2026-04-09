r"""Functions to parametrize subleading hadronic effects in $B\to V\ell^+\ell^-$
decays."""


import flavio
from flavio.classes import AuxiliaryQuantity, Implementation
from flavio.physics.bdecays.common import meson_quark
from flavio.physics.common import conjugate_par
from flavio.physics import ckm
from flavio.config import config


def _eps_u_over_t(par, B, V):
    qiqj = meson_quark[(B, V)]
    xi_u = ckm.xi('u', qiqj)(par)
    xi_t = ckm.xi('t', qiqj)(par)
    return xi_u/xi_t


def _deltaC_polynomial_par(q2, par, B, V, pname, h):
    process = B + '->' + V
    return (par[process + ' ' + pname + ' a_' + h + ' Re']
            + par[process + ' ' + pname + ' b_' + h + ' Re'] * q2
            + 1j * (par[process + ' ' + pname + ' a_' + h + ' Im']
                    + par[process + ' ' + pname + ' b_' + h + ' Im'] * q2))


def _deltaC_constant_par(par, B, V, pname, h):
    process = B + '->' + V
    return par[process + ' ' + pname + ' c_' + h + ' Re'] + 1j * par[process + ' ' + pname + ' c_' + h + ' Im']


class HelicityAmpsDeltaC(object):
    r"""Base Class for functions returning the shift of helcity amps for a
    shift in Wilson coefficients. Facilitates caching.

    This can be used to parametrize residual uncertainties due to subleading
    non-factorizable hadronic effects."""

    def __init__(self, B, V, par, q2):
        """Initialize the class and cache results needed more often."""
        self.B = B
        self.V = V
        self.q2 = q2
        self.par = par
        self.prefactor = flavio.physics.bdecays.bvll.amplitudes.prefactor(None, self.par, B, V)
        self.scale = config['renormalization scale']['bvll']
        self.mb = flavio.physics.running.running.get_mb(par, self.scale)
        self.mB = par['m_'+B]
        self.mV = par['m_'+V]
        self.ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(self.q2, self.par, self.B, self.V)

    def ha_deltaC(self, deltaC, C_name):
        wc  = {'7': 0, '7p': 0, 'v': 0, 'a': 0, 's': 0, 'p': 0,
               't': 0,'vp': 0, 'ap': 0, 'sp': 0, 'pp': 0, 'tp': 0, }
        wc[C_name] = deltaC
        return flavio.physics.bdecays.angular.helicity_amps_v(self.q2,
                    self.mB, self.mV, self.mb, 0,
                    0, 0,  # ml=0 as this only affects scalar contributions
                    self.ff, wc, self.prefactor)


class HelicityAmpsDeltaC_77p_polynomial(HelicityAmpsDeltaC):
    r"""Helicity amps from an effective shift in the Wilson coefficients
    $C_7$ (for the 0 and $-$ amplitudes), $C_7'$ (for the $+$ amplitude),
    and optional low-$q^2$ shifts in $C_9$ for all helicities, all taken as
    first-order polynomials in $q^2$.

    The extra $C_9$ pieces default to zero through the parameter tables, so
    the existing predictions are unchanged unless these parameters are set
    explicitly."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        ha = {}
        contributions = {
            ('0', 'V'): [('deltaC7', '0', '7'), ('deltaC9', '0', 'v')],
            ('pl', 'V'): [('deltaC7p', '+', '7p'), ('deltaC9', '+', 'v')],
            ('mi', 'V'): [('deltaC7', '-', '7'), ('deltaC9', '-', 'v')],
        }
        for amp_key, amp_contributions in contributions.items():
            total = 0
            for pname, h, wc_name in amp_contributions:
                deltaC = _deltaC_polynomial_par(self.q2, self.par, self.B, self.V, pname, h)
                total += self.ha_deltaC(deltaC, wc_name)[amp_key]
            ha[amp_key] = total
        return ha


class HelicityAmpsDeltaC_77p_hk_polynomial(HelicityAmpsDeltaC):
    r"""CKM-resolved analogue of `HelicityAmpsDeltaC_77p_polynomial` for
    $b\to d$ vector modes with structure
    $(1+\lambda_u/\lambda_t)H-(\lambda_u/\lambda_t)K$."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        eps_u = _eps_u_over_t(self.par, self.B, self.V)
        ha = {}
        deltaC7_0_H = _deltaC_polynomial_par(self.q2, self.par, self.B, self.V, 'deltaC7', '0')
        deltaC7_0_K = _deltaC_polynomial_par(self.q2, self.par, self.B, self.V, 'deltaC7K', '0')
        deltaC7p_pl_H = _deltaC_polynomial_par(self.q2, self.par, self.B, self.V, 'deltaC7p', '+')
        deltaC7p_pl_K = _deltaC_polynomial_par(self.q2, self.par, self.B, self.V, 'deltaC7pK', '+')
        deltaC7_mi_H = _deltaC_polynomial_par(self.q2, self.par, self.B, self.V, 'deltaC7', '-')
        deltaC7_mi_K = _deltaC_polynomial_par(self.q2, self.par, self.B, self.V, 'deltaC7K', '-')
        deltaC7_0 = (1 + eps_u) * deltaC7_0_H - eps_u * deltaC7_0_K
        deltaC7p_pl = (1 + eps_u) * deltaC7p_pl_H - eps_u * deltaC7p_pl_K
        deltaC7_mi = (1 + eps_u) * deltaC7_mi_H - eps_u * deltaC7_mi_K
        ha['0', 'V'] = self.ha_deltaC(deltaC7_0, '7')['0', 'V']
        ha['pl', 'V'] = self.ha_deltaC(deltaC7p_pl, '7p')['pl', 'V']
        ha['mi', 'V'] = self.ha_deltaC(deltaC7_mi, '7')['mi', 'V']
        return ha


class HelicityAmpsDeltaC_9_shift(HelicityAmpsDeltaC):
    r"""Helicity amps from an effective $q^2$-independent shift in the Wilson
    coefficients."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        ha = {}
        deltaC9_0 = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9', '0')
        deltaC9_pl = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9', '+')
        deltaC9_mi = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9', '-')
        ha['0', 'V'] = self.ha_deltaC(deltaC9_0, 'v')['0', 'V']
        ha['pl', 'V'] = self.ha_deltaC(deltaC9_pl, 'v')['pl', 'V']
        ha['mi', 'V'] = self.ha_deltaC(deltaC9_mi, 'v')['mi', 'V']
        return ha


class HelicityAmpsDeltaC_9_hk_shift(HelicityAmpsDeltaC):
    r"""CKM-resolved analogue of `HelicityAmpsDeltaC_9_shift` for $b\to d$
    vector modes with structure
    $(1+\lambda_u/\lambda_t)H-(\lambda_u/\lambda_t)K$."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        eps_u = _eps_u_over_t(self.par, self.B, self.V)
        ha = {}
        deltaC9_0_H = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9', '0')
        deltaC9_0_K = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9K', '0')
        deltaC9_pl_H = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9', '+')
        deltaC9_pl_K = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9K', '+')
        deltaC9_mi_H = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9', '-')
        deltaC9_mi_K = _deltaC_constant_par(self.par, self.B, self.V, 'deltaC9K', '-')
        deltaC9_0 = (1 + eps_u) * deltaC9_0_H - eps_u * deltaC9_0_K
        deltaC9_pl = (1 + eps_u) * deltaC9_pl_H - eps_u * deltaC9_pl_K
        deltaC9_mi = (1 + eps_u) * deltaC9_mi_H - eps_u * deltaC9_mi_K
        ha['0', 'V'] = self.ha_deltaC(deltaC9_0, 'v')['0', 'V']
        ha['pl', 'V'] = self.ha_deltaC(deltaC9_pl, 'v')['pl', 'V']
        ha['mi', 'V'] = self.ha_deltaC(deltaC9_mi, 'v')['mi', 'V']
        return ha


def fct_deltaC7C7p_polynomial(B, V):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return HelicityAmpsDeltaC_77p_polynomial(B, V, par, q2)()
    return fct


def fct_deltaC7C7p_hk_polynomial(B, V):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return HelicityAmpsDeltaC_77p_hk_polynomial(B, V, par, q2)()
    return fct


def fct_deltaC9_constant(B, V):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return HelicityAmpsDeltaC_9_shift(B, V, par, q2)()
    return fct


def fct_deltaC9_hk_constant(B, V):
    def fct(wc_obj, par_dict, q2, cp_conjugate):
        par = par_dict.copy()
        if cp_conjugate:
            par = conjugate_par(par)
        return HelicityAmpsDeltaC_9_hk_shift(B, V, par, q2)()
    return fct


# AuxiliaryQuantity & Implementation: subleading effects at LOW q^2

for had in [('B0','K*0'), ('B+','K*+'), ('Bs','phi'), ]:
    process = had[0] + '->' + had[1] + 'll' # e.g. B0->K*0mumu
    quantity = process + ' subleading effects at low q2'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' subleading hadronic effects (i.e. all effects not included'
                    r'elsewhere) at $q^2$ below the charmonium resonances')


    # Implementation: C7-C7'-polynomial
    iname = process + ' deltaC7, 7p polynomial'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC7C7p_polynomial(B=had[0], V=had[1]))
    i.set_description(r"Effective shift in the Wilson coefficient $C_7(\mu_b)$"
                      r" (in the $0$ and $-$ helicity amplitudes) and"
                      r" $C_7'(\mu_b)$ (in the $+$ helicity amplitude)"
                      r" as a first-order polynomial in $q^2$, optionally"
                      r" supplemented by first-order polynomial shifts in"
                      r" $C_9(\mu_b)$ for the $0$, $+$, and $-$ helicity"
                      r" amplitudes. The added $C_9$ parameters default to"
                      r" zero, so the previous predictions are recovered"
                      r" automatically.")


# AuxiliaryQuantity & Implementation: CKM-resolved subleading effects in Bs->K*0ll at LOW q^2

process = 'Bs->K*0ll'
quantity = process + ' subleading effects at low q2'
a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                ' subleading hadronic effects (i.e. all effects not included'
                r'elsewhere) at $q^2$ below the charmonium resonances')
iname = process + ' deltaC7, 7p H+K polynomial'
i = Implementation(name=iname, quantity=quantity,
               function=fct_deltaC7C7p_hk_polynomial(B='Bs', V='K*0'))
i.set_description(r"Effective shift in $C_7(\mu_b)$ (in the $0$ and $-$ helicity amplitudes)"
                  r" and $C_7'(\mu_b)$ (in the $+$ helicity amplitude) with CKM structure "
                  r"$(1+\lambda_u/\lambda_t)H-(\lambda_u/\lambda_t)K$ as a first-order polynomial in $q^2$.")


# AuxiliaryQuantity & Implementation: subleading effects at HIGH q^2

for had in [('B0','K*0'), ('B+','K*+'), ('Bs','phi'), ]:
    process = had[0] + '->' + had[1] + 'll' # e.g. B0->K*0mumu
    quantity = process + ' subleading effects at high q2'
    a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
    a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                    ' subleading hadronic effects (i.e. all effects not included'
                    r'elsewhere) at $q^2$ above the charmonium resonances')

    # Implementation: C9 constant shift
    iname = process + ' deltaC9 shift'
    i = Implementation(name=iname, quantity=quantity,
                   function=fct_deltaC9_constant(B=had[0], V=had[1]))
    i.set_description(r"Effective constant shift in the Wilson coefficient $C_9(\mu_b)$.")


# AuxiliaryQuantity & Implementation: CKM-resolved subleading effects in Bs->K*0ll at HIGH q^2

process = 'Bs->K*0ll'
quantity = process + ' subleading effects at high q2'
a = AuxiliaryQuantity(name=quantity, arguments=['q2', 'cp_conjugate'])
a.description = ('Contribution to ' + process + ' helicity amplitudes from'
                ' subleading hadronic effects (i.e. all effects not included'
                r'elsewhere) at $q^2$ above the charmonium resonances')
iname = process + ' deltaC9 H+K shift'
i = Implementation(name=iname, quantity=quantity,
               function=fct_deltaC9_hk_constant(B='Bs', V='K*0'))
i.set_description(r"Effective constant shift in $C_9(\mu_b)$ with CKM structure "
                  r"$(1+\lambda_u/\lambda_t)H-(\lambda_u/\lambda_t)K$.")
