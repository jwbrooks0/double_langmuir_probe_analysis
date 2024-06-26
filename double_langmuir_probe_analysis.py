# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 08:20:29 2021

@author: jwbrooks
"""


# %% import libraries
import numpy as _np
import matplotlib.pyplot as _plt
import xarray as _xr
from scipy.constants import c as _C, mu_0 as _MU_0


# %% additional constants

# Misc
_AMU = 1.66054e-27             # amu to kg
_EP_0 = 1.0 / (_MU_0 * _C**2)  # vacuum permittivity

#  Atomic masses
_M_XE = 131.293 * _AMU         # mass of xenon
_M_AR = 39.948 * _AMU         # mass of argon


# %% analyze_DP_data()
def analyze_DP_data(    DP_data, 
                        probe_area, 
                        probe_radius, 
                        m_i, #m_i=_M_XE, 
                        probe_geom, #probe_geom='cylindrical',
                        V_lim=[], 
                        guesses=(1e17, 2, 0, 0), 
                        plot=False,
                        filename=None, 
                        plotall=False,
                        verbose=False,
                        allow_for_V_and_I_offsets=True,
                        density_fit_lims=[1e10, _np.infty],
                        TeV_fit_lims=[1e-4, _np.infty],
                        ):
    
    """
    Analyzes a Double Langmuir Probe (DP) IV trace and provides density and temperature.  
    
    Parameters
    ----------
    DP_data : xarray.core.dataarray.DataArray
        1D data array of probe current and dimensions of volts ("V")
    probe_area : float
        Probe area in m^2
    probe_radius : float
        Probe radius in m
    probe_geom : str
        Geometry of probe.  Should be "planar", "cylindrical" or "spherical"
    V_lim : list of two floats
        Trims the voltage range of the data to between these two points
    m_i : float
        Mass of the ion (e.g. argon, xenon, etc)
    guesses : tuple of list of four floats
        Guess values for density (m^-3),  temperature (eV), V_offset (V) and I_offset (A).  
    plot : bool
        Provides an optional plot of the results
    filename : str
        Optionally saves the plot to file.  Note that plot must be equal to True.  
    plotall : bool
        Plots many of the intermediate steps.  Useful for debugging.  
    verbose : bool
        Prints intermediate results.  Useful for debugging.
        
    Returns
    -------
    density : float
        Calculated density.  Units are m^-3.  
    temp : float
        Calculated temperature.  Units are eV.
    fit : xarray.core.dataarray.DataArray
        The final fit to the model.
        
    References
    ----------
        
    This code is primarily based on Brian Beal's "Improved analysis techniques for cylindrical and spherical double probes" .
    We also reference: Daniel Brown's "Experimental Assessment of Double Langmuir Probe Analysis Techniques in a Hall Thruster Plume".  
    Citations below.
     * https://aip.scitation.org/doi/10.1063/1.4739221 (cylindrical and spherical double Langmuir probe)
     * https://doi.org/10.2514/6.2012-3869
     * https://doi.org/10.2514/1.B35531 (planar double Langmuir probe)
     
    """

    from scipy.optimize import root_scalar, curve_fit
    from functools import partial
    from scipy.constants import e as _E, m_e as _M_E, c as _C, mu_0 as _MU_0
    
    # check inputs
    assert probe_geom in ["planar", "cylindrical", "spherical"], """
        probe_geom should be one of: planar, cylindrical, spherical"""
    assert TeV_fit_lims[0] > 0, """ 
        TeV_fit_lims must be greater than 0 """
    assert density_fit_lims[0] > 0, """ 
        density_fit_lims must be greater than 0 """
    assert guesses[0] > 1e10, """ 
        Your density-guess should be in a linear scale: e.g. 1e^{16} """

    # rename parameters
    A = probe_area 
    r_probe = probe_radius  

    # trim Voltage in data if requested
    if V_lim != []:
        DP_data = DP_data.where((DP_data.V >= V_lim[0]) & (DP_data.V <= V_lim[1])).dropna('V')
        
    # build curve_fit limits (bounds): log10(density), temp, Voffset, Ioffset
    if allow_for_V_and_I_offsets is True:
        bounds = ([_np.log10(density_fit_lims[0]), TeV_fit_lims[0], -_np.infty, -_np.infty], [_np.log10(density_fit_lims[1]), TeV_fit_lims[1], _np.infty, _np.infty])
    else:
        bounds = ([_np.log10(density_fit_lims[0]), TeV_fit_lims[0]], [_np.log10(density_fit_lims[1]), TeV_fit_lims[1]])
        
    # %% subfunctions

    def _lambda_d(temperature_in_eV, density):
        """ definition of the debye length """
        return _np.sqrt(_EP_0 * temperature_in_eV / (density * _E))
    
    def _I0(n0, xi):
        """ eq. 6 in Beal - Ion saturation current """
        sqrt_interior = _np.max([xi / (2 * _np.pi * m_i), 0]) # make sure the sqrt_interior is positive.  
        I0 = _E**1.5 * n0 * A * _np.sqrt(sqrt_interior)
        return I0

    def _basic_DP_model(Vp, log10n0, xi, Voffset, Ioffset):
        """ eq. 9 in Beal - basic DP model without advanced sheath expansion terms """
        n0 = 10 ** log10n0
        if allow_for_V_and_I_offsets==False:
            return _I0(n0, xi) * 0.61 * _np.sqrt(2 * _np.pi) * _np.tanh((Vp - 0) / (2 * xi)) + 0
        else:
            return _I0(n0, xi) * 0.61 * _np.sqrt(2 * _np.pi) * _np.tanh((Vp - Voffset) / (2 * xi)) + Ioffset

    def _expanded_DP_model(Vp, log10n0, xi, Voffset, Ioffset, a, b, V1):
        """ eq. 7 in Beal - advanced DP model with advanced sheath expansion terms """
        n0 = 10 ** log10n0
        # return _I0(n0, xi) * (a * (-V1 / xi)**b * _np.tanh((Vp - 0) / (2 * xi)) + (a * (-V1 / xi)**b - a * (-(V1 + Vp - 0) / xi)**b) / (_np.exp((Vp - 0) / xi) + 1)) + 0
        return _I0(n0, xi) * (a * (-V1 / xi)**b * _np.tanh((Vp - Voffset) / (2 * xi)) + (a * (-V1 / xi)**b - a * (-(V1 + Vp - Voffset) / xi)**b) / (_np.exp((Vp - Voffset) / xi) + 1)) + Ioffset

    def fit_basic_DP_model(DP_data, guesses, bounds, plot=False):
        """ fits to eq. 9 in Beal """
        Vp = DP_data.V.data
        Ip = DP_data.data
        
        # convert density-guess to log scale
        guesses_log = list(guesses)
        guesses_log[0] = _np.log10(guesses_log[0])
        guesses_log = tuple(guesses_log)

        # perform fit
        params_fit_log, _ = curve_fit(_basic_DP_model, Vp, Ip, guesses_log, bounds=bounds)
        fit_data = _xr.DataArray(_basic_DP_model(Vp, *params_fit_log), dims=DP_data.dims, coords=DP_data.coords)

        # convert density-fit back to linear scale
        params_fit = params_fit_log * 1.0
        params_fit[0] = 10 ** params_fit_log[0]
        
        # optional plot
        if plot is True:
            fig, ax = _plt.subplots()
            DP_data.plot(marker='x', linestyle='', label='raw')
            if True:
                fit_guess = _xr.DataArray(_basic_DP_model(Vp, *guesses_log), dims=DP_data.dims, coords=DP_data.coords)
                fit_guess.plot(label='guess', ls="--")
            fit_data.plot(label='fit')
            ax.legend()
            
            if True:
                title = f"{params_fit[0]:.3e}, {params_fit[1]:.3f}, {params_fit[2]:.3f}, {params_fit[3]:.3f}"
                ax.set_title(title)
            
        return params_fit  # density, temp, Voffset, Ioffset

    def fit_expanded_DP_model(DP_data, a, b, V1, guesses, bounds, plot=False):
        """ fits to eq. 7 in Beal """
        Vp = DP_data.V.data
        Ip = DP_data.data
        
        # convert density-guess to log scale
        guesses_log = guesses * 1.0
        guesses_log[0] = _np.log10(guesses_log[0])
        
        # define several values of _expanded_DP_model() to be fixed constants (and not fit parameters); https://stackoverflow.com/a/67286996
        fitfun = partial(_expanded_DP_model, a=a, b=b, V1=V1)

        # perform fit
        params_fit_log, _ = curve_fit(fitfun, Vp, Ip, guesses_log, bounds=bounds)
        
        # params_fit_log, _ = curve_fit(fitfun, Vp, Ip, bounds=bounds)
        fit_data = _xr.DataArray(_expanded_DP_model(Vp, a=a, b=b, V1=V1, *params_fit_log), dims=DP_data.dims, coords=DP_data.coords, attrs=DP_data.attrs)

        # convert density-fit back to linear scale
        params_fit = params_fit_log * 1.0
        params_fit[0] = 10 ** params_fit_log[0]
        
        # optional plot
        if plot is True:
            fig, ax = _plt.subplots()
            ax.axhline(0, linestyle='--', color='grey')
            ax.axvline(0, linestyle='--', color='grey')
            DP_data.plot(marker='x', linestyle='', label='raw')
            if True:
                fit_guess = _xr.DataArray(_expanded_DP_model(Vp, a=a, b=b, V1=V1, *guesses_log), dims=DP_data.dims, coords=DP_data.coords)
                fit_guess.plot(label='guess', ls="--")
            fit_data.plot(label='fit', color='red', linewidth=1.5)
            ax.legend()
            if True:
                title = f"{params_fit[0]:.3e}, {params_fit[1]:.3f}, {params_fit[2]:.3f}, {params_fit[3]:.3f}, {a:.3f}, {b:.3f}"
                ax.set_title(title)

        return params_fit  # density, temp, Voffset, Ioffset

    def solve_lambdad_a_b(density, temperature_in_eV, probe_geom):
        """ Solves for the debye legngth and parameters a and b.
        For planar probes: a and b come from Lobbia 2017
        For cylindrical and spherical, a and b come from Beal 2012 """

        # debye length definition
        lambda_d = _lambda_d(temperature_in_eV, density)

        # a and b terms
        if 'cylindrical' in probe_geom:
            ## a and b terms from Beal 2012, Table 1
            a = 1.18 - 0.00080 * (r_probe / lambda_d)**1.35
            b = 0.0684 + (0.722 + 0.928 * r_probe / lambda_d)**(-0.729)
        elif "spherical" in probe_geom:
            print("Warning: I've never tested this code with data from a spherical probe.  ")
            ## a and b terms from Beal 2012, Table 1
            a = 1.98 + 4.49 * (r_probe / lambda_d) ** (-1.31)
            b = -2.95 + 3.61 * (r_probe / lambda_d) ** (-0.125)
            ## these numbers are from Lobbia 2017.  Not sure why they don't agree with Beal. #TODO figure out why
            #a = 1.58 + (-0.056 + 0.816 * (r_probe / lambda_d)) ** (-0.744)
            #b = -0.933 + (0.0148 + 0.119 * (r_probe / lambda_d)) ** (-0.125)
        elif "planar" in probe_geom:
            ## these numbers are from Lobbia 2017.
            a = 3.47 * (r_probe / lambda_d) ** (-0.749)
            b = 0.806 * (r_probe / lambda_d) ** (-0.0692)
        else:
            raise Exception('Did not recognize your probe geometry')
            
        if probe_geom in ['cylindrical', "spherical"]:
            if (r_probe / lambda_d < 3) or (r_probe / lambda_d > 50):
                print("Warning: r_probe / lambda_d = %.2f but should be between 3 and 50 as stated in Table 1 of Beal and Eq. 7 of Lobbia.  Deviating may result in code errors or flawed measurements.  " % (r_probe / lambda_d))  
        elif probe_geom in ["planar"]:
            if (r_probe / lambda_d < 10) or (r_probe / lambda_d > 45):
                print("Warning: r_probe / lambda_d = %.2f but should be between 10 and 45 as stated in Eq. 8 of Lobbia.  Deviating may result in code errors or flawed measurements.  " % (r_probe / lambda_d))  
        
        return lambda_d, a, b

    def solve_for_V1(DP_data, xi, a, b, plot=False):
        """ Solves Eq. 8 in Beal using a 'crude' root finding implementation """  
        # TODO consider improving root finding algorithm.  At present, it seems consistent, but because it's a "brute force" method, it runs VERY slow

        Vp = DP_data.V.data
        V1 = _np.zeros(len(Vp), dtype=float)

        # Perform root finding for each value of V_probe
        for i, Vp_i in enumerate(Vp):
            def Eq8(V1):
                LHS = a * ((-V1 / xi)**b + (-(V1 + Vp_i) / xi)**b) - (m_i / _M_E)**0.5 * _np.exp(V1 / xi) * (1 + _np.exp(Vp_i / xi))
                return LHS
            V1[i] = root_scalar(Eq8, bracket=[-1000, _np.min([0, -Vp_i])]).root
        V1 = _xr.DataArray(V1, dims=DP_data.dims, coords=DP_data.coords, attrs={'name': 'V1', 'long_name': 'V1'})

        # optional plot
        if plot is True:
            fig, ax = _plt.subplots()
            V1.plot(ax=ax)

        return V1

    def converge(DP_data, density, temp, a, b, V1, guess_fit_params, bounds, 
                 verbose=False, plotall=False):
        """ Repeats steps 2 to 4 until convergence """

        densities = [density]
        repeat = True
        count = 0
        while repeat is True:
            if verbose:
                print(count)

            # step 2
            lambda_d, a, b = solve_lambdad_a_b(density, temp, probe_geom=probe_geom)

            # step 3
            V1 = solve_for_V1(DP_data, xi=temp, a=a, b=b)

            # step 4
            expanded_fit_params = fit_expanded_DP_model(DP_data, a=a, b=b, V1=V1, guesses=guess_fit_params, plot=plotall, bounds=bounds)
            density, temp, Voffset, Ioffset = expanded_fit_params
            guess_fit_params = expanded_fit_params
            if verbose:
                print(expanded_fit_params)

            # add latest density to a list
            densities.append(density)
            count += 1

            if verbose:
                if count > 4:
                    print('gradient = ', _np.mean(_np.abs(_np.gradient(_np.abs(_np.array(densities))))[-4:-1]))

            # perform a minimum of 10 iterations and then start checking for convergence.  If my `crude' gradient descent becomes small, it is assumed that convergence is achieved.
            if count >= 10:
                if _np.mean(_np.abs(_np.gradient(_np.abs(_np.array(densities))))[-4:-1]) < 1e6:
                    repeat = False

        if verbose:
            print('n = {n:.3e}, T = {temp:.3f}, V0 = {Voffset:.3e},\nI0 = {Ioffset:.3e}, a = {a:.3e}, b = {b:.3e}'.format(n=density, temp=temp, Voffset=Voffset, Ioffset=Ioffset, a=a, b=b))
            
        fit = _expanded_DP_model(DP_data.V.data, 
                           log10n0=_np.log10(density), 
                           xi=temp, 
                           Voffset=Voffset, 
                           Ioffset=Ioffset, 
                           a=a, b=b, V1=V1)
        print(_np.log10(density), temp, Voffset, Ioffset)
        print(expanded_fit_params)
        return fit, expanded_fit_params

    # %% Main

    # step 1 - fit data to eq. 9 and eq. 6 to get a reasonable guess of the temperature and density
    basic_fit_params = fit_basic_DP_model(DP_data, guesses=guesses, plot=plotall, bounds=bounds)
    guess_density, guess_temp, Voffset, Ioffset = basic_fit_params
    if verbose:
        print('initial fit parameters on basic model:')
        print(basic_fit_params)

    # step 2 - solve for lambda_d, a, and b
    lambda_d, a, b = solve_lambdad_a_b(guess_density, guess_temp, probe_geom=probe_geom)

    # step 3 - solve for V1(Vp)
    V1 = solve_for_V1(DP_data, xi=guess_temp, a=a, b=b, plot=plotall)

    # step 4 - solve for updated temperature and density values
    first_fit_params = fit_expanded_DP_model(DP_data=DP_data, a=a, b=b, V1=V1, guesses=basic_fit_params, plot=plotall, bounds=bounds)
    density, temp, Voffset, Ioffset = first_fit_params
    if verbose:
        print('initial fit parameters on expanded model:')
        print(first_fit_params)

    # step 5 - iterate on steps 2 to 4 until convergence
    fit, final_fit_params = converge(DP_data, density, temp, a, b, V1, 
                                     guess_fit_params=first_fit_params, 
                                     plotall=plotall, 
                                     verbose=verbose, 
                                     bounds=bounds)
    
    # optional plot
    if plot is True:
        
        lambda_d, a, b = solve_lambdad_a_b(final_fit_params[0], final_fit_params[1], probe_geom=probe_geom)
        
        fig, ax = _plt.subplots()
        ax.axhline(0, linestyle='--', color='grey')
        ax.axvline(0, linestyle='--', color='grey')
        DP_data.plot(marker='x', linestyle='', label='raw')
        
        if plotall is True:
            guesses_log = list(guesses)
            guesses_log[0] = _np.log10(guesses_log[0])
            fit_guess = _xr.DataArray(_expanded_DP_model(DP_data.V.data, *guesses_log, a=a, b=b, V1=V1), dims=DP_data.dims, coords=DP_data.coords)
            fit_guess.plot(ax=ax, label='Initial guess', ls="--")
        
        if plotall is True:
            basic_fit_params_log = basic_fit_params * 1.0
            basic_fit_params_log[0] = _np.log10(basic_fit_params_log[0])
            fit_basic = _xr.DataArray(_expanded_DP_model(DP_data.V.data, *basic_fit_params_log, a=a, b=b, V1=V1), dims=DP_data.dims, coords=DP_data.coords)
            fit_basic.plot(ax=ax, label='Basic fit', ls="-.")
            
        if plotall is True:
            first_fit_params_log = first_fit_params * 1.0
            first_fit_params_log[0] = _np.log10(first_fit_params_log[0])
            firstfit_data = _xr.DataArray(_expanded_DP_model(DP_data.V.data, *first_fit_params_log, a=a, b=b, V1=V1), dims=DP_data.dims, coords=DP_data.coords)
            firstfit_data.plot(ax=ax, label='first fit', color='k', ls=":", linewidth=1.5)
            
        fit.plot(ax=ax, label='final fit', color='red', linewidth=1.5)
            
        temp = final_fit_params[1]
        density = final_fit_params[0]
        lambda_d = _lambda_d(temperature_in_eV=temp, density=density)
        r_over_lambda = r_probe / lambda_d
        title = f'n0 = {density:.3e} m^-3 \ntemp = {temp:.3f} eV\nr_probe / lambda_d = {r_over_lambda:.3f}'
        ax.set_title(title)
        ax.legend(loc="upper left")
        fig.set_tight_layout(True)
        
        if type(filename) is not type(None):
            fig.savefig(filename, dpi=150)
                
    return fit, final_fit_params


if __name__ == '__main__':
    """
    The following two examples are pulled from Beal's paper.  (see references)
    
    Note that there is a discrepancy (i.e. typos) in the probe geometries between the two papers referenced above.  
    Below, we choose the dimensions that give us the best agreement with the paper.  
    """
    
    # %% Example 1
    if True:
        probe_length = 0.1969
        probe_diameter = 0.0048 * 2 # note that we've doubled the diameter.  There is a discrepancy in dimensions when comparing the two papers.
        V = _np.array([-14.10823436, -12.66852966, -11.19307608,  -9.49287957,
                -8.17640629,  -6.55222209,  -5.03126994,  -3.5103178 ,
                -2.19955089,  -1.41413374,  -1.29017336,  -0.7609043 ,
                -0.74980982,  -0.41200456,  -0.11648102,   0.02882086,
                 0.19915373,   0.46390092,   0.91743576,   1.21082779,
                 1.7969749 ,   3.0007817 ,   4.40092243,   5.99758307,
                 7.52979454,   9.00828835,  10.52924049,  12.05019264,
                13.57114479,  14.97510062])
        V = _xr.DataArray(V, dims='V', coords=[V], attrs = {'long_name': 'DP voltage', 'units': 'V'})
        
        IV = _np.array([-6.34595e-04, -6.18963e-04, -6.01950e-04, -5.84530e-04,
               -5.63626e-04, -5.35795e-04, -5.16066e-04, -4.79979e-04,
               -4.30871e-04, -3.93982e-04, -3.25608e-04, -2.72065e-04,
               -2.08396e-04, -1.20914e-04, -2.06511e-05,  3.46343e-05,
                9.64660e-05,  2.09831e-04,  2.82632e-04,  3.69833e-04,
                4.40770e-04,  4.87801e-04,  5.18764e-04,  5.52252e-04,
                5.76294e-04,  5.95037e-04,  6.15903e-04,  6.35192e-04,
                6.49014e-04,  6.58850e-04])
        IV = _xr.DataArray(IV, dims='V', coords=[V], attrs = {'long_name': 'DP current', 'units': 'A'}).sortby('V')
    
        probe_radius = probe_diameter / 2
        probe_area = (_np.pi * probe_diameter * probe_length + _np.pi * probe_diameter**2 / 4)
        
        analyze_DP_data(    DP_data=IV, 
                            probe_area=probe_area, 
                            probe_radius=probe_radius, 
                            m_i=_M_XE, 
                            guesses=(2e15, 1, 0, 0), 
                            plot=True,
                            filename=None, 
                            plotall=False,
                            verbose=False,
                            density_fit_lims=[1e10, _np.infty],
                            TeV_fit_lims=[1e-4, _np.infty],
                            probe_geom="cylindrical",
                            )
        fig = _plt.gcf()
        fig.savefig('dataset1.png', dpi=150)
        
        
    # %% Example 2
    if True:
        
        probe_length = 0.1969
        probe_diameter = 0.0048 * 1 # There is a discrepancy in dimensions when comparing the two papers.
        V = _np.array([-14.01134038, -12.37077387, -10.53036379, -8.77065988,
                  -7.01078211, -5.418378025, -4.188673381, -3.026725493,
                  -2.059888776, -1.328925821, -0.457067703, 0.295952373, 
                  1.04445134, 1.638569274, 2.594068176, 3.621358974, 
                  4.410115251, 5.931873684, 7.429331119, 9.118551251, 
                  10.84835489, 12.60785136, 14.25019047, 15.41437982, ])
        V = _xr.DataArray(V, dims='V', coords=[V], attrs = {'long_name': 'DP voltage', 'units': 'V'})
    
        IV = _np.array([-16.94733903, -16.10708996, -14.91762878, -13.65315567, 
                  -12.14908955, -10.60398693, -8.749187106, -6.836143647, 
                  -5.075066261, -3.26749274, -1.323614422, 0.483828343, 
                  2.202873705, 4.051986344, 5.866790874, 8.11671034, 
                  10.03321313, 11.23979288, 12.74628083, 14.15816318, 
                  15.43787759, 16.4164729, 16.82657669, 17.58160021, ]) * 1e-6
        IV = _xr.DataArray(IV, dims='V', coords=[V], attrs = {'long_name': 'DP current', 'units': 'A'}).sortby('V')
    
        probe_radius = probe_diameter / 2
        probe_area = (_np.pi * probe_diameter * probe_length + _np.pi * probe_diameter**2 / 4)
        
        analyze_DP_data(    IV, 
                            probe_area=probe_area, 
                            probe_radius=probe_radius, 
                            m_i=_M_XE, 
                            guesses=(1e13, 1, 0, 0), 
                            plot=True,
                            filename=None, 
                            plotall=False,
                            verbose=False,
                            probe_geom="cylindrical",
                            )
        fig = _plt.gcf()
        fig.savefig('dataset2.png', dpi=150)
     