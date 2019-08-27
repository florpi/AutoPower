import camb
from camb import model, initialpower


def get_power_spectrum(h, omc, omb = 0.0486,
     redshift = 0., nkpoints = 200, mode = 'non-linear'):
    
    H0 = 100. * h
    ombh2 = omb * h **2
    omch2 = omc * h **2
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0 = H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params()
    
    pars.set_matter_power(redshifts=[redshift], kmax=2.0)

    if mode == 'linear':

        pars.NonLinear = model.NonLinear_none
    
    elif mode == 'non-linear':
        #Non-Linear spectra (Halofit)
        pars.NonLinear = model.NonLinear_both

    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    k, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1,
                                                        npoints = nkpoints)
    
    return k, pk[0, :] # 0 for only one redshift
  
