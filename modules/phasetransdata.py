import os
import numpy
import scipy.io
import modules.io as io
import modules.misc_func as misc
from numba import njit,types
from tqdm.notebook import tqdm

def calc_phasetrans_params(
    d,
    param_name_control,
    n_time_steps_sample=None,
    calc_suscept_bootstrap_error=False,
    param_name_orderpar='rho',
    param_name_systemsize='N',
    **bootstrap_args
):
    """
    Computes the order parameter and susceptibility from time series data.
    Phase transition quantities must be computed from an uncorrelated order parameter time series
    Use 'n_time_steps_sample' to subsamples the time series by selecting every nth time step.
    This is important to calculate averages ignoring autocorrelated data.
    Typically use n_time_steps_sample=20*N (or something proportional to N as a heuristic).

    Parameters
    ----------
    d : dict
        Dictionary containing simulation or experimental data. Must include:
            d[param_name_control]     : float
                Control parameter value for the measurement (e.g., temperature, coupling strength).
            d[param_name_systemsize]  : int or float
                System size (e.g., total number of particles, lattice size to the dimension power: L^D ).
            d[param_name_orderpar]    : array-like
                Time series of the order parameter (raw; may be correlated).

    param_name_control : str
        Key name in `d` corresponding to the control parameter.

    n_time_steps_sample : int, optional
        If provided, subsamples the time series by selecting every nth time step.
        This is important to calculate averages ignoring autocorrelated data.
        Typically use 20*N as a heuristic.

    calc_suscept_bootstrap_error : bool, default False
        If True, estimates the error in susceptibility using bootstrap resampling.

    param_name_orderpar : str, default 'rho'
        Key name in `d` corresponding to the order parameter time series.

    param_name_systemsize : str, default 'N'
        Key name in `d` corresponding to the system size.

    **bootstrap_args : dict
        Additional keyword arguments passed to the bootstrap function.

    Returns
    -------
    param : float
        Value of the control parameter.

    order_param_mean : float
        Mean of the order parameter.

    order_param_stddev : float
        Standard deviation (error estimate) of the order parameter.

    susceptibility : float
        Susceptibility, computed as N * Var(order_param).

    susceptibility_stddev : float
        Error estimate of the susceptibility (NaN if bootstrap is not used).

    n_time_steps_sample, n_data
        Returns these two parameters for debugging
        n_time_steps_sample -> sampling period of order_param time series
        n_data              -> number of data points from order_param time series used for calculating averages
    """
    n_data   = len(d[param_name_orderpar])
    get_data = lambda X: X
    if misc.exists(n_time_steps_sample):
        #ind                     = numpy.linspace(0,n_data-1,n_time_steps_sample).astype(int)
        ind                     = slice(0,n_data,n_time_steps_sample)
        get_data                = lambda X: X[ind]
        k_start, k_stop, k_step = ind.indices(n_data)
        n_data                  = (k_stop - k_start + k_step - 1) // k_step
        #print(f' using {n_data} points for average [[ rho.size == {len(d[param_name_orderpar])} ]]')
    if calc_suscept_bootstrap_error:
        #sus_std = d[param_name_systemsize] * misc.bootstrap_func(d[param_name_orderpar],numpy.nanstd,return_bs_confint_se=True,**bootstrap_args)[2]
        resample_size  = rr if (rr:=int(max(n_data/100,1000))) < n_data else n_data
        bootstrap_args = misc._get_kwargs(bootstrap_args, n_resamples=10, resample_size=resample_size)
        sus_std        = d[param_name_systemsize] * numpy.nanstd(misc.bootstrap_with_resample_size(get_data(d[param_name_orderpar]),misc.my_stddev,**bootstrap_args))
    else:
        sus_std        = numpy.nan
    var_rho = numpy.nanvar(get_data(d[param_name_orderpar]))
    return d[param_name_control], numpy.nanmean(get_data(d[param_name_orderpar])), numpy.sqrt(var_rho), float(d[param_name_systemsize]) * var_rho, sus_std, n_time_steps_sample, n_data

@njit(types.float64[:](types.float64[:],types.int64))
def bin_time_series(X,dt):
    """
    distributes X[t] into consecutive bins of size dt (in units of index of X),
    such that
    Xr[k] = (1/Mab) * (sum of X[a:b])
    returns
        Xr -> binned X
    """
    nbins = X.size // dt
    Xr    = numpy.zeros(nbins,dtype=numpy.float64)
    t     = numpy.linspace(0,X.size,nbins+1).astype(numpy.int64)
    for k,(a,b) in enumerate(zip(t[:-1],t[1:])):
        Xr[k] = float(numpy.sum(X[a:b])) / float(b - a)
    return Xr

def select_timerange_from_data(d,k1=None,k2=None):
    """
    the data in d is trimmed to convey only the index ranges (inclusive) [0:k1] and [time.size-k2:end]
    the variables X_ind, X_values, X_time, time and rho are affected
    returns d with these variables trimmed
    """
    has_k1 = not(type(k1) is type(None))
    has_k2 = not(type(k2) is type(None))
    if not has_k1 and not has_k2:
        return d
    if has_k1 and has_k2:
        t1,t2 = d.time[k1-1], d.time[-k2]
    k2            = d.time.size - k2 if has_k2 else k2
    idx           = misc._select_timerange_build_index(d.time.size, k1=k1, k2=k2)
    d.time, d.rho = misc._select_timerange_apply_index([d.time, d.rho], idx)
    if d.X_time.size > 0:
        if has_k1 and not has_k2:
            n1 = misc.find_last(d.X_time,d.time[-1] ) + 1 #if has_k1 else None # finds the last occurrence of time[0]
            n2 = None #if has_k2 else None # finds the first occurrence of time[-1]
        if has_k2 and not has_k1:
            n1 = None #if has_k1 else None # finds the last occurrence of time[0]
            n2 = misc.find_first(d.X_time,d.time[0]) #if has_k2 else None # finds the first occurrence of time[-1]
        if has_k1 and has_k2:
            n1 =  misc.find_last(d.X_time,t1)+1
            n2 = misc.find_first(d.X_time,t2)
        #print(f'n1={n1}, n2={n2}')
        #print(f'time[0]={d.time[0]}, time[-1]={d.time[-1]}')
        idx_X = misc._select_timerange_build_index(d.X_time.size, n1, n2)
        d.X_time,d.X_ind,d.X_values = misc._select_timerange_apply_index([d.X_time,d.X_ind,d.X_values], idx_X)
    return d

def load_phasetrans_file(fname):
    d       = scipy.io.loadmat(fname,squeeze_me=True)
    pt_data = [io.recarray_to_structtype(dd,convert_to_structarray=False) for dd in d['data']]
    if 'd_files' in d:
        d_files = [([io.recarray_to_structtype(ddd,convert_to_structarray=False) for ddd in dd] if len(dd)>0 else []) for dd in d['d_files']]
    else:
        d_files = [[] for _ in d['N']]
    return d['N'],pt_data,d_files


def save_phasetrans_file(fname,N_values,pt_data,d_files):
    if os.path.isfile(fname):
        os.remove(fname)
        print(f' ::: WARNING ::: Replacing file ... {fname}')
    scipy.io.savemat(fname,{'N':N_values,'data':io.list_of_arr_to_arr_of_obj(pt_data ),'d_files':io.list_of_arr_to_arr_of_obj(d_files)})


def _merge_data_phasetrans_struct(s1,s2,f,ind,param_name_control):
    get_data = lambda s: s[f] if f in s else numpy.full(s[param_name_control].shape,numpy.nan)
    return numpy.concatenate((get_data(s1).flatten(),get_data(s2).flatten()))[ind]


def merge_phasetrans_params_struct(s1,s2,param_name_control):
    if type(s1) is list:
        assert type(s2) is list, 'if s1 is a list, s2 must also be a list (if lengths do not match, only merge up to smallest length between s1 and s2)'
        return [ merge_phasetrans_params_struct(ss1,ss2,param_name_control) for ss1,ss2 in zip(s1,s2) ]
    f1     = s1.keys()
    f2     = s2.keys()
    fields = set(f1) | set(f2)
    #assert fields == set(f1), 'both input structs must have the same fields'
    assert param_name_control in fields, f'"{param_name_control}" must be a field in both input structs'
    fields.remove(param_name_control)
    p,ind = numpy.unique(numpy.concatenate((s1[param_name_control].flatten(),s2[param_name_control].flatten())), return_index=True)
    d     = {param_name_control:p}
    d.update({f:_merge_data_phasetrans_struct(s1,s2,f,ind,param_name_control) for f in fields})
    return misc.structtype(**d)


def calc_mean_phasetrans_params(pt_lst,param_name_control='',param_name_orderpar='',return_as_struct=True):
    """
    pt_lst = list of structtype where each pt_list[i]
                contains a phase transition table returned by calc_phassetrans_params_struct with return_as_struct==False
                all phase transitions structs in the list must share the same param_name_control grid
                pt_lst[i][:,0] -> param_name_control        
                pt_lst[i][:,1] -> param_name_orderpar       
                pt_lst[i][:,2] -> param_name_orderpar+'_std'
                pt_lst[i][:,3] -> 'suscept'                 
                pt_lst[i][:,4] -> 'suscept_std'             
                pt_lst[i][:,5] -> 'n_time_steps_sample'     
                pt_lst[i][:,6] -> 'n_data'                  
    """
    if not (type(pt_lst) is list):
        pt_lst = [pt_lst]
    if return_as_struct or (type(pt_lst[0]) is misc.structtype):
        assert misc.nonempty_str(param_name_control) and misc.nonempty_str(param_name_orderpar), 'to return a struct or to input an array of structs, you must provide a param_name_orderpar and a param_name_control for the conversion from structtype to matrix'
    if type(pt_lst[0]) is misc.structtype:
        #raise ValueError('Each input must be a phase transition table returned by calc_phasetrans_params_struct with return_as_struct==False')
        pt_lst = [convert_phasetrans_struct_to_matrix(pt,param_name_control,param_name_orderpar) for pt in pt_lst]
    pt_lst = numpy.atleast_3d(pt_lst)
    assert all(numpy.all(pt[:,0]==pt_lst[0,:,0]) for pt in pt_lst), 'all phase transition tables in pt_lst must share the same control parameter grid pt_list[i][:,0]'
    measure_cols        = [1,3] # # order param and susceptibility
    error_cols          = [2,4] # # order param and susceptibility errors
    extra_cols          = [5,6] # # other data cols in phasetrans matrix
    res                 = numpy.full(pt_lst[0].shape, numpy.nan, dtype=float)
    res[:,0           ] = pt_lst[0,:,0] # control parameter
    res[:,measure_cols] = numpy.nanmean(pt_lst[:,:,measure_cols],axis=0) # order param and susceptibility
    res[:,error_cols  ] = numpy.nanmax( pt_lst[:,:,error_cols  ],axis=0) # order param and susceptibility errors
    res[:,extra_cols  ] = numpy.nanmin( pt_lst[:,:,extra_cols  ],axis=0) # other data cols in phasetrans matrix
    if return_as_struct:
        return convert_phasetrans_matrix_to_struct(res,param_name_control,param_name_orderpar)
    else:
        return res

def convert_phasetrans_struct_to_matrix(pt_struct,param_name_control,param_name_orderpar):
    phasetrans_data      = numpy.full((pt_struct[param_name_control].size,7),numpy.nan)
    phasetrans_data[:,0] = pt_struct[param_name_control        ]
    phasetrans_data[:,1] = pt_struct[param_name_orderpar       ]
    phasetrans_data[:,2] = pt_struct[param_name_orderpar+'_std']
    phasetrans_data[:,3] = pt_struct['suscept'                 ]
    phasetrans_data[:,4] = pt_struct['suscept_std'             ]
    phasetrans_data[:,5] = pt_struct['n_time_steps_sample'     ]
    phasetrans_data[:,6] = pt_struct['n_data'                  ]
    return phasetrans_data

def convert_phasetrans_matrix_to_struct(phasetrans_data,param_name_control,param_name_orderpar):
    res = misc.structtype()
    res[param_name_control        ] = phasetrans_data[:,0]
    res[param_name_orderpar       ] = phasetrans_data[:,1]
    res[param_name_orderpar+'_std'] = phasetrans_data[:,2]
    res['suscept'                 ] = phasetrans_data[:,3]
    res['suscept_std'             ] = phasetrans_data[:,4]
    res['n_time_steps_sample'     ] = phasetrans_data[:,5]
    res['n_data'                  ] = phasetrans_data[:,6]
    return res

def cutoff_transient(d,orderpar_field,t_trans,time_field='time',dt_field='dt'):
    if not misc.exists(t_trans):
        return d
    if type(d) is list:
        return [ cutoff_transient(d,orderpar_field,t_trans,time_field=time_field,dt_field=dt_field) for dd in d ]
    assert all(f in d for f in [orderpar_field,time_field,dt_field]), 'check fields in data variable, some were not found'
    d[orderpar_field] = d[orderpar_field][d[time_field]*d[dt_field]>=t_trans]
    d[time_field]     = d[time_field][    d[time_field]*d[dt_field]>=t_trans]
    return d

def calc_phasetrans_params_struct(
    f_list,
    param_name_control,
    time_k1=None,
    time_k2=None,
    t_trans=None,
    n_time_steps_sample=None,
    return_file_data=False,
    param_name_orderpar='rho',
    param_name_systemsize='N',
    param_name_time='time',
    param_name_dt='dt',
    calc_suscept_bootstrap_error=False,
    return_as_struct=True,
    **bootstrap_args
):
    """
    Computes phase transition parameters from a list of .mat files containing time series data.
    Cuts off the first t_trans time steps (in units of d.dt) from the time series before averaging [[ i.e. cuts off from d.time[0]*d.dt to d.time[k]*d.dt==t_trans ]]
    Phase transition quantities must be computed from an uncorrelated order parameter time series
    Use 'n_time_steps_sample' to subsamples the time series by selecting every nth time step.
    This is important to calculate averages ignoring autocorrelated data.
    Typically use n_time_steps_sample=20*N (or something proportional to N as a heuristic).

    For each file, calculates:
        - Control parameter value
        - Mean of the order parameter
        - Standard deviation of the order parameter
        - Susceptibility (defined as N * Var(order parameter))
        - Susceptibility error (optional, via bootstrap)

    Parameters
    ----------
    f_list : list of str
        List of file paths to .mat files containing simulation or experimental data.

    param_name_control : str
        Key name for the control parameter (e.g., temperature, coupling strength).

    time_k1 : float, optional
        Start time for trimming the time series data (used only if `return_file_data=True`).

    time_k2 : float, optional
        End time for trimming the time series data (used only if `return_file_data=True`).

    n_time_steps_sample : int, optional
        If provided, subsamples the time series by selecting every nth time step.

    return_file_data : bool, default False
        If True, returns the trimmed data from each file in addition to the computed results.

    param_name_orderpar : str, default 'rho'
        Key name for the order parameter time series.

    param_name_systemsize : str, default 'N'
        Key name for the system size.

    param_name_time : str, default 'time'
        Key name for the time variable in the data.

    calc_suscept_bootstrap_error : bool, default False
        If True, estimates the error in susceptibility using bootstrap resampling.

    **bootstrap_args : dict
        Additional keyword arguments passed to the bootstrap function.

    Returns
    -------
    res : structtype if return_as_struct==True
        Structured object containing:
            res[param_name_control]         : array of control parameter values
            res[param_name_orderpar]        : array of order parameter means
            res[param_name_orderpar+'_std'] : array of order parameter standard deviations
            res['suscept']                  : array of susceptibility values
            res['suscept_std']              : array of susceptibility errors
            res['n_time_steps_sample']      : number of time steps between samples of the order parameter
            res['n_data']                   : number of data points resulting from selecting the order parameter sampled with the interval n_time_steps_sample

    d_trim : list
        List of trimmed data dictionaries (only returned if `return_file_data=True`).
    """
    # returns lambda, <rho>, and susceptibility = N*var(rho)
    if return_file_data:
        variable_names    = None
        return_structtype = True
    else:
        variable_names    = [param_name_control,param_name_systemsize,param_name_time,param_name_orderpar]
        return_structtype = False
    phasetrans_data = numpy.empty((len(f_list),7),dtype=float)
    d_trim          = []
    progress_bar = tqdm(enumerate(f_list), desc='Processing files...',total=len(f_list))
    for k,f in progress_bar: # for each input file
        progress_bar.set_postfix_str(f)
        d                    = io.import_mat_file(f,variable_names=variable_names,return_structtype=return_structtype)
        if return_file_data:
            d_trim.append(select_timerange_from_data(d, time_k1, time_k2))
        phasetrans_data[k,:] = calc_phasetrans_params(cutoff_transient(d,param_name_orderpar,t_trans,param_name_time,param_name_dt),param_name_control,n_time_steps_sample=n_time_steps_sample,calc_suscept_bootstrap_error=calc_suscept_bootstrap_error,**bootstrap_args)
    progress_bar.close()
    ind             = numpy.argsort(phasetrans_data[:,0])  # sorting according to param value
    phasetrans_data = phasetrans_data[ind,:]
    if return_as_struct:
        return convert_phasetrans_matrix_to_struct(phasetrans_data,param_name_control,param_name_orderpar), d_trim
    else:
        return phasetrans_data, d_trim
