import numpy
import scipy.io
import modules.io as io
import modules.misc_func as misc
from tqdm.notebook import tqdm

def calc_phasetrans_params(d,param_name_control,n_time_points=None,calc_suscept_bootstrap_error=False,param_name_orderpar='rho',param_name_systemsize='N',**bootstrap_args):
    """
    calculates order parameter and susceptibility from time series data in d
    d must contain the variables:
        d[param_name]            -> parameter value for the particular measurement of the order parameter
        d[param_name_systemsize] -> system size
        d[param_name_orderpar]   -> time series of the order parameter (raw; does not need to be uncorrelated)
    returns, in this order:
        param                 -> parameter value where the order parameter is measured
        order_param           -> order parameter mean
        order_param stddev    -> order parameter error
        susceptibility        -> N * Var(order_param)
        susceptibility stddev -> susceptbility error
    """
    n_data   = len(d[param_name_orderpar])
    get_data = lambda X: X
    if misc.exists(n_time_points) and (n_time_points < n_data):
        ind      = numpy.linspace(0,n_data-1,n_time_points).astype(int)
        get_data = lambda X: X[ind]
        n_data   = n_time_points
    if calc_suscept_bootstrap_error:
        #sus_std = d[param_name_systemsize] * misc.bootstrap_func(d[param_name_orderpar],numpy.nanstd,return_bs_confint_se=True,**bootstrap_args)[2]
        bootstrap_args = misc._get_kwargs(bootstrap_args, n_resamples=10, resample_size=int(max(n_data/100,1000)))
        sus_std        = d[param_name_systemsize] * numpy.nanstd(misc.bootstrap_with_resample_size(get_data(d[param_name_orderpar]),misc.my_stddev,**bootstrap_args))
    else:
        sus_std        = numpy.nan
    var_rho = numpy.nanvar(get_data(d[param_name_orderpar]))
    return d[param_name_control], numpy.nanmean(get_data(d[param_name_orderpar])), numpy.sqrt(var_rho), float(d[param_name_systemsize]) * var_rho, sus_std

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
    scipy.io.savemat(fname,{'N':N_values,'data':io.list_of_arr_to_arr_of_obj(pt_data ),'d_files':io.list_of_arr_to_arr_of_obj(d_files)})

def merge_phasetrans_params_struct(s1,s2,param_name_control):
    if type(s1) is list:
        assert type(s2) is list, 'if s1 is a list, s2 must also be a list (if lengths do not match, only merge up to smallest length between s1 and s2)'
        return [ merge_phasetrans_params_struct(ss1,ss2,param_name_control) for ss1,ss2 in zip(s1,s2) ]
    f1     = s1.keys()
    f2     = s2.keys()
    fields = set(f1) & set(f2)
    assert fields == set(f1), 'both input structs must have the same fields'
    fields.remove(param_name_control)
    p,ind = numpy.unique(numpy.concatenate((s1[param_name_control].flatten(),s2[param_name_control].flatten())), return_index=True)
    d     = {param_name_control:p}
    d.update({f:numpy.concatenate((s1[f],s2[f]))[ind] for f in fields})
    return misc.structtype(**d)


def calc_phasetrans_params_struct(f_list,param_name_control,time_k1=None,time_k2=None,n_time_points=None,return_file_data=False,param_name_orderpar='rho',param_name_systemsize='N',param_name_time='time',calc_suscept_bootstrap_error=False,**bootstrap_args):
    # returns lambda, <rho>, and susceptibility = N*var(rho)
    if return_file_data:
        variable_names    = None
        return_structtype = True
    else:
        variable_names    = [param_name_control,param_name_systemsize,param_name_time,param_name_orderpar]
        return_structtype = False
    phasetrans_data = numpy.empty((len(f_list),5),dtype=float)
    d_trim          = []
    progress_bar = tqdm(enumerate(f_list), desc='Processing files...',total=len(f_list))
    for k,f in progress_bar:
        progress_bar.set_postfix_str(f)
        d                    = io.import_mat_file(f,variable_names=variable_names,return_structtype=return_structtype)
        phasetrans_data[k,:] = calc_phasetrans_params(d,param_name_control,n_time_points=n_time_points,calc_suscept_bootstrap_error=calc_suscept_bootstrap_error,**bootstrap_args)
        if return_file_data:
            d_trim.append(select_timerange_from_data(d, time_k1, time_k2))
    progress_bar.close()
    ind             = numpy.argsort(phasetrans_data[:,0])  # sorting according to param value
    phasetrans_data = phasetrans_data[ind,:]
    res = misc.structtype()
    res[param_name_control        ] = phasetrans_data[:,0]
    res[param_name_orderpar       ] = phasetrans_data[:,1]
    res[param_name_orderpar+'_std'] = phasetrans_data[:,2]
    res['suscept'                 ] = phasetrans_data[:,3]
    res['suscept_std'             ] = phasetrans_data[:,4]
    return res, d_trim
