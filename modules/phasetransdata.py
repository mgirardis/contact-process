import re
import numpy
import modules.io as io

def get_par_value_from_str(str_list,par_name):
    if type(str_list) is str:
        return int(match.group(1)) if (match := re.search(par_name + r'(\d+(?:\.\d+)?)', str_list)) else -1.0
    else:
        return [get_par_value_from_str(idl,par_name) for idl in str_list]

def calc_phasetrans_params(d):
    return d.l, numpy.nanmean(d.rho), float(d.N) * numpy.nanvar(d.rho)

def _select_timerange_build_index(n: int, k1=None, k2=None):
    """
    Build an index array selecting [:k1] and [k2:] from an array of length n.
    """
    if k1 is None and k2 is None:
        return numpy.arange(n)
    if k1 is None:
        return numpy.arange(k2, n)
    if k2 is None:
        return numpy.arange(0, k1)
    return numpy.r_[0:k1, k2:n]

def _select_timerange_apply_index(arrays, idx):
    """
    Apply precomputed indices idx to a list of arrays.
    Always returns contiguous arrays.
    """
    return [numpy.ascontiguousarray(a[idx]) for a in arrays]

def find_first(X,v):
    k = numpy.argmax(X==v)
    return k if X[k]==v else None

def find_last(X,v):
    k = X.size - 1 - numpy.argmax(X[::-1]==v)
    return k if X[k]==v else None

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
    idx           = _select_timerange_build_index(d.time.size, k1=k1, k2=k2)
    d.time, d.rho = _select_timerange_apply_index([d.time, d.rho], idx)
    if d.X_time.size > 0:
        if has_k1 and not has_k2:
            n1 = find_last(d.X_time,d.time[-1] ) + 1 #if has_k1 else None # finds the last occurrence of time[0]
            n2 = None #if has_k2 else None # finds the first occurrence of time[-1]
        if has_k2 and not has_k1:
            n1 = None #if has_k1 else None # finds the last occurrence of time[0]
            n2 = find_first(d.X_time,d.time[0]) #if has_k2 else None # finds the first occurrence of time[-1]
        if has_k1 and has_k2:
            n1 =  find_last(d.X_time,t1)+1
            n2 = find_first(d.X_time,t2)
        #print(f'n1={n1}, n2={n2}')
        #print(f'time[0]={d.time[0]}, time[-1]={d.time[-1]}')
        idx_X = _select_timerange_build_index(d.X_time.size, n1, n2)
        d.X_time,d.X_ind,d.X_values = _select_timerange_apply_index([d.X_time,d.X_ind,d.X_values], idx_X)
    return d

def calc_phasetrans_params_struct(f_list,k1=None,k2=None,return_file_data=False):
    # returns lambda, <rho>, and susceptibility = N*var(rho)
    phasetrans_data = []
    d_trim          = []
    for f in f_list:
        d = io.import_mat_file(f)
        phasetrans_data.append(calc_phasetrans_params(d))
        if return_file_data:
            d_trim.append(select_timerange_from_data(d, k1, k2))
    A = numpy.asarray(sorted(phasetrans_data, key=lambda dd:dd[0])) # sorting according to l value
    return io.structtype(l=A[:,0],rho=A[:,1],suscept=A[:,2]), d_trim

def unpack_list_of_tuples(lst):
    """
    given a list of len m where element is an n-tuple, then returns an n-tuple where each element containins m elements
    (similar to numpy.transpose)

    this is useful to unpack a list comprehension applied to a function that has multiple returns
    """
    return tuple( list(x) for x in zip(*lst) )
