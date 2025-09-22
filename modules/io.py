# -*- coding: utf-8 -*-
import os
import io
import copy
import numpy
import scipy.io
import contextlib
import modules.misc_func as misc

def add_simulation_parameters(parser):
    parser.add_argument('-l'           , nargs=1, required=False, metavar='l_PARAM'         , type=float , default=[1.1]            , help='CP rate (l_c=3.297848 for ring, iterdynamics=md; l_c=2 for ring, iterdynamics=to; l_c=1 for mean-field)')
    parser.add_argument('-N'           , nargs=1, required=False, metavar='N_PARAM'         , type=int   , default=[10000]          , help='number of sites in the network')
    parser.add_argument('-M'           , nargs=1, required=False, metavar='M_PARAM'         , type=int   , default=[100]            , help='number of memory time steps for quasistationary simulation (whenever the system goes into absorbing, it is placed in a random state chosen from the last M visited states)')
    parser.add_argument('-X0'          , nargs=1, required=False, metavar='X0_PARAM'        , type=int   , default=[1]              , help='IC to each site (a scalar 0 or 1)')
    parser.add_argument('-fX0'         , nargs=1, required=False, metavar='fX0_PARAM'       , type=float , default=[0.5]            , help='fraction of sites assigned to X0 as IC (remaining are zero)')
    parser.add_argument('-dtsample'    , nargs=1, required=False, metavar='dtsample_PARAM'  , type=int   , default=[1]              , help='sampling interval (in units of dt=1/N if expandtime is set; this is ignored when update == parallel).\n\tWe only sample the system in time steps that are a multiple of dtsample.\n\tThis reduces dramatically the saved data in sequential update if dtsample=N.\n\tFor example,\n\t\tif expandtime==True:\n\t\t\t* dt=1/N;\n\t\t\t* sampling interval s = dtsample;\n\t\t\t* running time T = tTotal * N\n\t\t\t* simulation time t = 1 to T,\n\t\t\t* sampling when t%%s == 0 (i.e., t is a multiple of dtsample)\n\t\t\t\tif dtsample = 1: sample dt=1/N time interval (every time step)\n\t\t\t\tif dtsample = N: sample every N*dt = 1 Monte Carlo step (in practice, this means N time steps)\n\t\tif expandtime==False:\n\t\t\t* dt = 1\n\t\t\t* sampling interval s = dtsample;\n\t\t\t* running time T = tTotal;\n\t\t\t* simulation time t = 1 to T\n\t\t\t* sampling when t%%s == 0 (i.e., t is a multiple of dtsample)\n\t\t\t\tif dtsample == 1: sample every time step\n\t\t\t\tif dtsample == N: sample every Monte Carlo step (i.e., N time steps)')
    parser.add_argument('-tTotal'      , nargs=1, required=False, metavar='tTotal_PARAM'    , type=int   , default=[10000]          , help='total number of time steps. This becomes tTotal/dt if expandtime == True')
    parser.add_argument('-tTrans'      , nargs=1, required=False, metavar='tTrans_PARAM'    , type=int   , default=[0]              , help='number of transient time steps. This becomes tTrans/dt if expandtime == True')
    parser.add_argument('-outputFile'  , nargs=1, required=False, metavar='OUTPUT_FILE_NAME', type=str   , default=['cp.mat']       , help='name of the output file')
    parser.add_argument('-graph'       , nargs=1, required=False, metavar='GRAPH_TYPE'      , type=str   , default=['ring']         , choices=['mf', 'alltoall', 'ring', 'ringfree']        , help='mf,alltoall -> mean-field simulation; ring -> 1+1 simulation with periodic boundary conditions; ringfree -> ring with free boundaries')
    parser.add_argument('-iterdynamics', nargs=1, required=False, metavar='ITER_TYPE'       , type=str   , default=['marro_dickman'], choices=['marro_dickman', 'tome_oliveira', 'md', 'to'], help='marro_dickman,md -> described in pg 178pdf/162book Marro & Dickman book; tome_oliveira,to -> described in pg 308pdf Tome & Oliveira book before eq 13.6')
    parser.add_argument('-update'      , nargs=1, required=False, metavar='UPDATE_TYPE'     , type=str   , default=['seq']          , choices=['seq','sequential','par','parallel']         , help='seq -> standard update scheme: 1 particle update/ts (paragraph after eq 3.35 in Henkel book); par -> parallel update (attempts to update all sites at each ts, matches the E/I network)')
    parser.add_argument('-sim'         , nargs=1, required=False, metavar='SIM_TYPE'        , type=str   , default=['timeevo']      , choices=['timeevo', 'aval']                           , help='timeevo -> simple time evolution simulation (quasistatic if M > 0); aval -> avalanche simulation; seeds 1 site every time activity dies out')
    parser.add_argument('-noX0Rand'    , required=False, action='store_true', default=False, help='if set, Xi is generated sequentially')
    parser.add_argument('-saveSites'   , required=False, action='store_true', default=False, help='if set, saves the Xi variable for every site (may consume a lot of memory!)')
    parser.add_argument('-writeOnRun'  , required=False, action='store_true', default=False, help='if set, writes the Xi variables to an output *_spk.txt file during the main time loop (needs -saveSites to be set, and avoids memory errors at the expense of a slower simulation)')
    parser.add_argument('-mergespkfile', required=False, action='store_true', default=False, help='if set, tries to merge the output *.mat with the output *_spk.txt file (if it exists) into a single *.mat file, and removes the *_spk.txt file. It requires both -saveSites and -writeOnRun to be set. WARNING: for some unknown reason, sometimes the merging is not successful.')
    parser.add_argument('-expandtime'  , required=False, action='store_true', default=False, help='if set, then uses the dt=1/N to expand the total simulation time: tTotal_eff = tTotal / dt (only for sequential update)')
    return parser

def get_help_string(parser):
    help_buffer = io.StringIO()
    with contextlib.redirect_stdout(help_buffer):
        parser.print_help()
    return help_buffer.getvalue()

def import_mat_file(path,variable_names=None,return_structtype=True):
    d = scipy.io.loadmat(path, squeeze_me=True, variable_names=variable_names)
    if not return_structtype:
        return d
    return misc.structtype(**{ k:v for k,v in d.items() if ((not k.startswith('__')) and (not k.endswith('__'))) })

def import_spk_file(fname):
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File {fname} does not exist.")
    X_values, X_ind, X_time = [], [], []
    with open(fname,'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            t, k, X = line.split(',')
            X_values.append(float(X))
            X_ind.append(int(k))
            X_time.append(float(t))
    return X_values, X_ind, X_time

def import_spk_file_no_error(spkFileName,remove_spk_file=True,verbose=True):
    X_values, X_ind, X_time = [], [], []
    try:
        X_values, X_ind, X_time = import_spk_file(spkFileName)
        if remove_spk_file:
            os.remove(spkFileName)
        if verbose:
            print(f'  ... spk file {spkFileName} was written and loaded' + (' and removed' if remove_spk_file else '') + ' successfully')
    except FileNotFoundError:
        if verbose:
            print(f':::: WARNING :::: spk file {spkFileName} not found')
    return X_values, X_ind, X_time

def _get_spk_file_name(fname_main_output, spkFileName=''):
    fname_spk_output     = os.path.splitext(fname_main_output)[0] + '_spk.txt'
    if not os.path.isfile(fname_spk_output):
        fname_spk_output = os.path.join(os.path.split(fname_main_output)[0],os.path.split(spkFileName)[-1])
    return fname_spk_output

def _get_merged_file_name(fname_main_output):
    fname, _ = os.path.splitext(fname_main_output)
    return fname + '_merged.mat'

def merge_simulation_files(fname_main_output, fname_spk_output='', remove_spk_file=True, verbose=True):
    d = import_mat_file(fname_main_output)
    if len(fname_spk_output) == 0:
        fname_spk_output    = _get_spk_file_name(fname_main_output, d.spkFileName)
    if not os.path.isfile(fname_spk_output):
        print(f':::: WARNING :::: spk file {fname_spk_output} not found, cannot merge files ...')
        print('                   This program uses numba and cannot write txt files,')
        print('                   so the output was possibly written to the stdout instead.')
        return
    if verbose:
        print(f'* Merging files: {fname_main_output} and {fname_spk_output}')
    X_values, X_ind, X_time = import_spk_file_no_error(fname_spk_output,remove_spk_file=remove_spk_file,verbose=verbose)
    d.X_values              = X_values
    d.X_ind                 = X_ind
    d.X_time                = X_time
    d.writeOnRun            = False
    d.spkFileName           = ''
    fname_merged            = _get_merged_file_name(fname_main_output)
    scipy.io.savemat(fname_merged,dict(**d),long_field_names=True,do_compression=True)
    if verbose:
        print(f'  ... merged file saved as {fname_merged}')

def save_simulation_file(argv, args, rho, time, X_data):
    # X_data[i,:] = [t,k,X]
    if not (type(X_data) is numpy.ndarray):
        X_data = numpy.array(X_data, dtype=float)
        if X_data.shape == (0,):
            X_data = X_data.reshape((0,3))
    args.sim          = args.sim.name.lower()          if hasattr(args.sim         ,'name') else str(args.sim         ).lower()
    args.graph        = args.graph.name.lower()        if hasattr(args.graph       ,'name') else str(args.graph       ).lower()
    args.update       = args.update.name.lower()       if hasattr(args.update      ,'name') else str(args.update      ).lower()
    args.iterdynamics = args.iterdynamics.name.lower() if hasattr(args.iterdynamics,'name') else str(args.iterdynamics).lower()
    scipy.io.savemat(args.outputFile,dict(cmd_line=' '.join(argv),time=time,rho=rho, X_values=X_data[:,2], X_ind=X_data[:,1], X_time=X_data[:,0],**args),long_field_names=True,do_compression=True)
    print(f'  ... simulation file saved ::: {args.outputFile}')

def get_output_filename(path):
    fname,fext = os.path.splitext(path)
    if fext.lower() != '.mat':
        return fname + '.mat'
        #if fext == '.txt':
        #    return fname + '.mat'
        #else:
        #    return fname + fext + '.mat'
    return path

def keep_keys(d,keys_to_keep):
    c = d.copy()
    for k in d:
        if k not in keys_to_keep:
            c.pop(k,None)
    return c

def get_func_param(f):
    f_code = f.__code__
    args = f_code.co_varnames[:f_code.co_argcount + f_code.co_kwonlyargcount]
    return [ a for a in args if a != 'self']

def fix_args_lists_as_scalars(args):
    if type(args) is dict:
        a = args
    else:
        a = args.__dict__
    for k,v in a.items():
        if (type(v) is list) and (len(v) == 1):
            a[k] = v[0]
    if type(args) is dict:
        args = a
    else:
        args.__dict__ = a
    return args

def get_new_file_name(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.isfile(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    return path

def fill_with_nan(s):
    n = max(len(a) for a in s)
    return [ numpy.concatenate((a.flatten(),numpy.full(n-a.size, numpy.nan))) for a in s ]

def structtype_to_recarray(s):
    if isinstance(s,list) or isinstance(s,tuple):
        s_new = misc.structtype(struct_fields=list(s[0].keys()),field_values=[ None for _ in s[0].keys() ])
        for k in s[0].keys():
            s_new[k] = list_of_arr_to_arr_of_obj([ v[k] for v in s ])
        return struct_array_for_scipy(s_new.GetFields(','),*s_new.values())
    return struct_array_for_scipy(s.GetFields(','),*[list_of_arr_to_arr_of_obj([v]) for v in s.values()])

def recarray_to_structtype(r,convert_to_structarray=True):
    unpack_array = lambda a: a if numpy.isscalar(a) else (a.item(0) if a.size==1 else a)
    s = misc.structtype(**{ field : unpack_array(r[field]) for field in r.dtype.names })
    if convert_to_structarray:
        try:
            if len(r) > 1:
                return convert_struct_of_arrays_to_structarray(s)
        except TypeError as e:
            if 'len() of unsized object' in str(e):
                print(' ::: WARNING ::: Cannot convert to structarray... Check len(r)')
            else:
                raise e
    return s

def convert_struct_of_arrays_to_structarray(s):
    fields = s.GetFields(',').split(',')
    N      = len(s[fields[0]])
    if not all(len(s[f]) == N for f in fields):
        raise ValueError('all fields in s must have the same length')
    if N > 1:
        s_structarray = [ misc.structtype(**{k:None for k in s.keys()}) for _ in range(N) ]
        for i in range(N):
            for k in s.keys():
                s_structarray[i][k] = s[k][i]
        return s_structarray
    else:
        return s

def struct_array_for_scipy(field_names,*fields_data):
    """
    returns a data structure which savemat in scipy.io interprets as a MATLAB struct array
    the order of field_names must match the order in which the remaining arguments are passed to this function
    such that
    s(j).(field_names(i)) == fields_data[i][j], identified by field_names[i]

    field_names ->  comma-separated string listing the field names;
                        'field1,field2,...' -> field_names(1) == 'field1', etc...
    fields_data ->  each extra argument entry is a list with the data for each field of the struct
                        fields_data[i][j] :: data for field i in the element j of the struct array: s(j).(field_names(i))
    
    returns
        numpy record array S where
        S[field_names[i]][j] == fields_data[i][j]
    """
    fn_list = field_names.split(',')
    assert len(fn_list) == len(fields_data),'you must give one field name for each field data'
    return numpy.core.records.fromarrays([f for f in fields_data],names=fn_list,formats=[object]*len(fn_list))

def list_of_arr_to_arr_of_obj(X):
    n = len(X)
    Y = numpy.empty((n,),dtype=object)
    for i,x in enumerate(X):
        Y[i] = x
    return Y


def namespace_to_structtype(a):
    return misc.structtype(**fix_args_lists_as_scalars(copy.deepcopy(a.__dict__)))

#def get_write_spike_data_functions(saveSites,writeOnRun):
#    if saveSites:
#        if writeOnRun:
#            write_spk_time = write_spk_data #lambda t_ind,k_ind: spkTimeFile.write(str(t_ind) + ',' + str(k_ind) + '\n')
#            save_spk_time  = save_spk_data_fake
#        else:
#            write_spk_time = write_spk_data_fake
#            save_spk_time  = save_spk_data
#    else:
#        save_spk_time  = save_spk_data_fake
#        write_spk_time = write_spk_data_fake
#    return write_spk_time,save_spk_time
#
#def save_spk_data_fake(X_values, X_ind, X_time, X, k, t):
#    pass
#
#def save_spk_data(X_values, X_ind, X_time, X, k, t):
#    if X:
#        X_values.append(X)
#        X_ind.append(k)
#        X_time.append(t)
#
#def write_spk_data_fake(spkFile,t,k,X):
#    pass
#
#def write_spk_data(spkFile,t,k,X):
#    if X:
#        spkFile.write(str(t) + ',' + str(k) + ',' + str(X) + '\n')
#    #print(t,',',k)
#
#def open_file(spkFileName,saveSites_and_writeOnRun):
#    if saveSites_and_writeOnRun:
#        spk_file = open(spkFileName,'w')
#        spk_file.write('#t,k,Xk\n') # header
#        print(f'  ... temp spk file opened: {spkFileName}')
#        return spk_file
#    return None
#
#def close_file(spkFile,saveSites_and_writeOnRun):
#    if saveSites_and_writeOnRun:
#        spkFile.close()
#        print('  ... temp spk file closed')
#
