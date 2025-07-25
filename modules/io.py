# -*- coding: utf-8 -*-
import os
import io
import copy
import numpy
import scipy.io
import contextlib
import collections

def add_simulation_parameters(parser):
    parser.add_argument('-l',           nargs=1, required=False, metavar='l_PARAM',      type=float, default=[1.1],   help='CP rate (l_c=3.297848 for ring; l_c=1 for mean-field)')
    parser.add_argument('-N',           nargs=1, required=False, metavar='N_PARAM',      type=int,   default=[10000], help='number of sites in the network')
    parser.add_argument('-M',           nargs=1, required=False, metavar='M_PARAM',      type=int,   default=[100],   help='number of memory time steps for quasistationary simulation (whenever the system goes into absorbing, it is placed in a random state chosen from the last M visited states)')
    parser.add_argument('-X0',          nargs=1, required=False, metavar='X0_PARAM',     type=int,   default=[1],     help='IC to each site (a scalar 0 or 1)')
    parser.add_argument('-fX0',         nargs=1, required=False, metavar='fX0_PARAM',    type=float, default=[0.5],   help='fraction of sites assigned to X0 as IC (remaining are zero)')
    parser.add_argument('-tTotal',      nargs=1, required=False, metavar='tTotal_PARAM', type=int,   default=[10000], help='total number of time steps')
    parser.add_argument('-tTrans',      nargs=1, required=False, metavar='tTrans_PARAM', type=int,   default=[0],     help='number of transient time steps')
    parser.add_argument('-graph',       nargs=1, required=False, metavar='GRAPH_TYPE',   type=str,   default=['ring'],        choices=['alltoall', 'ring', 'ringfree'], help='alltoall -> mean-field simulation; ring -> 1+1 simulation with periodic boundary conditions; ringfree -> ring with free boundaries')
    parser.add_argument('-update',      nargs=1, required=False, metavar='UPDATE_TYPE',  type=str,   default=['seq'],         choices=['seq','sequential','par','parallel'], help='seq -> standard update scheme: 1 particle update/ts (paragraph after eq 3.35 in Henkel book); par -> parallel update (attempts to update all articles at each ts, matches the E/I network)')
    parser.add_argument('-sim',         nargs=1, required=False, metavar='SIM_TYPE',     type=str,   default=['timeevo'],     choices=['timeevo', 'aval'], help='timeevo -> simple time evolution stimulation (quasistatic if M > 0); aval -> avalanche simulation; seeds 1 site every time activity dies out')
    #parser.add_argument('-activation',  nargs=1, required=False, metavar='ACTIV_TYPE',   type=str,   default=['rate'],        choices=['rate', 'prob'], help='rate -> each site is activated if random < l*r (r=frac of act neigh); prob -> each site is activated if random < p*r (p=l/(1+l) and r=frac of act neigh -- seems to yield a wrong l_c, but seems to be the correct one according to books)')
    #parser.add_argument('-algorithm',   nargs=1, required=False, metavar='ALGO_TYPE',    type=str,   default=['tomeoliveira'],choices=['mine', 'tomeoliveira'], help='mine -> my alogirhtm -- seems to be wrong; tomeoliveira -> algorithm describe in pg 308pdf/402 of the Tome & Oliveira book, before eq (13.6)')
    parser.add_argument('-noX0Rand',    required=False, action='store_true', default=False, help='if set, Xi is generated sequentially')
    parser.add_argument('-saveSites',   required=False, action='store_true', default=False, help='if set, saves the Xi variable for every site (may consume a lot of memory!)')
    parser.add_argument('-writeOnRun',  required=False, action='store_true', default=False, help='if set, writes the Xi variables to an output *_spk.txt file during the main time loop (needs -saveSites to be set, and avoids memory errors at the expense of a slower simulation)')
    parser.add_argument('-expandtime',  required=False, action='store_true', default=False, help='if set, then uses the dt=1/N to expand the total simulation time: tTotal_eff = tTotal / dt (only for sequential update)')
    parser.add_argument('-outputFile',  nargs=1, required=False, metavar='OUTPUT_FILE_NAME', type=str, default=['cp.mat'], help='name of the output file')
    return parser

def get_help_string(parser):
    help_buffer = io.StringIO()
    with contextlib.redirect_stdout(help_buffer):
        parser.print_help()
    return help_buffer.getvalue()

def import_mat_file(path):
    return structtype(**{ k:v for k,v in scipy.io.loadmat(path, squeeze_me=True).items() if ((not k.startswith('__')) and (not k.endswith('__'))) })

def import_spk_file(fname):
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File {fname} does not exist.")
    X_values, X_ind, X_time = [], [], []
    with open(fname,'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            t, k, X = line.strip().split(',')
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

def merge_simulation_files(fname_main_output, fname_spk_output='', remove_spk_file=True, verbose=True):
    d = import_mat_file(fname_main_output)
    if len(fname_spk_output) == 0:
        fname_spk_output    = _get_spk_file_name(fname_main_output, d.spkFileName)
    if verbose:
        print(f'Merging files: {fname_main_output} and {fname_spk_output}')
    X_values, X_ind, X_time = import_spk_file_no_error(fname_spk_output,remove_spk_file=remove_spk_file,verbose=verbose)
    d.X_values              = X_values
    d.X_ind                 = X_ind
    d.X_time                = X_time
    d.writeOnRun            = False
    d.spkFileName           = ''
    scipy.io.savemat(fname_main_output,dict(**d),long_field_names=True,do_compression=True)
    if verbose:
        print(f'  ... merged file saved as {fname_main_output}')

def save_simulation_file(argv, args, rho, X_values, X_ind, X_time, remove_spk_file=True,verbose=True):
    if args.saveSites and args.writeOnRun:
        X_values, X_ind, X_time = import_spk_file_no_error(args.spkFileName,remove_spk_file=remove_spk_file,verbose=verbose)
    if len(X_time)>0:
        X_time = numpy.asarray(X_time) * args.dt
    scipy.io.savemat(args.outputFile,dict(cmd_line=' '.join(argv),time=numpy.arange(len(rho))*args.dt,rho=rho, X_values=X_values, X_ind=X_ind, X_time=X_time,**args),long_field_names=True,do_compression=True)

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

def namespace_to_structtype(a):
    return structtype(**fix_args_lists_as_scalars(copy.deepcopy(a.__dict__)))

class structtype(collections.abc.MutableMapping):
    def __init__(self,struct_fields=None,field_values=None,**kwargs):
        if not(type(struct_fields) is type(None)):
            #assert not(type(values) is type(None)),"if you provide field names, you must provide field values"
            if not self._is_iterable(struct_fields):
                struct_fields = [struct_fields]
                field_values = [field_values]
            kwargs.update({f:v for f,v in zip(struct_fields,field_values)})
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
        return self
    def SetAttr(self,field,value):
        if not self._is_iterable(field):
            field = [field]
            value = [value]
        self.__dict__.update({f:v for f,v in zip(field,value)})
        return self
    def GetFields(self,sep='; '):
        return sep.join([ k for k in self.__dict__.keys() if (k[0:2] != '__') and (k[-2:] != '__') ])
        #return self.__dict__.keys()
    def IsField(self,field):
        return field in self.__dict__.keys()
    def RemoveField(self,field):
        return self.__dict__.pop(field,None)
    def RemoveFields(self,*fields):
        r = []
        for k in fields:
            r.append(self.__dict__.pop(k,None))
        return r
    def KeepFields(self,*fields):
        keys = list(self.__dict__.keys())
        for k in keys:
            if not (k in fields):
                self.__dict__.pop(k,None)
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.items()
    def values(self):
        return self.__dict__.values()
    def pop(self,key,default_value=None):
        if type(key) is str:
            return self.__dict__.pop(key,default_value)
        elif isinstance(key,collections.abc.Iterable):
            r = []
            for k in key:
                r.append(self.__dict__.pop(k,default_value))
            return r
        else:
            raise ValueError('key must be a string or a list of strings')
    def __setitem__(self,label,value):
        self.__dict__[label] = value
    def __getitem__(self,label):
        return self.__dict__[label]
    def __repr__(self):
        char_lim      = 50
        char_arg_name = 20
        get_repr      = lambda r: r[:char_lim]+'...' if len(r) > char_lim else r
        type_name     = type(self).__name__
        arg_strings   = []
        star_args     = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_name     = (name[:(char_arg_name-3)] + '...') if len(name) > char_arg_name else name.rjust(char_arg_name)
                arg_strings.append('%s: %s' % (arg_name, get_repr(repr(value)).replace('\n','').strip()  ))
            else:
                star_args[name] = get_repr(repr(value))
        if star_args:
            arg_strings.append('**%s' % star_args)
        sep = '\n' #if len(arg_strings) > 3 else '; '
        return '%s(\n%s\n)' % (type_name, sep.join(arg_strings))
    def _get_kwargs(self):
        return sorted(self.__dict__.items())
    def _get_args(self):
        return []
    def _is_iterable(self,obj):
        return (type(obj) is list) or (type(obj) is tuple)
    def __delitem__(self,*args):
        self.__dict__.__delitem__(*args)
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return iter(self.__dict__)

def get_write_spike_data_functions(saveSites,writeOnRun):
    if saveSites:
        if writeOnRun:
            write_spk_time = write_spk_data #lambda t_ind,k_ind: spkTimeFile.write(str(t_ind) + ',' + str(k_ind) + '\n')
            save_spk_time  = save_spk_data_fake
        else:
            write_spk_time = write_spk_data_fake
            save_spk_time  = save_spk_data
    else:
        save_spk_time  = save_spk_data_fake
        write_spk_time = write_spk_data_fake
    return write_spk_time,save_spk_time

def save_spk_data_fake(X_values, X_ind, X_time, X, k, t):
    pass

def save_spk_data(X_values, X_ind, X_time, X, k, t):
    if X:
        X_values.append(X)
        X_ind.append(k)
        X_time.append(t)

def write_spk_data_fake(spkFile,t,k,X):
    pass

def write_spk_data(spkFile,t,k,X):
    if X:
        spkFile.write(str(t) + ',' + str(k) + ',' + str(X) + '\n')
    #print(t,',',k)

def open_file(spkFileName,saveSites_and_writeOnRun):
    if saveSites_and_writeOnRun:
        spk_file = open(spkFileName,'w')
        spk_file.write('#t,k,Xk\n') # header
        print(f'  ... temp spk file opened: {spkFileName}')
        return spk_file
    return None

def close_file(spkFile,saveSites_and_writeOnRun):
    if saveSites_and_writeOnRun:
        spkFile.close()
        print('  ... temp spk file closed')

