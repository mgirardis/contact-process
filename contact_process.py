# -*- coding: utf-8 -*-
import copy
import collections
import itertools
import argparse
import random
import numpy
import time
import scipy.io
import datetime
import sys
import os

def main():

    # for debug
    #sys.argv = 'python contact_process.py -l 3.4 -N 10000 -tTotal 1000 -graph ring -X0 1 -outputFile cp_ring.mat'.split(' ')[1:]
    parser = argparse.ArgumentParser(description='Contact process in 1+1 dimensions or mean-field all-to-all graph\n\n(l_c=3.297848 for ring; l_c=1 for mean-field)')
    parser.add_argument('-l',           nargs=1, required=False, metavar='l_PARAM',      type=float, default=[1.1],   help='CP rate (l_c=3.297848 for ring; l_c=1 for mean-field)')
    parser.add_argument('-N',           nargs=1, required=False, metavar='N_PARAM',      type=int,   default=[10000], help='number of sites in the network')
    parser.add_argument('-M',           nargs=1, required=False, metavar='M_PARAM',      type=int,   default=[100],   help='number of memory time steps for quasistationary simulation (whenever the system goes into absorbing, it is placed in a random state chosen from the last M visited states)')
    parser.add_argument('-X0',          nargs=1, required=False, metavar='X0_PARAM',     type=int,   default=[1],     help='IC to each site (a scalar 0 or 1)')
    parser.add_argument('-fX0',         nargs=1, required=False, metavar='fX0_PARAM',    type=float, default=[0.5],   help='fraction of sites assigned to X0 as IC (remaining are zero)')
    parser.add_argument('-tTotal',      nargs=1, required=False, metavar='tTotal_PARAM', type=int,   default=[10000], help='total number of time steps')
    parser.add_argument('-tTrans',      nargs=1, required=False, metavar='tTrans_PARAM', type=int,   default=[0],     help='number of transient time steps')
    parser.add_argument('-graph',       nargs=1, required=False, metavar='GRAPH_TYPE',   type=str,   default=['ring'],        choices=['alltoall', 'ring', 'ringfree'], help='alltoall -> mean-field simulation; ring -> 1+1 simulation with periodic boundary conditions; ringfree -> ring with free boundaries')
    parser.add_argument('-update',      nargs=1, required=False, metavar='UPDATE_TYPE',  type=str,   default=['seq'],         choices=['seq','sequential','par','parallel'], help='seq -> standard update scheme: 1 particle update/ts (paragraph after eq 3.35 in Henkel book); par -> parallel update (attempts to update all articles at each ts, matches the E/I network')
    parser.add_argument('-sim',         nargs=1, required=False, metavar='SIM_TYPE',     type=str,   default=['timeevo'],     choices=['timeevo', 'aval'], help='timeevo -> simple time evolution stimulation (quasistatic if M > 0); aval -> avalanche simulation; seeds 1 site every time activity dies out')
    #parser.add_argument('-activation',  nargs=1, required=False, metavar='ACTIV_TYPE',   type=str,   default=['rate'],        choices=['rate', 'prob'], help='rate -> each site is activated if random < l*r (r=frac of act neigh); prob -> each site is activated if random < p*r (p=l/(1+l) and r=frac of act neigh -- seems to yield a wrong l_c, but seems to be the correct one according to books)')
    #parser.add_argument('-algorithm',   nargs=1, required=False, metavar='ALGO_TYPE',    type=str,   default=['tomeoliveira'],choices=['mine', 'tomeoliveira'], help='mine -> my alogirhtm -- seems to be wrong; tomeoliveira -> algorithm describe in pg 308pdf/402 of the Tome & Oliveira book, before eq (13.6)')
    parser.add_argument('-noX0Rand',    required=False, action='store_true', default=False, help='if set, Xi is generated sequentially')
    parser.add_argument('-expandtime',  required=False, action='store_true', default=False, help='if set, then uses the dt=1/N to expand the total simulation time: tTotal_eff = tTotal / dt (only for sequential update)')
    parser.add_argument('-outputFile',  nargs=1, required=False, metavar='OUTPUT_FILE_NAME', type=str, default=['cp.mat'], help='name of the output file')
    args =  namespace_to_structtype(parser.parse_args())
    args.X0Rand = not args.noX0Rand
    args.dt = Get_Simulation_Timescale(args)

    print("* Running simulation... Total time steps = %d" % (int(numpy.round(args.tTotal / args.dt))))
    simulation_func = Get_Simulation_Func(args)
    print('update type == %s'%args.update)
    start_sim_time = time.monotonic()
    rho = simulation_func(**keep_keys(args.GetDict(),get_func_param(simulation_func)))
    end_sim_time = time.monotonic()
    print("* End of simulation... Total time: {}".format(datetime.timedelta(seconds=end_sim_time - start_sim_time)))

    args.expandtime = not is_parallel_update(args.update)
    args.dt = Get_Simulation_Timescale(args)
    args.outputFile = get_new_file_name(get_output_filename(args.outputFile))
    print('* Writing ... %s'%args.outputFile)
    scipy.io.savemat(args.outputFile,dict(cmd_line=' '.join(sys.argv),time=numpy.arange(len(rho))*args.dt,rho=rho,**args),long_field_names=True,do_compression=True)

def get_output_filename(path):
    fname,fext = os.path.splitext(path)
    if fext.lower() != '.mat':
        if fext == '.txt':
            return fname + '.mat'
        else:
            return fname + fext + '.mat'

def is_parallel_update(update):
    return (update == 'par') or (update == 'parallel')

def Get_Simulation_Func(args):
    if args.sim == 'aval':
        if not is_parallel_update(args.update):
            args.update = 'par' # forcing parallel update for avalanche
            print('forcing parallel update because sim == %s'%args.sim)
    if args.graph == 'alltoall':
        if is_parallel_update(args.update):
            return Run_MF_parallel
        else:
            return Run_MF_sequential
    else: # (args.graph == 'ring') or (args.graph == 'ringfree')
        if is_parallel_update(args.update):
            return Run_RingGraph_parallel
        else:
            return Run_RingGraph_sequential

def Get_Simulation_Timescale(args):
    if (args.update == 'par') or (args.update == 'parallel'):
        return 1.0
    else:
        if args.expandtime:
            return 1.0 / float(args.N) # this time scale is suggested in Dickman and Marro book; Henkel normalizes the time scale by total rate too -- pg 87 pdf, paragraph after eq. 3.35 book Henkel
        else:
            return 1.0

def get_random_state(X,f_act):
    # X -> site vector (in/out parameter); numpy.ndarray
    # f_act -> fraction of active elements
    X[random.sample(range(len(X)),k=int(f_act*len(X)))] = 1.0

def Run_MF_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,sim):
    # all sites update in the same time step -- matches the GL model
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        #X[random.sample(range(N),k=int(fX0*N))] = 1.0
        get_random_state(X,fX0)
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N)],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    #neigh = [ ((k-1)%N,(k+1)%N) for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions
    #n_neigh = float(len(neigh[0])) # number of neighbors of each site
    #p = l / (1.0+l) if activation == 'prob' else l # probability is the transition rate divided by total rate (1+lambda)
    is_aval_sim = sim == 'aval'
    alpha = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl = float(N)
    rho_temp = 0.0
    rho_prev = sum(X) / N_fl
    rho_memory = CyclicStack(M)
    rho_memory[0] = fX0
    for t in range(tTrans):
        rho_temp = 0.0
        for i in range(N):
            X[i] = state_iter(X[i],rho_prev,alpha)
            rho_temp += X[i]
        if rho_temp < 1.0:
            if is_aval_sim:
                rho_temp = 1.0
            else:
                if M == 0:
                    break
                get_random_state(X, random.choice(rho_memory))
                rho_temp = sum(X)
        rho_temp = rho_temp / N_fl
        rho_prev = rho_temp
        rho_memory[t+1] = rho_temp

    rho = [ 0.0 for k in range(tTotal) ]
    rho_memory = CyclicStack(M)
    rho[0] = rho_prev
    rho_memory[0] = rho_prev
    for t in range(1,tTotal):
        rho_temp = 0.0
        for i in range(N):
            X[i] = state_iter(X[i],rho_prev,alpha)
            rho_temp += X[i]
        if is_aval_sim:
            rho[t] = rho_temp / N_fl
            if rho_temp < 1.0:
                rho_prev = 1.0 / N_fl
            else:
                rho_prev = rho[t]
        else:
            if rho_temp < 1.0:
                if M == 0:
                    break
                get_random_state(X, random.choice(rho_memory))
                rho_temp = sum(X)
            rho[t] = rho_temp / N_fl
            rho_prev = rho[t]
            rho_memory[t] = rho_prev
    return rho

def Run_MF_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M):
    # only 1 site is attempted update at each time step
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        #X[random.sample(range(N),k=int(fX0*N))] = 1.0
        get_random_state(X,fX0)
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N)],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    #neigh = [ ((k-1)%N,(k+1)%N) for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions
    #n_neigh = float(len(neigh[0])) # number of neighbors of each site
    #p = l / (1.0+l) if activation == 'prob' else l # probability is the transition rate divided by total rate (1+lambda)
    alpha = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl = float(N)
    tTrans_eff = int(numpy.round(tTrans / dt))
    tTotal_eff = int(numpy.round(tTotal / dt))
    n_neigh = N_fl - 1.0
    rho_memory = CyclicStack(M)
    rho_memory[0] = fX0
    sum_of_X = sum(X)
    for t in range(tTrans_eff):
        i = random.randint(0,N-1) # selecting update site
        Xa = X[i]
        X[i] = state_iter(X[i],(sum_of_X-X[i])/n_neigh,alpha) # updating site i
        if (X[i] == 1.0) and (Xa == 0.0):
            sum_of_X += 1 # a particle was added to the lattice
        if (X[i] == 0.0) and (Xa == 1.0):
            sum_of_X -= 1 # a particle was removed from the lattice
        #sum_of_X = sum(X)
        if sum_of_X < 1.0:
            if M == 0:
                break
            get_random_state(X, random.choice(rho_memory))
            sum_of_X = sum(X)
        rho_memory[t+1] = sum_of_X / N_fl

    rho = [ 0.0 for k in range(tTotal_eff) ]
    sum_of_X = sum(X)
    rho[0] = sum_of_X / N_fl
    rho_memory = CyclicStack(M)
    rho_memory[0] = rho[0]
    for t in range(1,tTotal_eff):
        i = random.randint(0,N-1) # selecting update site
        Xa = X[i]
        X[i] = state_iter(X[i],(sum_of_X-X[i])/n_neigh,alpha) # updating site i
        if (X[i] == 1.0) and (Xa == 0.0):
            sum_of_X += 1
        if (X[i] == 0.0) and (Xa == 1.0):
            sum_of_X -= 1
        #sum_of_X = sum(X)
        if sum_of_X < 1.0:
            if M == 0:
                break
            get_random_state(X, random.choice(rho_memory))
            sum_of_X = sum(X)
        rho[t] = sum_of_X / N_fl
        rho_memory[t] = rho[t]
    return rho

def Run_RingGraph_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph,sim):
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        #X[random.sample(range(N),k=int(fX0*N))] = 1.0
        get_random_state(X,fX0)
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N) ],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    neigh = get_ring_neighbors(graph,N)
    #n_neigh = float(len(neigh[0])) # number of neighbors of each site
    #p = l / (1.0+l) if activation == 'prob' else l  # probability is the transition rate divided by total rate (1+lambda), the division by n_neigh is carried out in state_iter
    is_aval_sim = sim == 'aval'
    alpha = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl = float(N)
    rho_memory = CyclicStack(M)
    rho_memory[0] = fX0
    for t in range(tTrans):
        sum_of_X = 0.0
        for i in range(N):
            X[i] = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha)
            sum_of_X += X[i]
        if sum_of_X < 1.0:
            if is_aval_sim:
                sum_of_X = 2.0
                k = int((N-1)/2)
                X[k] = 1.0
                #X[k-1] = 1.0
                #X[k+1] = 1.0
            else:
                if M == 0:
                    break
                get_random_state(X, random.choice(rho_memory))
                sum_of_X = sum(X)
        rho_memory[t+1] = sum_of_X / N_fl
    rho = [ 0.0 for k in range(tTotal) ]
    rho[0] = sum(X) / N_fl
    rho_memory = CyclicStack(M)
    rho_memory[0] = rho[0]
    rho_prev = rho[0]
    for t in range(1,tTotal):
        rho_temp = 0.0
        for i in range(N):
            X[i] = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha)
            rho_temp += X[i]
        if is_aval_sim:
            rho[t] = rho_temp / N_fl
            if rho_temp < 1.0:
                k = int((N-1)/2)
                X[k] = 1.0
                #X[k-1] = 1.0 # two seeds force X[k] == 1.0 in the next time step
                #X[k+1] = 1.0
        else:
            if rho_temp < 1.0:
                if M == 0:
                    break
                get_random_state(X, random.choice(rho_memory))
                rho_temp = sum(X)
            rho[t] = rho_temp / N_fl
            rho_memory[t] = rho[t]
    return rho

def Run_RingGraph_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph):
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        #X[random.sample(range(N),k=int(fX0*N))] = 1.0
        get_random_state(X,fX0)
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N) ],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    neigh = get_ring_neighbors(graph,N)
    #n_neigh = float(len(neigh[0])) # number of neighbors of each site
    #p = l / (1.0+l) if activation == 'prob' else l  # probability is the transition rate divided by total rate (1+lambda), the division by n_neigh is carried out in state_iter
    alpha = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl = float(N)
    tTrans_eff = int(numpy.round(tTrans / dt))
    tTotal_eff = int(numpy.round(tTotal / dt))
    sum_of_X = sum(X)
    rho_memory = CyclicStack(M)
    rho_memory[0] = fX0
    for t in range(tTrans_eff):
        i = random.randint(0,N-1) # selecting update site
        Xa = X[i]
        X[i] = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha) # updating site i
        if (X[i] == 1.0) and (Xa == 0.0):
            sum_of_X += 1
        if (X[i] == 0.0) and (Xa == 1.0):
            sum_of_X -= 1
        #sum_of_X = sum(X)
        if sum_of_X < 1.0:
            if M == 0:
                break
            get_random_state(X, random.choice(rho_memory))
            sum_of_X = sum(X)
        rho_memory[t+1] = sum_of_X / N_fl
    rho = [ 0.0 for k in range(tTotal_eff) ]
    sum_of_X = sum(X)
    rho[0] = sum(X) / N_fl
    rho_memory = CyclicStack(M)
    rho_memory[0] = rho[0]
    for t in range(1,tTotal_eff):
        i = random.randint(0,N-1) # selecting update site
        Xa = X[i]
        X[i] = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha) # updating site i
        if (X[i] == 1.0) and (Xa == 0.0):
            sum_of_X += 1
        if (X[i] == 0.0) and (Xa == 1.0):
            sum_of_X -= 1
        #sum_of_X = sum(X)
        if sum_of_X < 1.0:
            if M == 0:
                break
            get_random_state(X, random.choice(rho_memory))
            sum_of_X = sum(X)
        rho[t] = sum_of_X / N_fl
        rho_memory[t] = sum_of_X / N_fl
    return rho

def get_ring_neighbors(graph,N):
    if graph == 'ring':
        return get_ring_neighbors_periodic(N)
    elif graph == 'ringfree':
        return get_ring_neighbors_free(N)
    else:
        raise ValueError('get_neighbors not defined for graph %s'%graph)

def get_ring_neighbors_periodic(N):
    return [ [(k-1)%N,(k+1)%N] for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions

def get_ring_neighbors_free(N):
    n = get_ring_neighbors_periodic(N)
    n[0] = [1] # first site connects only to the right
    n[N-1] = [N-2] # last site connects only to the left
    return n

def state_iter(X,n,inv_l):
    # described in pg 308pdf/402 Tome Oliveira book before eq 13.6
    # At each time step we choose a site at random, say site i.
    #   (a) If i is occupied, than we generate a random number xi uniformly distributed in the interval [0;1].
    #       If xi <= 1/lambda, the particle is annihilated and the site becomes empty.
    #       Otherwise, the site remains occupied.
    #   (b) If i is empty, then one of its neighbors is chosen at random.
    #       If the neighboring site is occupied then we create a particle at site i.
    #       Otherwise, the site i remains empty. 
    # X -> state of node
    # n -> fraction of active neighbors of X
    # inv_l -> inverse of activation rate: inv_l = 1/lambda = alpha in the book
    # returns the new state based on the previous state X for a given node
    #         site is occupied, so it eliminates               site is empty, so it creates
    #         the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
    return float( numpy.random.random() > inv_l ) if X else float(numpy.random.random()<n)

def stack_add(stack,k):
    # adds k to the top of the stack s [i.e., to position s(1) ]
    # s is a vector with fixed length
    # stack_add will shift all elements of s to the right, and add k as the first element in s
    stack = numpy.roll(stack,1)
    stack[0] = k

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
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,label,value):
        self.__dict__[label] = value
    def GetDict(self):
        return self.__dict__
    def __setitem__(self,label,value):
        self.__dict__[label] = value
    def __getitem__(self,label):
        return self.__dict__[label]
    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))
    def _get_kwargs(self):
        return sorted(self.__dict__.items())
    def _get_args(self):
        return []
    def __delitem__(self,*args):
        self.__dict__.__delitem__(*args)
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return iter(self.__dict__)

class CyclicStack(collections.abc.MutableSequence):
    def __init__(self,maxsize=None,values=0.0):
        # creates a stack with fixed size given by maxsize or len(values)
        # the initial values have no use at all
        # this stack is cyclic, so every time you add an element with index > maxsize, it gets added cyclic to the beginning of the stack (with % function)
        # this stack will get items cyclicly relative to the added items (ignoring the initial values, initial values are only for initialization)
        self._stack = []
        if isinstance(values,collections.abc.Iterable):
            for x in values:
                self._stack.append(x)
            self._maxsize = len(values)
        else:
            if maxsize is None:
                raise ValueError('maxsize must be set if val is not iterable')
            for i in range(maxsize):
                self._stack.append(values)
            self._maxsize = maxsize
        self._k = 0 # internal counter
    def __setitem__(self,ind,val):
        self._stack[ind%self._maxsize] = val
        self._k = self._k+1 if self._k < self._maxsize else self._maxsize
    def __getitem__(self,index):
        return self._stack[index%self._k]
    def insert(self,index,val):
        raise ValueError('not possible to insert into the stack... use stack[index] instead')
    def __delitem__(self,index):
        raise ValueError('not possible to delete from the stack...')
    def __len__(self):
        return self._k
    def __iter__(self):
        return itertools.islice(self._stack,self._k).__iter__()
    def __str__(self):
        if self._k < self._maxsize:
            return str(self._stack[:self._k])
        else:
            return str(self._stack)
    def __repr__(self):
        if self._k < self._maxsize:
            return self._stack[:self._k].__repr__()
        else:
            return self._stack.__repr__()

if __name__ == '__main__':
    main()

########################## 
###
###
### Functions below are probably wrong, but are kept for reference 
###
###

def Run_MF_parallel_maybe_wrong(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,activation,expandtime,algorithm):
    # all sites update in the same time step -- matches the GL model
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        #X[numpy.random.permutation(N)[:int(fX0*N)]] = 1.0
        X[random.sample(range(N),k=int(fX0*N))] = 1.0
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N)],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    #neigh = [ ((k-1)%N,(k+1)%N) for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions
    #n_neigh = float(len(neigh[0])) # number of neighbors of each site
    p = l / (1.0+l) if activation == 'prob' else l # probability is the transition rate divided by total rate (1+lambda)
    N_fl = float(N)
    rho_temp = 0.0
    rho_prev = sum(X) / N_fl
    for t in range(tTrans):
        rho_temp = 0.0
        for i in range(N):
            X[i] = state_iter_maybe_wrong(X[i],rho_prev,p)
            rho_temp += X[i]
        rho_temp = rho_temp / N_fl
        rho_prev = rho_temp

    rho = [ 0.0 for k in range(tTotal) ]
    rho[0] = rho_prev
    for t in range(1,tTotal):
        for i in range(N):
            X[i] = state_iter_maybe_wrong(X[i],rho_prev,p)
            rho[t] += X[i]
        rho[t] = rho[t] / N_fl
        rho_prev = rho[t]
    return rho

def Run_MF_sequential_maybe_wrong(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,activation,expandtime,algorithm):
    # only 1 site is attempted update at each time step
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        #X[numpy.random.permutation(N)[:int(fX0*N)]] = 1.0
        X[random.sample(range(N),k=int(fX0*N))] = 1.0
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N)],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    #neigh = [ ((k-1)%N,(k+1)%N) for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions
    #n_neigh = float(len(neigh[0])) # number of neighbors of each site
    p = l / (1.0+l) if activation == 'prob' else l # probability is the transition rate divided by total rate (1+lambda)
    N_fl = float(N)
    tTrans_eff = int(numpy.round(tTrans / dt))
    tTotal_eff = int(numpy.round(tTotal / dt))
    n_neigh = N_fl - 1.0
    for t in range(tTrans_eff):
        i = random.randint(0,N-1) # selecting update site
        X[i] = state_iter_maybe_wrong(X[i],(sum(X)-X[i])/n_neigh,p) # updating site i

    rho = [ 0.0 for k in range(tTotal_eff) ]
    sum_of_X = sum(X)
    rho[0] = sum_of_X / N_fl
    for t in range(1,tTotal_eff):
        i = random.randint(0,N-1) # selecting update site
        X[i] = state_iter_maybe_wrong(X[i],(sum_of_X-X[i])/n_neigh,p) # updating site i
        sum_of_X = sum(X)
        rho[t] = sum_of_X / N_fl
    return rho

def Run_RingGraph_parallel_maybe_wrong(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,activation,expandtime,algorithm):
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        X[random.sample(range(N),k=int(fX0*N))] = 1.0
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N) ],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    neigh = [ [(k-1)%N,(k+1)%N] for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions
    n_neigh = float(len(neigh[0])) # number of neighbors of each site
    p = l / (1.0+l) if activation == 'prob' else l  # probability is the transition rate divided by total rate (1+lambda), the division by n_neigh is carried out in state_iter
    N_fl = float(N)
    for t in range(tTrans):
        for i in range(N):
            X[i] = state_iter_maybe_wrong(X[i],sum(X[neigh[i]])/n_neigh,p)
    rho = [ 0.0 for k in range(tTotal) ]
    rho[0] = sum(X) / N_fl
    for t in range(1,tTotal):
        for i in range(N):
            X[i] = state_iter_maybe_wrong(X[i],sum(X[neigh[i]])/n_neigh,p)
            rho[t] += X[i]
        rho[t] = rho[t] / N_fl
    return rho

def Run_RingGraph_sequential_maybe_wrong(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,activation,expandtime,algorithm):
    X0 = float(X0)
    if X0Rand:
        X = numpy.zeros(N)
        X[random.sample(range(N),k=int(fX0*N))] = 1.0
    else:
        X = numpy.asarray([ (X0 if i < int(fX0*N) else 0.0) for i in range(N) ],dtype=float)
    #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    neigh = [ [(k-1)%N,(k+1)%N] for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions
    n_neigh = float(len(neigh[0])) # number of neighbors of each site
    p = l / (1.0+l) if activation == 'prob' else l  # probability is the transition rate divided by total rate (1+lambda), the division by n_neigh is carried out in state_iter
    N_fl = float(N)
    tTrans_eff = int(numpy.round(tTrans / dt))
    tTotal_eff = int(numpy.round(tTotal / dt))
    for t in range(tTrans_eff):
        i = random.randint(0,N-1) # selecting update site
        X[i] = state_iter_maybe_wrong(X[i],sum(X[neigh[i]])/n_neigh,p) # updating site i
    rho = [ 0.0 for k in range(tTotal_eff) ]
    rho[0] = sum(X) / N_fl
    for t in range(1,tTotal_eff):
        i = random.randint(0,N-1) # selecting update site
        X[i] = state_iter_maybe_wrong(X[i],sum(X[neigh[i]])/n_neigh,p) # updating site i
        rho[t] = sum(X) / N_fl
    return rho

def state_iter_maybe_wrong(X,n,p):
    # X -> state of node
    # n -> fraction of active neighbors of X
    # p -> activation probability
    return 0.0 if X else float(numpy.random.random()<(p*n))