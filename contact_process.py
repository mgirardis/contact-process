# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy
import random
import argparse
import datetime
import itertools
import collections
from modules.io import *

def main(): 

    # for debug
    #sys.argv = 'python contact_process.py -l 3.4 -N 10000 -tTotal 1000 -graph ring -X0 1 -outputFile cp_ring.mat'.split(' ')[1:]
    #sys.argv  = 'python contact_process.py -l 3.0 -N 1000 -tTotal 10000 -graph ring -X0 1 -outputFile cp_ring_l3.0_aval_N1000_t10000.mat -saveSites -writeOnRun'.split(' ')[1:]
    parser = argparse.ArgumentParser(description='Contact process in 1+1 dimensions or mean-field all-to-all graph\n\n(l_c=3.297848 for ring; l_c=1 for mean-field)',formatter_class=argparse.RawTextHelpFormatter)
    parser = add_simulation_parameters(parser)

    args             = namespace_to_structtype(parser.parse_args())
    args.docstring   = get_help_string(parser)
    args.X0Rand      = not args.noX0Rand
    args.expandtime  = not is_parallel_update(args.update)
    args.dt          = Get_Simulation_Timescale(args)
    args.outputFile  = get_new_file_name(get_output_filename(args.outputFile))
    args.spkFileName = args.outputFile.replace('.mat','_spk.txt') if args.writeOnRun else ''

    output_dir       = os.path.dirname(args.outputFile)
    os.makedirs(output_dir, exist_ok=True)

    print('* Simulation parameters:')
    print(args)

    print("* Running simulation... Total time steps = %d" % (int(numpy.round(args.tTotal / args.dt))))
    simulation_func              = Get_Simulation_Func(args)
    sim_time_start               = time.monotonic()
    rho, X_values, X_ind, X_time = simulation_func(**keep_keys(dict(**args),get_func_param(simulation_func)))
    sim_time_end                 = time.monotonic()
    print("* End of simulation... Total time: {}".format(datetime.timedelta(seconds=sim_time_end - sim_time_start)))

    print('* Writing ... %s'%args.outputFile)
    save_simulation_file(sys.argv, args, rho, X_values, X_ind, X_time)
    del rho, X_values, X_ind, X_time # releasing memory to avoid memory error when merging files
    if args.saveSites and args.writeOnRun and args.mergespkfile:
        merge_simulation_files(args.outputFile, args.spkFileName, remove_spk_file=False, verbose=True)

    print('done')
    print(' ')

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

def get_initial_network_state_for_output(X, saveSites):
    if saveSites:
        X_values      = [ x for x   in X            if x ]
        X_ind         = [ i for i,x in enumerate(X) if x ]
        X_time        = [ 0 for x   in X            if x ]
    else:
        X_values, X_ind, X_time = [], [], []
    return X_values, X_ind, X_time

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
        print('spk file opened: %s'%spkFileName)
        return spk_file
    return None

def close_file(spkFile,saveSites_and_writeOnRun):
    if saveSites_and_writeOnRun:
        spkFile.close()
        print('spk file close')

def Run_MF_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,sim,saveSites,writeOnRun,spkFileName):
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
    is_aval_sim   = sim == 'aval'
    alpha         = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl          = float(N)
    rho_temp      = 0.0
    rho_prev      = sum(X) / N_fl
    rho_memory    = CyclicStack(M)
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
        rho_temp        = rho_temp / N_fl
        rho_prev        = rho_temp
        rho_memory[t+1] = rho_temp # keeping a memory of the fraction of active sites in the previous M steps (since it is mean-field, it doesn't matter which sites are active)

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_values, X_ind, X_time      = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho                     = [ 0.0 for k in range(tTotal) ]
    rho_memory              = CyclicStack(M)
    rho[0]                  = rho_prev
    rho_memory[0]           = rho_prev
    for t in range(1,tTotal):
        rho_temp = 0.0
        for i in range(N):
            X[i] = state_iter(X[i],rho_prev,alpha)
            rho_temp += X[i]
            save_spk_time(X_values, X_ind, X_time, X[i], i, t) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            write_spk_time(spk_file, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
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
    close_file(spk_file,saveSites and writeOnRun)
    return rho, X_values, X_ind, X_time

def Run_MF_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,saveSites,writeOnRun,spkFileName):
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
        i    = random.randint(0,N-1) # selecting update site
        Xa   = X[i]
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

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_values, X_ind, X_time      = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho           = [ 0.0 for k in range(tTotal_eff) ]
    sum_of_X      = sum(X)
    rho[0]        = sum_of_X / N_fl
    rho_memory    = CyclicStack(M)
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
        save_spk_time(X_values, X_ind, X_time, X[i], i, t*dt) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        write_spk_time(spk_file, t*dt, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        if sum_of_X < 1.0:
            if M == 0:
                break
            get_random_state(X, random.choice(rho_memory))
            sum_of_X = sum(X)
        rho[t] = sum_of_X / N_fl
        rho_memory[t] = rho[t]
    close_file(spk_file,saveSites and writeOnRun)
    return rho, X_values, X_ind, X_time

def Run_RingGraph_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph,sim,saveSites,writeOnRun,spkFileName):
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
    is_aval_sim   = sim == 'aval'
    alpha         = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl          = float(N)
    rho_memory    = CyclicStack(M)
    rho_memory[0] = fX0
    for t in range(tTrans):
        sum_of_X = 0.0
        for i in range(N):
            X[i] = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha)
            sum_of_X += X[i]
        if sum_of_X < 1.0:
            if is_aval_sim:
                sum_of_X = 2.0
                k        = int((N-1)/2)
                X[k]     = 1.0
                #X[k-1] = 1.0
                #X[k+1] = 1.0
            else:
                if M == 0:
                    break
                get_random_state(X, random.choice(rho_memory))
                sum_of_X = sum(X)
        rho_memory[t+1] = sum_of_X / N_fl
    
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_values, X_ind, X_time      = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)
    
    rho           = [ 0.0 for k in range(tTotal) ]
    rho[0]        = sum(X) / N_fl
    rho_memory    = CyclicStack(M)
    rho_memory[0] = rho[0]
    rho_prev      = rho[0]
    for t in range(1,tTotal):
        rho_temp = 0.0
        for i in range(N):
            X[i] = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha)
            rho_temp += X[i]
            save_spk_time(X_values, X_ind, X_time, X[i], i, t) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            write_spk_time(spk_file, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        if is_aval_sim:
            rho[t] = rho_temp / N_fl
            if rho_temp < 1.0:
                k        = int((N-1)/2) # middle of the network
                X[k]     = 1.0
                rho_temp = 1.0
                #X[k-1] = 1.0 # two seeds force X[k] == 1.0 in the next time step
                #X[k+1] = 1.0
        else:
            if rho_temp < 1.0:
                if M == 0:
                    break
                get_random_state(X, random.choice(rho_memory))
                rho_temp = sum(X)
            rho[t]        = rho_temp / N_fl
            rho_memory[t] = rho[t]
    close_file(spk_file,saveSites and writeOnRun)
    return rho, X_values, X_ind, X_time

def Run_RingGraph_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph,saveSites,writeOnRun,spkFileName):
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
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_values, X_ind, X_time      = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho           = [ 0.0 for k in range(tTotal_eff) ]
    sum_of_X      = sum(X)
    rho[0]        = sum(X) / N_fl
    rho_memory    = CyclicStack(M)
    rho_memory[0] = rho[0]
    for t in range(1,tTotal_eff):
        i = random.randint(0,N-1) # selecting update site
        Xa = X[i]
        X[i] = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha) # updating site i
        if (X[i] == 1.0) and (Xa == 0.0):
            sum_of_X += 1
        if (X[i] == 0.0) and (Xa == 1.0):
            sum_of_X -= 1
        save_spk_time(X_values, X_ind, X_time, X[i], i, t*dt) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        write_spk_time(spk_file, t*dt, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        #sum_of_X = sum(X)
        if sum_of_X < 1.0:
            if M == 0:
                break
            get_random_state(X, random.choice(rho_memory))
            sum_of_X = sum(X)
        rho[t] = sum_of_X / N_fl
        rho_memory[t] = sum_of_X / N_fl
    close_file(spk_file,saveSites and writeOnRun)
    return rho, X_values, X_ind, X_time

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
