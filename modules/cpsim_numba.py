import numpy
import random
from enum import IntEnum
from numba import njit,types,typeof
from numba.typed import List

class GraphType(IntEnum):
    ALLTOALL = 0
    RING     = 1
    RINGFREE = 2

def str_to_GraphType(graph_str):
    if (graph_str == 'alltoall') or (graph_str == 'mf'):
        return GraphType.ALLTOALL
    elif (graph_str == 'ring'):
        return GraphType.RING
    elif (graph_str == 'ringfree'):
        return GraphType.RINGFREE
    else:
        raise ValueError(f'Unknown graph type: {graph_str}')

def GraphType_to_str(graph:GraphType):
    return graph.name.lower()
    #if graph == GraphType.ALLTOALL:
    #    return 'alltoall'
    #elif graph == GraphType.RING:
    #    return 'ring'
    #elif graph == GraphType.RINGFREE:
    #    return 'ringfree'
    #else:
    #    raise ValueError(f'Unknown graph type: {graph}')

def is_parallel_update(update):
    return (update == 'par') or (update == 'parallel')

def Get_Simulation_Func(args):
    if args.sim == 'aval':
        if not is_parallel_update(args.update):
            args.update = 'par' # forcing parallel update for avalanche
            print('forcing parallel update because sim == %s'%args.sim)
    if args.graph == GraphType.ALLTOALL:
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


#@njit
#def random_sample(N, K):
#    # Initialize a NumPy array with the range of values from 0 to N-1
#    population = numpy.arange(N, dtype=numpy.int64)
#    
#    # Perform Fisher-Yates shuffle for the first K elements
#    for i in range(K):
#        # Pick a random index from i to N-1
#        j = random.randint(i, N-1)
#        # Swap the elements at indices i and j
#        population[i], population[j] = population[j], population[i]
#    
#    # Return the first K elements as a NumPy array
#    return population[:K]

#@njit
#def get_random_state(X,f_act):
#    # X -> site vector (in/out parameter); numpy.ndarray
#    # f_act -> fraction of active elements
#    ind = random_sample(len(X), int(f_act * len(X)))
#    for i in ind:
#        X[i] = 1.0


@njit
def get_ordered_state(X,f_act):
    N = len(X)
    K = int(f_act * N)
    for i in range(K):
        X[i] = 1.0
    for i in range(K,N):
        X[i] = 0.0

@njit
def get_random_state(X,f_act):
    # X -> site vector (in/out parameter); numpy.ndarray
    # f_act -> fraction of active elements
    N = len(X)
    get_ordered_state(X,f_act)
    for i in range(N-1):
        #j = random.randint(i, N-1)
        j = numpy.random.randint(i, N)
        X[i],X[j] = X[j],X[i] # Shuffle X in-place using Fisher-Yates
    #return X

type_X_data_item = types.Tuple((types.float64, types.int64, types.int64))
type_X_data      = types.ListType(type_X_data_item)

@njit(types.void(type_X_data,types.int64,types.int64,types.int64))
def save_spk_data_fake(X_data, t, k, X):
    return None

@njit(types.void(type_X_data,types.int64,types.int64,types.int64))
def save_spk_data(X_data, t, k, X):
    if X:
        X_data.append((t,k,X))
        #X_values.append(X)
        #X_ind.append(k)
        #X_time.append(t)
    return None

@njit(types.void(type_X_data,types.int64,types.int64,types.int64))
def write_spk_data_fake(spkFile,t,k,X):
    return None

@njit(types.void(type_X_data,types.int64,types.int64,types.int64))
def write_spk_data(spkFile,t,k,X):
    if X:
        #spkFile.write(str(t) + ',' + str(k) + ',' + str(X) + '\n')
        print(t,',',k,',',X)
    return None

@njit(type_X_data(types.int64[:],types.boolean))
def get_initial_network_state_for_output(X, saveSites):
    #if saveSites:
    #    X_values      = numpy.array([ x for x   in X            if x ],dtype=numpy.int64)
    #    X_ind         = numpy.array([ i for i,x in enumerate(X) if x ],dtype=numpy.int64)
    #    X_time        = numpy.array([ 0 for x   in X            if x ],dtype=numpy.int64)
    #else:
    #    X_values, X_ind, X_time = numpy.array([],dtype=numpy.int64), numpy.array([],dtype=numpy.int64), numpy.array([],dtype=numpy.int64)
    #return X_values, X_ind, X_time
    X_data = List.empty_list(type_X_data_item) #numpy.empty((0,3),dtype=numpy.int64) # t,k,X
    if saveSites:
        for k,x in enumerate(X):
            save_spk_data(X_data, x, k, 0)
    return X_data

type_write_spk_data = typeof(write_spk_data)
type_save_spk_data  = typeof(save_spk_data)
@njit(types.Tuple((type_write_spk_data, type_save_spk_data))(types.boolean, types.boolean))
def get_write_spike_data_functions(saveSites,writeOnRun):
    if saveSites:
        if writeOnRun:
            write_spk_time = write_spk_data #lambda t_ind,k_ind: spkTimeFile.write(str(t_ind) + ',' + str(k_ind) + '\n')
            save_spk_time  = save_spk_data_fake
        else:
            write_spk_time = write_spk_data_fake
            save_spk_time  = save_spk_data
    else:
        write_spk_time = write_spk_data_fake
        save_spk_time  = save_spk_data_fake
    return write_spk_time,save_spk_time


@njit
def open_file(spkFileName,saveSites_and_writeOnRun):
    if saveSites_and_writeOnRun:
        #spk_file = open(spkFileName,'w')
        #spk_file.write('#t,k,Xk\n') # header
        #print('spk file opened: %s'%spkFileName)
        #return spk_file
        #print('*** writing file ',spkFileName,' during simulation')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('################################################## file ', spkFileName, ' will be printed to stdout')
        #print('################################################## due to numba limitation')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('[[[ BEGINNING OF FILE ]]]')
        print('#t,k,X')
    return None

@njit
def close_file(spkFile,spkFileName,saveSites_and_writeOnRun):
    if saveSites_and_writeOnRun:
        print('')
        #spkFile.close()
        #print('spk file close')
        #print('[[[ END OF FILE ]]]')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')
        #print('##################################################')


@njit
def get_ring_neighbors(graph:GraphType,N):
    if graph == GraphType.RING:
        return get_ring_neighbors_periodic(N)
    elif graph == GraphType.RINGFREE:
        return get_ring_neighbors_free(N)
    else:
        raise ValueError(f'get_neighbors not defined for graph {graph}')

@njit
def get_ring_neighbors_periodic(N):
    return [ [(k-1)%N,(k+1)%N] for k in range(N) ] # the (k+-1) mod N implements the periodic boundary conditions

@njit
def get_ring_neighbors_periodic(N):
    n = numpy.empty((N, 2), dtype=numpy.int64)
    for k in range(N):
        n[k, 0] = (k - 1) % N  # left neighbor
        n[k, 1] = (k + 1) % N  # right neighbor
    return n

@njit
def get_ring_neighbors_free(N):
    n = get_ring_neighbors_periodic(N)
    n[0,0]   = 1   # first site connects only to the right
    n[0,1]   = 1   # first site connects only to the right
    n[N-1,0] = N-2 # last site connects only to the left
    n[N-1,1] = N-2 # last site connects only to the left
    return n

@njit(types.boolean(types.int64))
def bool2int(x):
    return 1 if x else 0

@njit
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
    #return bool2int( numpy.random.random() > inv_l ) if X else bool2int(numpy.random.random() < n)
    if X: # site is occupied
        return bool2int(numpy.random.random() > inv_l)
    else: # site is empty
        return bool2int(numpy.random.random() < n)

@njit
def stack_add(stack,k):
    # adds k to the top of the stack s [i.e., to position s(1) ]
    # s is a vector with fixed length
    # stack_add will shift all elements of s to the right, and add k as the first element in s
    stack = numpy.roll(stack,1)
    stack[0] = k

type_cyclic_stack_data = types.Tuple((types.float64[:],types.int64))
@njit(type_cyclic_stack_data(types.int64))
def CyclicStack_Init(maxsize):
    #stack, maxsize, count = cyclic_stack_data
    stack = numpy.full(maxsize, 0.0)
    count = 0
    return stack,count

@njit(type_cyclic_stack_data(types.float64[:],types.int64,types.int64,types.int64,types.float64))
def CyclicStack_Set(stack, maxsize, count, index, value):
    #stack, maxsize, count = cyclic_stack_data
    stack[index % maxsize] = value
    if count < maxsize:
        count += 1
    return stack,count

@njit(types.float64(types.float64[:],types.int64,types.int64))
def CyclicStack_Get(stack, count, index):
    #stack, maxsize, count = cyclic_stack_data
    return stack[index % count]

@njit(types.float64(types.float64[:],types.int64))
def CyclicStack_GetRandom(stack, count):
    #stack, maxsize, count = cyclic_stack_data
    #return stack[random.randint(0,count-1)]
    return stack[numpy.random.randint(0,count)]

#@njit(types.int64(types.int64))
#def CyclicStack_Len(count):
#    #stack, maxsize, count = cyclic_stack_data
#    return count

@njit
def restart_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count):
    if sum_X < 1:
        if is_aval_sim:
            sum_X                = 1
            X[int((len(X)-1)/2)] = 1 # seeding the middle of the network
        else:
            if M == 0: # absorbing state reached and no memory to restart
                return False
            get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
    return True

@njit
def get_IC(X0, fX0, X0Rand, N):
    X0 = int(X0)
    X  = numpy.zeros(N,dtype=numpy.int64)
    if X0Rand:
        #X[random.sample(range(N),k=int(fX0*N))] = 1.0
        get_random_state(X,fX0)
    else:
        get_ordered_state(X,fX0)
    return X

@njit
def Run_MF_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,sim,saveSites,writeOnRun,spkFileName):
    # all sites update in the same time step -- matches the GL model
    X           = get_IC(X0, fX0, X0Rand, N)     
    is_aval_sim = sim == 'aval'
    alpha       = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl        = float(N)
    sum_X       = 0
    rho_prev    = float(sum(X)) / N_fl
    # rho_memory OLD -> rho_memory[0] ### this is the stack
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans):
        sum_X = 0
        for i in range(N):
            X[i]   = state_iter(X[i],rho_prev,alpha)
            sum_X += X[i]
        # updates rho_temp and X as needed if the network activity must be restarted
        if not restart_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count):
            break
        #rho_temp        = float(rho_temp) / N_fl
        rho_prev        = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho_prev) # keeping a memory of the fraction of active sites in the previous M steps (since it is mean-field, it doesn't matter which sites are active)

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho        = numpy.zeros(tTotal-tTrans,dtype=numpy.float64)
    rho[0]     = rho_prev
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho_prev)
    for t in range(1,tTotal-tTrans):
        sum_X = 0
        for i in range(N):
            X[i] = state_iter(X[i],rho_prev,alpha)
            sum_X += X[i]
            save_spk_time(X_data, t, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            write_spk_time(X_data, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        # updates rho_temp and X as needed if the network activity must be restarted
        if not restart_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count):
            break
        rho_prev      = float(sum_X) / N_fl
        rho[t]        = rho_prev
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data

@njit
def Run_MF_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,saveSites,writeOnRun,spkFileName):
    # only 1 site is attempted update at each time step
    X          = get_IC(X0, fX0, X0Rand, N)
    alpha      = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl       = float(N)
    tTrans_eff = int(numpy.round(tTrans / dt))
    tTotal_eff = int(numpy.round(tTotal / dt))
    n_neigh    = N_fl - 1.0
    sum_X      = sum(X)
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans_eff):
        #i      = random.randint(0,N-1) # selecting update site
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        X[i]   = state_iter(X[i],float(sum_X-X[i])/n_neigh,alpha) # updating site i
        sum_X += X[i] - Xa # +1 if activated i; -1 if deactivated i
        #sum_of_X = sum(X)
        if sum_X < 1:
            if M == 0:
                break
            get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho           = numpy.zeros(tTotal_eff-tTrans_eff,dtype=numpy.float64)
    sum_X         = sum(X)
    rho[0]        = float(sum_X) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal_eff-tTrans_eff):
        #i      = random.randint(0,N-1) # selecting update site
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        X[i]   = state_iter(X[i],float(sum_X-X[i])/n_neigh,alpha) # updating site i
        sum_X += X[i] - Xa # +1 if activated i; -1 if deactivated i
        save_spk_time(X_data, t*dt, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        write_spk_time(X_data, t*dt, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        if sum_X < 1:
            if M == 0:
                break
            get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho[t]              = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data

@njit
def Run_RingGraph_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph,sim,saveSites,writeOnRun,spkFileName):
    X             = get_IC(X0, fX0, X0Rand, N)
    neigh         = get_ring_neighbors(graph,N) #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    is_aval_sim   = sim == 'aval'
    alpha         = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl          = float(N)
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans):
        X_prev = X.copy()
        sum_X  = 0
        for i in range(N):
            X[i]   = state_iter(X[i],sum(X_prev[neigh[i]])/float(len(neigh[i])),alpha)
            sum_X += X[i]
        if not restart_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count):
            break
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)
    
    rho           = numpy.zeros(tTotal-tTrans, dtype=numpy.float64)
    rho[0]        = float(sum(X)) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal-tTrans):
        X_prev = X.copy()
        sum_X  = 0
        for i in range(N):
            # need to fix this line to use X_prev in the num of active neighbors
            X[i]   = state_iter(X[i],sum(X_prev[neigh[i]])/float(len(neigh[i])),alpha)
            sum_X += X[i]
            save_spk_time(X_data, t, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            write_spk_time(X_data, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        if not restart_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count):
            break
        rho[t]        = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data

@njit
def Run_RingGraph_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph,saveSites,writeOnRun,spkFileName):
    X          = get_IC(X0,fX0,X0Rand,N)
    neigh      = get_ring_neighbors(graph,N) #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    alpha      = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    N_fl       = float(N)
    tTrans_eff = int(numpy.round(tTrans / dt))
    tTotal_eff = int(numpy.round(tTotal / dt))
    sum_X      = sum(X)
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans_eff):
        #i      = random.randint(0,N-1) # selecting update site
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        X[i]   = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha) # updating site i
        sum_X += X[i] - Xa # +1 if activated i; -1 if deactivated i
        if sum_X < 1:
            if M == 0:
                break
            get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho           = numpy.zeros(tTotal_eff-tTrans_eff, dtype=numpy.float64)
    sum_X         = sum(X)
    rho[0]        = float(sum_X) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal_eff-tTrans_eff):
        #i      = random.randint(0,N-1) # selecting update site
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        X[i]   = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha) # updating site i
        sum_X += X[i] - Xa # +1 if activated i; -1 if deactivated i
        save_spk_time(X_data, t*dt, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        write_spk_time(X_data, t*dt, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        #sum_of_X = sum(X)
        if sum_X < 1.0:
            if M == 0:
                break
            get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho[t] = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data
