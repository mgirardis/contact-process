import numpy
from enum import IntEnum
from numba import njit,types,typeof
from numba.typed import List

class GraphType(IntEnum):
    ALLTOALL = 0
    RING     = 1
    RINGFREE = 2

class StateIterType(IntEnum):
    TOME_OLIVEIRA = 0
    MARRO_DICKMAN = 1

class SimulationType(IntEnum):
    TIMEEVO = 0
    AVAL    = 1

class UpdateType(IntEnum):
    PARALLEL = 0
    SEQUENTIAL = 1

def str_to_GraphType(graph_str):
    graph_str = graph_str.lower()
    if (graph_str == 'alltoall') or (graph_str == 'mf'):
        return GraphType.ALLTOALL
    elif (graph_str == 'ring'):
        return GraphType.RING
    elif (graph_str == 'ringfree'):
        return GraphType.RINGFREE
    else:
        raise ValueError(f'Unknown graph type: {graph_str}')

def str_to_SimulationType(sim_str):
    sim_str = sim_str.lower()
    if (sim_str == 'timeevo'):
        return SimulationType.TIMEEVO
    elif (sim_str == 'aval'):
        return SimulationType.AVAL
    else:
        raise ValueError(f'Unknown simulation type: {sim_str}')

def str_to_UpdateType(updt_str):
    updt_str = updt_str.lower()
    if (updt_str == 'parallel') or (updt_str == 'par'):
        return UpdateType.PARALLEL
    elif (updt_str == 'sequential') or (updt_str == 'seq'):
        return UpdateType.SEQUENTIAL
    else:
        raise ValueError(f'Unknown update type: {updt_str}')

def str_to_StateIterType(itype_str):
    itype_str = itype_str.lower()
    if (itype_str == 'tome_oliveira') or (itype_str == 'to') or (itype_str == 'tomeoliveira'):
        return StateIterType.TOME_OLIVEIRA
    elif (itype_str == 'marro_dickman') or (itype_str == 'md') or (itype_str == 'marrodickman'):
        return StateIterType.MARRO_DICKMAN
    else:
        raise ValueError(f'Unknown state iterator type: {itype_str}')

def is_parallel_update(update):
    return update == UpdateType.PARALLEL

def Get_Simulation_Func(args):
    if args.sim == SimulationType.AVAL:
        if not is_parallel_update(args.update):
            args.update     = UpdateType.PARALLEL # forcing parallel update for avalanche
            args.expandtime = False
            print(' ::: WARNING ::: forcing parallel update and no expandtime because sim == %s'%args.sim)
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
    if is_parallel_update(args.update):
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


@njit(types.int64[:](types.int64[:],types.float64))
def get_ordered_state(X,f_act):
    N = len(X)
    K = int(f_act * N)
    for i in range(K):
        X[i] = 1
    for i in range(K,N):
        X[i] = 0
    return X

@njit(types.int64[:](types.int64[:],types.float64))
def get_random_state(X,f_act):
    # X -> site vector (in/out parameter); numpy.ndarray
    # f_act -> fraction of active elements
    N = len(X)
    X = get_ordered_state(X,f_act)
    for i in range(N-1):
        #j = random.randint(i, N-1)
        j = numpy.random.randint(i, N)
        X[i],X[j] = X[j],X[i] # Shuffle X in-place using Fisher-Yates
    return X

type_X_data_item = types.Tuple((types.float64, types.int64, types.int64))
type_X_data      = types.ListType(type_X_data_item)

@njit(type_X_data(type_X_data,types.float64,types.int64,types.int64))
def save_spk_data_fake(X_data, t, k, X):
    return X_data

@njit(type_X_data(type_X_data,types.float64,types.int64,types.int64))
def save_spk_data(X_data, t, k, X):
    if X:
        X_data.append((t,k,X))
        #X_values.append(X)
        #X_ind.append(k)
        #X_time.append(t)
    return X_data

@njit(type_X_data(type_X_data,types.float64,types.int64,types.int64))
def write_spk_data_fake(X_data,t,k,X):
    return X_data

@njit(type_X_data(type_X_data,types.float64,types.int64,types.int64))
def write_spk_data(X_data,t,k,X):
    if X:
        #spkFile.write(str(t) + ',' + str(k) + ',' + str(X) + '\n')
        print(t,',',k,',',X)
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

@njit(type_X_data(types.int64[:],types.float64,types.boolean,types.boolean))
def save_initial_network_state(X, t0, saveSites, writeOnRun):
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = List.empty_list(type_X_data_item) # get_initial_network_state_for_output(X,saveSites and not writeOnRun)
    N                            = len(X)
    for i in range(N):
        X_data = save_spk_time( X_data, t0, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        _      = write_spk_time(X_data, t0, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
    return X_data


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

@njit(types.int64(types.boolean))
def bool2int(x):
    return 1 if x else 0

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Tome_Oliveira(X,n,inv_l):
    """
     described in pg 308pdf/402 Tome Oliveira book before eq 13.6
     At each time step we choose a site at random, say site i.
       (a) If i is occupied, than we generate a random number r uniformly distributed in the interval [0;1].
           If r <= 1/lambda = inv_l, the particle is annihilated and the site becomes empty.
           Otherwise, the site remains occupied.
       (b) If i is empty, then one of its neighbors is chosen at random.
           If the neighboring site is occupied then we create a particle at site i.
           Otherwise, the site i remains empty. 
    
     X -> state of node
     n -> fraction of active neighbors of X
     inv_l -> inverse of activation rate: inv_l = 1/lambda = alpha in the book
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ Xi(t+1)=1 and Xi(t)=0 ] = P[Xi(t)=0] * P[r<n]        = (1-rho) * n  [[[ rho -> fraction of active sites in total ]]]
     and a probability of annihilation Pa:
     Pa = P[ Xi(t+1)=0 and Xi(t)=1 ] = P[Xi(t)=1] * P[r<1/lambda] = rho / lambda
     these values contrast with Pc and Pa from the Dickman algorithm (see state_iter_Dickman)
     and end-up generating a different critical point lambda_c ~ 2  [[[ periodic ring, sequential update, Dickman's lambda_c ~ 3.3 ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
     """
    #return bool2int( numpy.random.random() > inv_l ) if X else bool2int(numpy.random.random() < n)
    if X: # site is occupied
        return bool2int(numpy.random.random() > inv_l) # r > 1/lambda: stays occupied; r < 1/lambda: annihilation
    else: # site is empty
        return bool2int(numpy.random.random() < n) # r < n: infection; r>n: stays empty

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Tome_Oliveira_mod(X,n,inv_l):
    """
     MODIFIED TO MATCH THE DICKMAN algorithm
     described in pg 308pdf/402 Tome Oliveira book before eq 13.6
     also matches the description in pg 77 (87 of pdf) of the Henkel-Hinrichsen-Lubeck book
     [probabilities given after Eq. 3.35].
     Tome-Oliveira description
     [I believe they meant 1/(1+lambda) instead of 1/lambda;
     and also, creation only if random neighbor is active AND probability lambda/(1+lambda)]:
     At each time step we choose a site at random, say site i.
       (a) If i is occupied, than we generate a random number r uniformly distributed in the interval [0;1].
           If r <= 1/lambda = inv_l, the particle is annihilated and the site becomes empty.
           Otherwise, the site remains occupied.
       (b) If i is empty, then one of its neighbors is chosen at random.
           If the neighboring site is occupied then we create a particle at site i.
           Otherwise, the site i remains empty. 
    
     X -> state of node
     n -> fraction of active neighbors of X
     inv_l -> inverse of activation rate: inv_l = 1/lambda = alpha in the book
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ Xi(t+1)=1 and Xi(t)=0 ] = P[Xi(t)=0] * P[r<n]        = (1-rho) * n  [[[ rho -> fraction of active sites in total ]]]
     and a probability of annihilation Pa:
     Pa = P[ Xi(t+1)=0 and Xi(t)=1 ] = P[Xi(t)=1] * P[r<1/lambda] = rho / lambda
     these values contrast with Pc and Pa from the Dickman algorithm (see state_iter_Dickman)
     and end-up generating a different critical point lambda_c ~ 2  [[[ periodic ring, sequential update, Dickman's lambda_c ~ 3.3 ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
    """
    #return bool2int( numpy.random.random() > inv_l ) if X else bool2int(numpy.random.random() < n)
    v = 1.0/(1.0 + inv_l)
    if X: # site is occupied
        return bool2int(numpy.random.random() > inv_l*v) # r > 1/lambda: stays occupied; r < 1/lambda: annihilation
    else: # site is empty
        return bool2int(numpy.random.random() < n*v) # r < n: infection; r>n: stays empty

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Dickman_mod(X,n,v):
    """
    SEEMS TO HAVE WRONG RATES (check debug table of transition rates)
     this code was adapted from what was
     described in pg 178pdf/162book Marro & Dickman book.
     Each step involves randomly choosing a process - creation with probability v=lambda/(1+lambda),
     annihilation with probability 1-v -- and a lattice site x.
     In an annihilation event, the particle (if any) at x is removed.
     Creation proceeds only if x is occupied and a randomly chosen nearest-neighbor y is vacant;
     if so, a new particle is placed at y.
     Time is incremented by At after each step, successful or not.
     (Normally one takes Delta t = 1/N on a lattice of N sites, so that a unit time interval,
     or MC step, corresponds, on average, to one attempted event per site.)
    
     X -> state of node
     n -> fraction of active neighbors of X
     v -> lambda / (1+lambda) (creation event probability)
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ e=c and Xi(t)=1 and Xj(t)=0 ] = P[e=c] * P[Xi(t)=1] * P[Xj(t)=0] = v * rho * (1-n)  [[[ rho -> fraction of active sites in total; e=event (c or a) ]]]
          I assume, I can invert the order of neighbor and selected site, so that
     Pc = v * (1-rho) * n  [[[ i.e., current selected site is inactive and there is an active neighbor ]]]
     and a probability of annihilation Pa:
     Pa = P[ e=a and Xi(t)=1 ]             = P[e=a] * P[Xi(t)=1]              = (1-v)*rho
     these values contrast with Pc and Pa from the Tome-Oliveira algorithm (see state_iter_Tome_Oliveira)
     and generate (hopefully) lambda_c ~ 3.3  [[[ periodic ring, sequential update ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
    v = 1.0 / (1.0 + inv_l) # v === lambda / (1+lambda); but as a function of inv_l = 1/lambda
    """
    if X: # site is occupied
          # Prob = rho
          # then a particle is annihilated with chance (1-v) [[[ hence, r > v; also, implicit is the '*rho' bit in the if condition ]]]
          # otherwise nothing happens (X=1 remains)
        return bool2int(numpy.random.random() > v)
    else: # site is empty
          # Prob = 1 - rho
          # then a particle is created with chance v*n [[[*(1-rho), implicit in the if condition]]]
          # otherwise, nothing happens (X=0 remains)
        return bool2int(numpy.random.random() < v*n)

@njit(types.int64(types.int64,types.float64,types.float64))
def state_iter_Dickman(X, n, v):
    """
     described in pg 178pdf/162book Marro & Dickman book
     Each step involves randomly choosing a process - creation with probability v=lambda/(1+lambda),
     annihilation with probability 1-v -- and a lattice site x.
     In an annihilation event, the particle (if any) at x is removed.
     Creation proceeds only if x is occupied and a randomly chosen nearest-neighbor y is vacant;
     if so, a new particle is placed at y.
     Time is incremented by At after each step, successful or not.
     (Normally one takes Delta t = 1/N on a lattice of N sites, so that a unit time interval,
     or MC step, corresponds, on average, to one attempted event per site.)
    
     X -> state of node
     n -> fraction of active neighbors of X
     v -> lambda / (1+lambda) (creation event probability)
    
     This algorithm generates, at each time step, a probability of creation Pc:
     Pc = P[ e=c and Xi(t)=1 and Xj(t)=0 ] = P[e=c] * P[Xi(t)=1] * P[Xj(t)=0] = v * rho * (1-n)  [[[ rho -> fraction of active sites in total; e=event (c or a) ]]]
          I assume, I can invert the order of neighbor and selected site, so that
     Pc = v * (1-rho) * n  [[[ i.e., current selected site is inactive and there is an active neighbor ]]]
     and a probability of annihilation Pa:
     Pa = P[ e=a and Xi(t)=1 ]             = P[e=a] * P[Xi(t)=1]              = (1-v)*rho
     these values contrast with Pc and Pa from the Tome-Oliveira algorithm (see state_iter_Tome_Oliveira)
     and generate (hopefully) lambda_c ~ 3.3  [[[ periodic ring, sequential update ]]]
    
     returns the new state based on the previous state X for a given node
             site is occupied, so it eliminates               site is empty, so it creates
             the particle with prob 1/lambda                 the particle with the same chance as that of finding an active neighbor
    v = 1.0 / (1.0 + inv_l) # v === lambda / (1+lambda); but as a function of inv_l = 1/lambda
    """
    if numpy.random.random() < v:
        # Birth attempt
        return 1 if ((X == 0) and (numpy.random.random() < n)) else X
    else:
        # Death attempt
        return 0 if X == 1 else X


#type_state_iter = typeof(state_iter_Dickman)
type_state_iter = types.FunctionType(types.int64(types.int64, types.float64, types.float64))
@njit(type_state_iter(types.int64))
def get_site_state_iterator(iterdynamics):
    if iterdynamics == StateIterType.TOME_OLIVEIRA:
        state_iter  = state_iter_Tome_Oliveira_mod
    else:
        state_iter  = state_iter_Dickman
    return state_iter

@njit(types.float64(types.int64, types.float64))
def get_site_state_iterator_alpha(iterdynamics,l):
    if iterdynamics == StateIterType.TOME_OLIVEIRA:
        alpha       = 1.0 / l # chance of annihilating if site is occupied, book TOme e Oliveira
    else:
        alpha       = l / (1.0 + l) # v, book of Marro & Dickman
    return alpha

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
def check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count):
    # returns True if activity must continue
    #         False if activity should die out
    if sum_X < 1: # activity died out
        if is_aval_sim: # if it is a simulation for avalanches
            # we always restart the activity
            sum_X                = 1
            X[int((len(X)-1)/2)] = 1 # seeding the middle of the network
        else:
            # otherwise, we pick a state from the memory
            # only if we have memory states (M>0)
            if M == 0: # absorbing state reached and no memory to restart
                return False, X, sum_X
            X     = get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
    return True, X, sum_X

@njit(types.int64[:](types.int64,types.float64,types.boolean,types.int64))
def get_IC(X0, fX0, X0Rand, N):
    X0 = int(X0)
    X  = numpy.zeros(N,dtype=numpy.int64)
    if X0Rand:
        #X[random.sample(range(N),k=int(fX0*N))] = 1.0
        X = get_random_state(X,fX0)
    else:
        X = get_ordered_state(X,fX0)
    return X

@njit
def Run_MF_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,iterdynamics,sim,saveSites,writeOnRun,spkFileName):
    # all sites update in the same time step -- matches the GL model
    X                   = get_IC(X0, fX0, X0Rand, N)     
    is_aval_sim         = sim == SimulationType.AVAL
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    sum_X               = 0
    rho_prev            = float(sum(X)) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans):
        sum_X = 0
        for i in range(N):
            X[i]   = state_iter(X[i],rho_prev,alpha)
            sum_X += X[i]
        # updates rho_temp and X as needed if the network activity must be restarted
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        #rho_temp        = float(rho_temp) / N_fl
        rho_prev        = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho_prev) # keeping a memory of the fraction of active sites in the previous M steps (since it is mean-field, it doesn't matter which sites are active)

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho                 = numpy.zeros(tTotal-tTrans,dtype=numpy.float64)
    rho[0]              = rho_prev
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho_prev)
    for t in range(1,tTotal-tTrans):
        sum_X = 0
        for i in range(N):
            X[i]   = state_iter(X[i],rho_prev,alpha)
            sum_X += X[i]
            X_data = save_spk_time(X_data, t, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            _      = write_spk_time(X_data, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        # updates rho_temp and X as needed if the network activity must be restarted
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        rho_prev      = float(sum_X) / N_fl
        rho[t]        = rho_prev
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data

@njit
def Run_MF_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,iterdynamics,saveSites,writeOnRun,spkFileName):
    # only 1 site is attempted update at each time step
    X                   = get_IC(X0, fX0, X0Rand, N)
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    tTrans_eff          = int(numpy.round(tTrans / dt))
    tTotal_eff          = int(numpy.round(tTotal / dt))
    n_neigh             = N_fl - 1.0
    sum_X               = sum(X)
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
            X     = get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)

    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho                 = numpy.zeros(tTotal_eff-tTrans_eff,dtype=numpy.float64)
    sum_X               = sum(X)
    rho[0]              = float(sum_X) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal_eff-tTrans_eff):
        #i      = random.randint(0,N-1) # selecting update site
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        X[i]   = state_iter(X[i],float(sum_X-X[i])/n_neigh,alpha) # updating site i
        sum_X += X[i] - Xa # +1 if activated i; -1 if deactivated i
        X_data = save_spk_time( X_data, t*dt, i, X[i]-Xa) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        _      = write_spk_time(X_data, t*dt, i, X[i]-Xa) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        if sum_X < 1:
            if M == 0:
                break
            X     = get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho[t]              = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data

@njit
def Run_RingGraph_parallel(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph,iterdynamics,sim,saveSites,writeOnRun,spkFileName):
    X                   = get_IC(X0, fX0, X0Rand, N)
    neigh               = get_ring_neighbors(graph,N) #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    is_aval_sim         = sim == SimulationType.AVAL
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,fX0)
    for t in range(1,tTrans):
        X_prev = X.copy()
        sum_X  = 0
        for i in range(N):
            X[i]   = state_iter(X[i],sum(X_prev[neigh[i]])/float(len(neigh[i])),alpha)
            sum_X += X[i]
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)
    
    rho                 = numpy.zeros(tTotal-tTrans, dtype=numpy.float64)
    rho[0]              = float(sum(X)) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal-tTrans):
        X_prev = X.copy()
        sum_X  = 0
        for i in range(N):
            # need to fix this line to use X_prev in the num of active neighbors
            X[i]   = state_iter(X[i],sum(X_prev[neigh[i]])/float(len(neigh[i])),alpha)
            sum_X += X[i]
            X_data = save_spk_time(X_data, t, i, X[i]) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
            _      = write_spk_time(X_data, t, i, X[i])               # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        continue_time_loop, X, sum_X = check_network_activity(X, is_aval_sim, sum_X, rho_memory, M, cs_count)
        if not continue_time_loop:
            break
        rho[t]        = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,rho[t])
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data

@njit
def Run_RingGraph_sequential(N,X0,fX0,X0Rand,l,tTrans,tTotal,dt,M,graph,iterdynamics,saveSites,writeOnRun,spkFileName):
    X                   = get_IC(X0,fX0,X0Rand,N)
    neigh               = get_ring_neighbors(graph,N) #neigh[i][0] -> index of left neighbor; neigh[i][1] -> index of right neighbor;
    state_iter          = get_site_state_iterator(iterdynamics)
    alpha               = get_site_state_iterator_alpha(iterdynamics,l)
    N_fl                = float(N)
    tTrans_eff          = int(numpy.round(tTrans / dt))
    tTotal_eff          = int(numpy.round(tTotal / dt))
    sum_X               = sum(X)
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
            X     = get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    # defining output functions and data variables
    write_spk_time,save_spk_time = get_write_spike_data_functions(saveSites,writeOnRun)
    X_data                       = save_initial_network_state(X, 0.0, saveSites, writeOnRun)
    spk_file                     = open_file(spkFileName, saveSites and writeOnRun)

    rho                 = numpy.zeros(tTotal_eff-tTrans_eff, dtype=numpy.float64)
    sum_X               = sum(X)
    rho[0]              = float(sum_X) / N_fl
    rho_memory,cs_count = CyclicStack_Init(M)
    rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,0,rho[0])
    for t in range(1,tTotal_eff-tTrans_eff):
        #i      = random.randint(0,N-1) # selecting update site
        i      = numpy.random.randint(0,N) # selecting update site
        Xa     = X[i]
        X[i]   = state_iter(X[i],sum(X[neigh[i]])/float(len(neigh[i])),alpha) # updating site i
        sum_X += X[i] - Xa # +1 if activated i; -1 if deactivated i
        X_data = save_spk_time( X_data, t*dt, i, X[i]-Xa) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        _      = write_spk_time(X_data, t*dt, i, X[i]-Xa) # this function can just be a dummy placeholder depending on saveSites and writeOnRun
        #sum_of_X = sum(X)
        if sum_X < 1:
            if M == 0:
                break
            X     = get_random_state(X, CyclicStack_GetRandom(rho_memory,cs_count))
            sum_X = sum(X)
        rho[t] = float(sum_X) / N_fl
        rho_memory,cs_count = CyclicStack_Set(rho_memory,M,cs_count,t,float(sum_X) / N_fl)
    close_file(spk_file,spkFileName,saveSites and writeOnRun)
    return rho, X_data
