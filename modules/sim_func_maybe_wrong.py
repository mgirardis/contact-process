import numpy
import random
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