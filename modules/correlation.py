import math
import numpy
import scipy.stats
import scipy.signal
from numba import njit
from enum import IntEnum
from modules.io import structtype

class SmoothingType(IntEnum):
    NONE       = 0
    GAUSSIAN   = 1
    MEXICAN    = 2
    MOVING_AVG = 3

class FilterType(IntEnum):
    NONE       = 0
    MEDIAN     = 1
    LOWPASS    = 2
    MOVING_AVG = 3

class PositionMFType(IntEnum):
    RING      = 0
    LATTICE2D = 1
    RANDOM2D  = 2

@njit
def _largest_factors(N):
    for i in range(N // 2, 0, -1):
        if N % i == 0:
            return (N // i, i)
    return (N, 1)  # Numba prefers fixed types; avoid None


@njit
def calc_1d_positions_periodicBC(N):
    # position every site along a ring, since we are using periodic BC
    # r[k,:] -> (x,y) position of site k
    theta = numpy.linspace(0, 2.0 * numpy.pi, N + 1)[:N]
    r     = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    for i in range(N):
        r[i, 0] = numpy.cos(theta[i])  # x-coordinate
        r[i, 1] = numpy.sin(theta[i])  # y-coordinate
    return r

@njit
def calc_1d_positions_freeBC(N):
    # position every site along the x-axis
    # r[k,:] -> (x,y) position of site k
    r = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    for i in range(N):
        r[i, 0] = float(i)  # x-coordinate
        r[i, 1] = 0.0       # y-coordinate
    return r

@njit
def calc_2d_positions(N):
    r     = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    Lx,Ly = _largest_factors(int(N))
    for i in range(Ly):
        for j in range(Lx):
            k = j + i * Lx
            r[k,0] = float(j)
            r[k,1] = float(i)
    return r

@njit
def calc_random_positions(N):
    Lx,Ly = _largest_factors(int(N))
    r     = numpy.empty((N, 2))  # Pre-allocate the 2D position array
    for i in range(N):
        r[i,0] = numpy.random.rand()*Lx
        r[i,1] = numpy.random.rand()*Ly
    return r

@njit
def calc_correlation_function_MF(C,avg_same_distance=True, position:PositionMFType = 0):
    if position == PositionMFType.RING:
        r = calc_1d_positions_periodicBC(C.shape[0])
    elif position == PositionMFType.LATTICE2D:
        r = calc_2d_positions(C.shape[0])
    elif position == PositionMFType.RANDOM2D:
        r = calc_random_positions(C.shape[0])
    else:
        raise ValueError('unknown position type')
    return calc_correlation_function_numba(C,r,avg_same_distance)

@njit
def calc_correlation_function_1d_periodicBC(C,avg_same_distance=True):
    r = calc_1d_positions_periodicBC(C.shape[0])
    return calc_correlation_function_numba(C,r,avg_same_distance)

@njit
def calc_correlation_function_1d_freeBC(C,avg_same_distance=True):
    r = calc_1d_positions_freeBC(C.shape[0])
    return calc_correlation_function_numba(C,r,avg_same_distance)

@njit
def calc_correlation_function_numba(C,r,avg_same_distance):
    N  = C.shape[0]
    s  = [] # we are defining C(s) as the correlation function
    Cf = [] # we are defining C(s) as the correlation function
    for i in range(N):
        for j in range(i+1,N):
            s.append(numpy.linalg.norm(r[i,:]-r[j,:]))# distance between i,j
            Cf.append(C[i,j])
    k = numpy.argsort(numpy.array(s))
    if avg_same_distance:
        return calc_average_same_distance(numpy.array(s)[k],numpy.array(Cf)[k])
    else:
        return numpy.array(s)[k],numpy.array(Cf)[k],numpy.zeros_like(k,dtype=numpy.float64)

@njit
def calc_average_same_distance(s, Cf):
    # Round s to reduce floating-point precision issues
    s_rounded = numpy.round(s, decimals=8)  # You can adjust decimals as needed

    # Sort s and Cf together by s_rounded
    idx = numpy.argsort(s_rounded)
    s_sorted = s_rounded[idx]
    Cf_sorted = Cf[idx]

    # Initialize output lists
    s_un   = []
    Cf_avg = []
    Cf_std = []

    # Average all Cf values that share the same s
    i = 0
    while i < len(s_sorted):
        sc = s_sorted[i]
        c  = Cf_sorted[i]
        c2 = Cf_sorted[i]*Cf_sorted[i]
        #sum_val   = Cf_sorted[i]
        count = 1
        i += 1
        while i < len(s_sorted) and s_sorted[i] == sc:
            c     += Cf_sorted[i]
            c2    += Cf_sorted[i]*Cf_sorted[i]
            count += 1
            i     += 1
        s_un.append(sc)
        Cf_avg.append(c / count)
        Cf_std.append(c2 / count - c*c/(count*count))
    return numpy.array(s_un), numpy.array(Cf_avg), numpy.array(Cf_std)

@njit
def my_exp(x): #definimos essa função para evitar o erro de overflow
    return math.exp(x) if x < 709.782712893384 else numpy.inf

@njit
def PoissonProcess_firingprob(r):
    return 1.0-my_exp(-r) # probability of firing is constant

@njit
def generate_Poisson_spikes(r, T, N):
    """
    Generates a matrix of Poisson spikes.
    Each column is an independent Poisson process with rate r.
    r -> Poisson rate [P=1-exp(-r) is the firing probability]
    T -> total number of time steps
    N -> number of independent processes (neurons)
    Returns a 2D numpy.ndarray of shape (T,N) where each element is either 0 or 1.
    """
    P = PoissonProcess_firingprob(r)
    X = numpy.zeros((T,N), dtype=numpy.float64)
    for k in range(N):
        X[numpy.random.rand(T)<P,k] = 1.0
    return X

def get_empty_list(N):
    return [ None for _ in range(N)]

def get_n_max_corrcoef(C,n=10,i=None,j=None):
    if type(C) is list:
        i,j = numpy.nonzero(numpy.tril(numpy.ones(C[0].shape)))
        ncf = get_empty_list(len(C))
        ind = get_empty_list(len(C))
        linind = get_empty_list(len(C))
        for k,c in enumerate(C):
            ncf[k],ind[k],linind[k] = get_n_max_corrcoef(c,n,i=i,j=j)
    else:
        CC = C.copy()
        if (i is None) or (j is None):
            i,j = numpy.nonzero(numpy.tril(numpy.ones(C[0].shape)))
        CC[i,j] = -numpy.inf
        CC[numpy.isnan(CC)] = -numpy.inf
        CC = CC.flatten()
        k = numpy.argsort(CC)[-n:]
        ncf = CC[k]
        linind = k
        z = numpy.unravel_index(k,C.shape)
        k = numpy.lexsort((z[1],z[0]))
        ncf = ncf[k]
        linind = linind[k]
        ind = [ (z[0][i],z[1][i]) for i in k ]
        #ind = [ (i,j) for i,j in zip(z[0],z[1]) ]
    return ncf,ind,linind

def calc_correlation_distribution(C,nbins=100,smooth=False,ignoreZeroCorr=True):
    if smooth is True:
        smooth = 'average'
    if type(C) is list:
        n = len(C)
        P = get_empty_list(n)
        bins = get_empty_list(n)
        avg = get_empty_list(n)
        std = get_empty_list(n)
        for i,c in enumerate(C):
            P[i],bins[i],avg[i],std[i] = calc_correlation_distribution(c,nbins=nbins,smooth=smooth,ignoreZeroCorr=ignoreZeroCorr)
        return P, bins, avg, std
    else:
        x = C[numpy.eye(C.shape[0])!=1]
        if ignoreZeroCorr:
            x = x[x!=0]
        P,bins = numpy.histogram(x,bins=nbins,density=True)
        if smooth == 'savgol':
            P = scipy.signal.savgol_filter(P, 5, 2)
        elif smooth == 'average':
            P = moving_average(P,n=10)
        P = P / numpy.sum(P)
        avg = numpy.nanmean(x)
        std = numpy.nanstd(x)
        return P, bins[:-1], avg, std

def calc_null_correlation(S,ntrials=None,**corr_kwargs):
    """
    Computes a null (baseline) correlation matrix by averaging randomized versions of the original correlation matrix.

    This function generates a null model of the correlation structure in the spike time series `S` by repeatedly
    randomizing the off-diagonal elements of the original correlation matrix and averaging the results over multiple trials.

    Parameters:
    ----------
    S : ndarray
        A 2D array representing spike time series. Each row or column corresponds to a time point or neuron,
        depending on the `rowvar` flag.

    ntrials : int, optional
        Number of randomization trials to perform. If None, defaults to the number of time points in `S`.

    corr_kwargs : dict, optional
        Additional keyword arguments passed to `calc_correlation_matrices`.

    Returns:
    -------
    C : ndarray
        A null correlation matrix obtained by averaging `ntrials` randomized versions of the original matrix.

    Notes:
    -----
    - The original correlation matrix is computed using `calc_correlation_matrices`.
    - Randomization is applied only to the upper triangular off-diagonal elements.
    - The diagonal values are preserved or set to NaN depending on `nandiag`.
    """
    if ntrials is None:
        ntrials = S.shape[0]
    A,_ = calc_correlation_matrices(S,**corr_kwargs)
    C   = A.copy()
    i,j = numpy.nonzero(numpy.triu(numpy.ones(A.shape),k=1))
    for k in range(ntrials):
        C += rand_corr_matrix(A.copy(),i=i,j=j)
    return C / ntrials

def calc_dispersion_PCA(C):
    """
    Computes the eigenvalues, dispersion (standard deviation) of principal components,
    and eigenvectors from a covariance matrix using Principal Component Analysis (PCA).

    Parameters:
    -----------
    C : numpy.ndarray
        A square covariance matrix (n x n) representing the relationships (covariance matrix == physical correlation matrix) between variables.

    Returns:
    --------
    lambda_eig : numpy.ndarray
        Array of eigenvalues of the covariance matrix, representing the variance explained by each principal component.

    lambda_dispersion : numpy.ndarray
        Array of dispersions (standard deviations) of the principal components, calculated as the square root of the absolute eigenvalues.

    eigenvectors : list of numpy.ndarray
        List of eigenvectors corresponding to each principal component, each as a 1D array.

    V_matrix : numpy.ndarray
        Matrix whose columns are the eigenvectors of the covariance matrix.

    Notes:
    ------
    - The eigenvalues may be complex if the input matrix is not symmetric.
    - The function takes the absolute value of eigenvalues before computing the square root to ensure real-valued dispersions.
    """
    if type(C) is list:
        return _unpack_list_of_tuples([ _calc_dispersion_PCA_numba(c) for c in C ])
    return _calc_dispersion_PCA_numba(C)

@njit
def _calc_dispersion_PCA_numba(C):
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if numpy.isnan(C[i, j]):
                C[i, j] = 0.0
    lambda_eig,V_matrix = numpy.linalg.eig(C)
    lambda_dispersion   = numpy.sqrt(numpy.abs(lambda_eig))
    eigenvectors        = [ V_matrix[:,m].flatten() for m in range(V_matrix.shape[1]) ]
    return lambda_eig,lambda_dispersion,eigenvectors,V_matrix

def calc_correlation_matrices(S,DeltaT=None,overlap_DeltaT=False,
    smooth : SmoothingType = False, smooth_stddev=0.1,smooth_J=None,smooth_mvavg_kernel_size=10,
    binarize=True,spk_threshold=59.0,
    rowvar=False,nandiag=True,filterSpkFreq:FilterType=False,filter_kernel_size=3,filter_null_avg_ntrials=10):
    """
    Computes a sequence of correlation (covariance) matrices from a spike time series matrix `S`.
    If DeltaT is given, the time series is divided into a sequence of (possibly overlapping) intervals of length 'DeltaT'.
    If both smoothing and filtering are applied, the spike data is first smoothed, and then filtered.
    If binarization is applied, the spike data is converted to binary before smoothing and filtering.

    Parameters:
    ----------
    S : ndarray
        A 2D array representing spike time series. Each row or column corresponds to a time point or neuron,
        depending on the `rowvar` flag. Default: rows are time points and columns are neurons.

    DeltaT : int, optional
        Length of each time interval (in number of rows of `S`) used to compute individual correlation matrices.
        If None, the entire time series is used as a single interval.
    overlap_DeltaT : bool, optional
        If True, uses overlapping time intervals for correlation computation. Otherwise, uses adjacent intervals.

    smooth : SmoothingType, optional
        Whether to apply smoothing to the spike data. See 'SmoothingType' for options.
    smooth_stddev : float, optional
        Standard deviation used for Gaussian smoothing. Ignored if `smooth` is False or 'mexican'.
    smooth_J : any, optional
        Additional parameter passed to the smoothing function. Typically used for 'mexican' smoothing.

    binarize : bool, optional
        If True, converts the spike data to binary format using a threshold.
    spk_threshold : float, optional
        Threshold used to binarize spike data. Values above this are considered spikes.

    rowvar : bool, optional
        If True, treats rows as variables (neurons) and columns as observations (time points).
        If False, treats columns as variables.
    nandiag : bool, optional
        If True, sets the diagonal of each correlation matrix to NaN to ignore self-correlations.

    filterSpkFreq : FilterType, optional
        Whether to apply filtering to the spike series before computing correlations. See FilterType for options.
    filter_kernel_size : int, optional
        Size of the kernel (number of time steps) used for filtering spike frequencies by median. Default is 3.
    filter_null_avg_ntrials : int, optional
        Number of trials used to average randomized correlation matrices. Default is 10.

    Returns:
    -------
    C : ndarray or list of ndarrays
        A single correlation matrix if only one interval is used, or a list of matrices for each interval.

    tRange : tuple or list of tuples
        A single tuple (start, end) if only one interval is used, or a list of such tuples for each interval.

    Notes:
    -----
    - NaN values in the correlation matrices are replaced with 0.0, except on the diagonal if `nandiag` is True.
    - The function uses covariance (`numpy.cov`) rather than correlation (`numpy.corrcoef`) for matrix computation.
    """
    if rowvar:
        S = numpy.transpose(S)
    if DeltaT is None:
        DeltaT = S.shape[0]
    if binarize:
        S = get_binary_spike_series(S,spk_threshold=spk_threshold)
    if smooth:
        S = smooth_spikes(S,smooth=smooth,stddev=smooth_stddev,J=smooth_J,kernel_size=smooth_mvavg_kernel_size)
    if filterSpkFreq:
        S = filter_spikes(S,kernel_size=filter_kernel_size,nullavg_ntrials=filter_null_avg_ntrials)
    if overlap_DeltaT:
        nT             = S.shape[0] - 1
        get_time_range = _get_corr_timerange_overlap
    else:
        nT             = int(numpy.ceil(float(S.shape[0]) / float(DeltaT)))
        get_time_range = _get_corr_timerange_adjacent
    
    C      = get_empty_list(nT)
    tRange = get_empty_list(nT)
    for n in range(nT):
        t1,t2 = get_time_range(n,DeltaT)
        if t2 > S.shape[0]:
            t2 = S.shape[0]
        if (t2-t1) > 2:
            tRange[n] = (t1,t2)
            C[n] = numpy.cov(S[t1:t2,:],rowvar=rowvar) #numpy.corrcoef(S[t1:t2,:],rowvar=rowvar)
            C[n][numpy.isnan(C[n])] = 0.0
            #if numpy.count_nonzero(numpy.isnan(C[i])) > 0:
            #print('index == %d -> [%d;%d]     ---- number of NaN: %d' % (i,t1,t2,numpy.count_nonzero(numpy.isnan(C[i]))))
            if nandiag:
                numpy.fill_diagonal(C[n],numpy.nan)

    if nT == 1: # squeeze output
        C      = C[0]
        tRange = tRange[0]
    else: # remove None values
        C      = [c for c in C      if c is not None]
        tRange = [t for t in tRange if t is not None] 
    return C, tRange

def filter_spikes(S,filter_type:FilterType=None,kernel_size=3,fs=10.0,cutoff=1.0,order=5):
    if filter_type == FilterType.NONE or filter_type is False:
        return S
    if filter_type == FilterType.MEDIAN:
        return filter_spk_freq_median(S, kernel_size)
    elif filter_type == FilterType.LOWPASS:
        return filter_butter_lowpass(S, cutoff, fs, order)
    elif filter_type == FilterType.MOVING_AVG:
        return numpy.apply_along_axis(moving_average, 0, S, n=kernel_size)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Supported types are given by FilterType enum.")

def _filter_create_butter_lowpass(cutoff, fs, order=5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype='low', analog=False)

def filter_butter_lowpass(S, cutoff, fs, order=5):
    b, a = _filter_create_butter_lowpass(cutoff, fs, order=order)
    n    = S.shape[1]
    for i in range(n):
        S[:,i] = scipy.signal.lfilter(b, a, S[:,i])
    return S

def filter_spk_freq_median(S,kernel_size=3):
    """ to be implemented: remove background spikes using median filter """
    n = S.shape[1]
    for i in range(n):
        S[:,i] = scipy.signal.medfilt(S[:,i],kernel_size)
    return S

def filter_corrmatrix_null_avg(C,null_avg):
    if type(C) is list:
        for n in range(len(C)):
            C[n] = filter_corrmatrix_null_avg(C[n],null_avg)
    else:
        C[numpy.nonzero(C < null_avg)] = 0.0
    return C

def _get_corr_timerange_overlap(n,DeltaT):
    return n, n+DeltaT

def _get_corr_timerange_adjacent(n,DeltaT):
    return n*DeltaT, (n+1)*DeltaT

def rand_corr_matrix(A, i=None, j=None):
    """
    Randomizes the off-diagonal upper triangular elements of a symmetric correlation matrix.

    This function generates a randomized version of a symmetric matrix `A` by shuffling its upper triangular
    off-diagonal elements. The diagonal is preserved, and symmetry is maintained by mirroring the randomized
    upper triangle to the lower triangle.

    Parameters:
    ----------
    A : ndarray
        A square symmetric matrix (typically a correlation or covariance matrix).

    i : ndarray, optional
        Row indices of the upper triangular off-diagonal elements to be randomized.
        If None, these indices are automatically computed.

    j : ndarray, optional
        Column indices corresponding to `i`. If None, these are automatically computed.

    Returns:
    -------
    A_rand : ndarray
        A symmetric matrix with the same diagonal as `A`, but with randomized off-diagonal elements.

    Notes:
    -----
    - Only the upper triangular part (excluding the diagonal) is randomized.
    - The resulting matrix is symmetric: `A_rand[i, j] == A_rand[j, i]`.
    - Useful for generating null models or testing statistical significance of correlation structures.
    """
    if i is None or j is None:
        i,j = numpy.nonzero(numpy.triu(numpy.ones(A.shape),k=1))
    x = A[i,j] # gets all the upper triangular elements in A
    x = x[numpy.random.permutation(len(x))]
    A[i,j] = x
    return numpy.triu(A,k=1) + numpy.tril(numpy.transpose(A))

def get_binary_spike_series(S,spk_threshold=59.0):
    return numpy.asarray(S>spk_threshold,dtype=float)

def smooth_spikes(S, smooth:SmoothingType=None, dt=0.01, stddev=0.1, J=None, kernel_size=10):
    """
    Applies temporal smoothing to binary spike time series using convolution.

    This function smooths each column of the input matrix `S`, which represents binary spike trains (0s and 1s),
    by convolving them with a specified smoothing kernel. If no kernel is provided, a Gaussian kernel is used by default.

    Parameters:
    ----------
    S : ndarray
        A 2D array where each column represents a binary spike time series for a neuron.
    smooth : SmoothingType, optional
        The smoothing kernel to use. See SmoothingType for options.
    dt : float, optional
        Time resolution of the spike data (used when generating kernels). Default is 0.01.
    stddev : float, optional
        Standard deviation of the Gaussian or Mexican hat kernel. Controls the smoothing scale.
    J : float or None, optional
        Additional parameter used when generating the Mexican hat kernel.
    kernel_size : int, optional
        kernel size used for moving average if 'movingavg' is specified as 'smoothFunc'. Default is 10.

    Returns:
    -------
    S_smooth : ndarray
        A 2D array of the same shape as `S`, where each column has been smoothed via convolution.

    Notes:
    -----
    - Convolution is performed with `mode='same'`, preserving the original length of each spike train.
    - This function is useful for estimating firing rates or preparing spike data for correlation analysis.
    """
    N = S.shape[1]
    if smooth == SmoothingType.NONE or smooth is False:
        return S
    if smooth == SmoothingType.MOVING_AVG:
        kernel = numpy.ones(kernel_size,dtype=float) / float(kernel_size)
    elif smooth == SmoothingType.GAUSSIAN:
        kernel = get_gaussian_kernel(stddev,dt=dt)
    elif smooth == SmoothingType.MEXICAN:
        kernel = get_mexican_hat_kernel(stddev, J=J, dt=dt)
    else:
        raise ValueError(f"Unknown smoothing type: {smooth}. Supported types are given by SmoothingType enum.")
    for i in range(N):
        S[:,i] = numpy.convolve(S[:,i],kernel,mode='same')
    return S

def moving_average(x, n=10):
    return numpy.convolve(x,numpy.ones(n)/n,mode='same')

def get_mexican_hat_kernel(sigma1, J=None, dt=None):
    if J is None:
        J=4.0*sigma1
    if dt is None:
        dt = 0.001
    sigma2 = numpy.sqrt(numpy.power(sigma1, 2.) + numpy.power(J, 2.))
    if sigma2 < sigma1:
        sigma1, sigma2 = sigma2, sigma1
    k1 = get_gaussian_kernel(sigma1,dt=dt)
    k2 = get_gaussian_kernel(sigma2,dt=dt)
    n2 = k2.shape[0]
    n1 = k1.shape[0]
    m = int( numpy.floor((n2-n1)/2.0) )
    n = int( numpy.ceil((n2-n1)/2.0) )
    return numpy.pad(k1,(m,n))-k2

def get_gaussian_kernel(sigma,dt=0.01):
    t = numpy.arange(-3*sigma,3*sigma,dt)
    G = scipy.stats.norm.pdf(t,scale=sigma) * dt
    return G / numpy.sum(G)


@njit
def _spike_times_to_spiketrain_numba(t, n, X, T, N, use_X):
    S = numpy.zeros((T+1, N), dtype=numpy.float64)
    for i in range(t.shape[0]):
        time = t[i]
        neuron = n[i]
        S[time, neuron] = X[i] if use_X else 1.0
    return S

def spike_times_to_spiketrain(t, n, X=None, T=None, N=None):
    t = numpy.asarray(t, dtype=numpy.int32)
    n = numpy.asarray(n, dtype=numpy.int32)
    T = int(T) if T is not None else int(numpy.max(t))
    N = int(N) if N is not None else int(numpy.max(n))
    if _exists(X):
        X = numpy.asarray(X, dtype=numpy.float64)
        assert X.shape == t.shape, 'X must match shape of t'
        use_X = True
    else:
        X = numpy.zeros_like(t, dtype=numpy.float64)
        use_X = False
    return _spike_times_to_spiketrain_numba(t, n, X, T, N, use_X)


#def spike_times_to_spiketrain(t,n,X=None,T=None,N=None):
#    t = (t if _is_numpy_array(t) else numpy.asarray(t)).astype(int)
#    n = (n if _is_numpy_array(n) else numpy.asarray(n)).astype(int)
#    T = int(T if _exists(T) else numpy.max(t))
#    N = int(N if _exists(N) else numpy.max(n))
#    if _exists(X):
#        X = X if _is_numpy_array(X) else numpy.asarray(X)
#        assert X.shape == t.shape, 'The input X must be the state of each node n at time t, so it must match the type and shape of t.'
#    _type = X.dtype if _exists(X) else float
#    S     = numpy.zeros((T+1,N),dtype=_type)
#    for i in range(N):
#        ind         = numpy.nonzero(n == i)[0]
#        S[t[ind],i] = X[ind] if _exists(X) else 1.0
#    return S

def membpotential_to_spiketrain(V_data,t=None,spk_threshold=59.0):
    # converts each column of data into a numpy binary array of spike trains
    # t is the time vector
    (T,N) = V_data.shape
    spktrains = get_empty_list(N)
    if t is None:
        t = numpy.arange(T)
    #dt = numpy.mean(numpy.squeeze(numpy.diff(t)))
    for j in range(N):
        spktrains[j] = numpy.zeros(V_data[:,j].size, dtype=float)
        spktrains[j][V_data[:,j]>spk_threshold] = 1.0  #neo.SpikeTrain(t[numpy.nonzero(data[:,j] > spk_threshold)], units='ms', t_start=t[0], t_stop=t[-1])
    return spktrains

def membpotential_to_spike_times(V_data,t=None,spk_threshold=59.0):
    # converts each column of data into an array of spike times
    # t is the time vector
    (T,N) = V_data.shape
    spktimes = get_empty_list(N)
    if t is None:
        t = numpy.arange(T)
    for j in range(N):
        spktimes[j] = t[numpy.nonzero(V_data[:,j] > spk_threshold)]
    return spktimes

def _get_unique_pair_indices(k):
    if not (type(k) is list):
        _,a = numpy.unique(k, axis=0, return_inverse=True)
        return k
    n = len(k)
    a = numpy.array(k).flatten()
    _,a = numpy.unique(a, axis=0, return_inverse=True)
    return _split_array(a,n)

def _split_array(a, n):
    k, m = divmod(len(a), n)
    return numpy.array([ a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n) ])

def _flatten_tuple_list(l):
    r = []
    for e in l:
        if type(e) is list:
            r += _flatten_tuple_list(e)
        else:
            r.append(e)
    return r

def _sort_tuple_list(l,k):
    return [ l[i] for i in k ]

def _unpack_list_of_tuples(lst):
    """
    given a list of len m where element is an n-tuple, then returns an n-tuple where each element containins m elements
    (similar to numpy.transpose)

    this is useful to unpack a list comprehension applied to a function that has multiple returns
    """
    return tuple( list(x) for x in zip(*lst) )

def _exists(X):
    return not(type(X) is type(None))

def _is_numpy_array(X):
    return isinstance(X,numpy.ndarray)

def linearized_fit(x_data, y_data, x_transform, y_transform, inverse_transform, mask=None):
    """
    Performs linear regression on transformed data using scipy.stats.linregress.

    Parameters:
    ----------
    x_data : array-like
        Original independent variable values.
    y_data : array-like
        Original dependent variable values.
    x_transform : callable
        Function to transform x_data for linearization.
    y_transform : callable
        Function to transform y_data for linearization.
    inverse_transform : callable
        Function to reverse the y_transform for plotting fitted curve in original space.
    mask : numpy-compatible index [e.g., range, int, logical, etc]
        if present, only performs fit on the masked data: x_data[mask], y_data[mask]

    Returns:
    -------
    coeffs : tuple
        (slope, intercept) of the linear fit in transformed space.
    r_squared : float
        Coefficient of determination (R²).
    residual_sum_squares : float
        Residual sum of squares in original space.
    x_fit : ndarray
        Dense x values for plotting the fitted curve.
    y_fit : ndarray
        Fitted y values in original space.
    """
    # Transform the data
    if _exists(mask):
        x_lin = x_transform(x_data[mask])
        y_lin = y_transform(y_data[mask])
    else:
        x_lin = x_transform(x_data)
        y_lin = y_transform(y_data)

    # Perform linear regression
    slope, intercept, r_value, _, _ = scipy.stats.linregress(x_lin, y_lin)

    # Generate fitted values in original space
    x_fit = numpy.linspace(min(x_data), max(x_data), 100)
    func  = lambda x,*fitpar: inverse_transform(fitpar[0] * x_transform(x) + fitpar[1])
    y_fit = func(x_fit,slope,intercept)

    # Compute residuals and RSS in original space
    y_pred    = inverse_transform(slope * x_lin + intercept)
    residuals = (y_data[mask] if _exists(mask) else y_data) - y_pred
    rss       = numpy.sum(residuals**2)

    return structtype(func=func,fitpar=(slope, intercept),R2=r_value**2,res_sum_sqr=rss,x_fit=x_fit,y_fit=y_fit)
