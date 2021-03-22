import numpy as np

def gradient_ftl(M_flat, *args):
    """
    from onlinelds.py
    """
    n = args[0]
    m = args[1]
    t_t = args[2]
    Y = args[3]
    X = np.real(args[4])

    M = M_flat.reshape(n, m)

    '''
    dJ=np.matrix(np.zeros((n,m)))
    for t in range(t_t):
        dJ += M*X[:,t]*np.transpose(X[:,t]) - Y[:,t]*np.transpose(X[:,t])

    dJ *= 2
    '''

    dJ = 2 * (M * X * X.transpose() - Y * X.transpose())

    return np.squeeze(np.array(dJ.reshape(-1, 1)))
