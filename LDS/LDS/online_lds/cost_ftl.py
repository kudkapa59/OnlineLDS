import numpy as np

def cost_ftl(M_flat, *args):
    """
    from onlinelds.py
    """
    n = args[0]
    m = args[1]
    t_t = args[2]
    Y = args[3]
    X = args[4]

    M = M_flat.reshape(n, m)
    '''
    J = 0
    for t in range(t_t):
        J += pow(np.linalg.norm(Y[:,t] - M*X[:,t]),2)
    '''

    J = np.real(np.trace(np.transpose(Y - M * X) * (Y - M * X)))

    return J
