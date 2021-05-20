def _em(observations, transition_offsets, observation_offsets,
        smoothed_state_means, smoothed_state_covariances, pairwise_covariances,
        given={}):
    """Apply the EM Algorithm to the Linear-Gaussian model
    Estimate Linear-Gaussian model parameters by maximizing the expected log
    likelihood of all observations.
    Parameters
    ----------
    observations : [n_timesteps, n_dim_obs] array
        observations for times [0...n_timesteps-1].  If observations is a
        masked array and any of observations[t] is masked, then it will be
        treated as a missing observation.
    transition_offsets : [n_dim_state] or [n_timesteps-1, n_dim_state] array
        transition offset
    observation_offsets : [n_dim_obs] or [n_timesteps, n_dim_obs] array
        observation offsets
    smoothed_state_means : [n_timesteps, n_dim_state] array
        smoothed_state_means[t] = mean of state at time t given all
        observations
    smoothed_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
        smoothed_state_covariances[t] = covariance of state at time t given all
        observations
    pairwise_covariances : [n_timesteps, n_dim_state, n_dim_state] array
        pairwise_covariances[t] = covariance between states at times t and
        t-1 given all observations.  pairwise_covariances[0] is ignored.
    given: dict
        if one of the variables EM is capable of predicting is in given, then
        that value will be used and EM will not attempt to estimate it.  e.g.,
        if 'observation_matrix' is in given, observation_matrix will not be
        estimated and given['observation_matrix'] will be returned in its
        place.
    Returns
    -------
    transition_matrix : [n_dim_state, n_dim_state] array
        estimated transition matrix
    observation_matrix : [n_dim_obs, n_dim_state] array
        estimated observation matrix
    transition_offsets : [n_dim_state] array
        estimated transition offset
    observation_offsets : [n_dim_obs] array
        estimated observation offset
    transition_covariance : [n_dim_state, n_dim_state] array
        estimated covariance matrix for state transitions
    observation_covariance : [n_dim_obs, n_dim_obs] array
        estimated covariance matrix for observations
    initial_state_mean : [n_dim_state] array
        estimated mean of initial state distribution
    initial_state_covariance : [n_dim_state] array
        estimated covariance of initial state distribution
    """
    if 'observation_matrices' in given:
        observation_matrix = given['observation_matrices']
    else:
        observation_matrix = _em_observation_matrix(
            observations, observation_offsets,
            smoothed_state_means, smoothed_state_covariances
        )

    if 'observation_covariance' in given:
        observation_covariance = given['observation_covariance']
    else:
        observation_covariance = _em_observation_covariance(
            observations, observation_offsets,
            observation_matrix, smoothed_state_means,
            smoothed_state_covariances
        )

    if 'transition_matrices' in given:
        transition_matrix = given['transition_matrices']
    else:
        transition_matrix = _em_transition_matrix(
            transition_offsets, smoothed_state_means,
            smoothed_state_covariances, pairwise_covariances
        )

    if 'transition_covariance' in given:
        transition_covariance = given['transition_covariance']
    else:
        transition_covariance = _em_transition_covariance(
            transition_matrix, transition_offsets,
            smoothed_state_means, smoothed_state_covariances,
            pairwise_covariances
        )

    if 'initial_state_mean' in given:
        initial_state_mean = given['initial_state_mean']
    else:
        initial_state_mean = _em_initial_state_mean(smoothed_state_means)

    if 'initial_state_covariance' in given:
        initial_state_covariance = given['initial_state_covariance']
    else:
        initial_state_covariance = _em_initial_state_covariance(
            initial_state_mean, smoothed_state_means,
            smoothed_state_covariances
        )

    if 'transition_offsets' in given:
        transition_offset = given['transition_offsets']
    else:
        transition_offset = _em_transition_offset(
            transition_matrix,
            smoothed_state_means
        )

    if 'observation_offsets' in given:
        observation_offset = given['observation_offsets']
    else:
        observation_offset = _em_observation_offset(
            observation_matrix, smoothed_state_means,
            observations
        )

    return (transition_matrix, observation_matrix, transition_offset,
            observation_offset, transition_covariance,
            observation_covariance, initial_state_mean,
            initial_state_covariance)


def _em_observation_matrix(observations, observation_offsets,
                          smoothed_state_means, smoothed_state_covariances):
    r"""Apply the EM algorithm to parameter `observation_matrix`
    Maximize expected log likelihood of observations with respect to the
    observation matrix `observation_matrix`.
    .. math::
        C &= ( \sum_{t=0}^{T-1} (z_t - d_t) \mathbb{E}[x_t]^T )
             ( \sum_{t=0}^{T-1} \mathbb{E}[x_t x_t^T] )^-1
    """
    _, n_dim_state = smoothed_state_means.shape
    n_timesteps, n_dim_obs = observations.shape
    res1 = np.zeros((n_dim_obs, n_dim_state))
    res2 = np.zeros((n_dim_state, n_dim_state))
    for t in range(n_timesteps):
        if not np.any(np.ma.getmask(observations[t])):
            observation_offset = _last_dims(observation_offsets, t, ndims=1)
            res1 += np.outer(observations[t] - observation_offset,
                             smoothed_state_means[t])
            res2 += (
                smoothed_state_covariances[t]
                + np.outer(smoothed_state_means[t], smoothed_state_means[t])
            )
    return np.dot(res1, linalg.pinv(res2))


def _em_observation_covariance(observations, observation_offsets,
                              transition_matrices, smoothed_state_means,
                              smoothed_state_covariances):
    r"""Apply the EM algorithm to parameter `observation_covariance`
    Maximize expected log likelihood of observations with respect to the
    observation covariance matrix `observation_covariance`.
    .. math::
        R &= \frac{1}{T} \sum_{t=0}^{T-1}
                [z_t - C_t \mathbb{E}[x_t] - b_t]
                    [z_t - C_t \mathbb{E}[x_t] - b_t]^T
                + C_t Var(x_t) C_t^T
    """
    _, n_dim_state = smoothed_state_means.shape
    n_timesteps, n_dim_obs = observations.shape
    res = np.zeros((n_dim_obs, n_dim_obs))
    n_obs = 0
    for t in range(n_timesteps):
        if not np.any(np.ma.getmask(observations[t])):
            transition_matrix = _last_dims(transition_matrices, t)
            transition_offset = _last_dims(observation_offsets, t, ndims=1)
            err = (
                observations[t]
                - np.dot(transition_matrix, smoothed_state_means[t])
                - transition_offset
            )
            res += (
                np.outer(err, err)
                + np.dot(transition_matrix,
                         np.dot(smoothed_state_covariances[t],
                                transition_matrix.T))
            )
            n_obs += 1
    if n_obs > 0:
        return (1.0 / n_obs) * res
    else:
        return res


def _em_transition_matrix(transition_offsets, smoothed_state_means,
                          smoothed_state_covariances, pairwise_covariances):
    r"""Apply the EM algorithm to parameter `transition_matrix`
    Maximize expected log likelihood of observations with respect to the state
    transition matrix `transition_matrix`.
    .. math::
        A &= ( \sum_{t=1}^{T-1} \mathbb{E}[x_t x_{t-1}^{T}]
                - b_{t-1} \mathbb{E}[x_{t-1}]^T )
             ( \sum_{t=1}^{T-1} \mathbb{E}[x_{t-1} x_{t-1}^T] )^{-1}
    """
    n_timesteps, n_dim_state, _ = smoothed_state_covariances.shape
    res1 = np.zeros((n_dim_state, n_dim_state))
    res2 = np.zeros((n_dim_state, n_dim_state))
    for t in range(1, n_timesteps):
        transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
        res1 += (
            pairwise_covariances[t]
            + np.outer(smoothed_state_means[t],
                       smoothed_state_means[t - 1])
            - np.outer(transition_offset, smoothed_state_means[t - 1])
        )
        res2 += (
            smoothed_state_covariances[t - 1]
            + np.outer(smoothed_state_means[t - 1],
                       smoothed_state_means[t - 1])
        )
    return np.dot(res1, linalg.pinv(res2))


def _em_transition_covariance(transition_matrices, transition_offsets,
                              smoothed_state_means, smoothed_state_covariances,
                              pairwise_covariances):
    r"""Apply the EM algorithm to parameter `transition_covariance`
    Maximize expected log likelihood of observations with respect to the
    transition covariance matrix `transition_covariance`.
    .. math::
        Q &= \frac{1}{T-1} \sum_{t=0}^{T-2}
                (\mathbb{E}[x_{t+1}] - A_t \mathbb{E}[x_t] - b_t)
                    (\mathbb{E}[x_{t+1}] - A_t \mathbb{E}[x_t] - b_t)^T
                + A_t Var(x_t) A_t^T + Var(x_{t+1})
                - Cov(x_{t+1}, x_t) A_t^T - A_t Cov(x_t, x_{t+1})
    """
    n_timesteps, n_dim_state, _ = smoothed_state_covariances.shape
    res = np.zeros((n_dim_state, n_dim_state))
    for t in range(n_timesteps - 1):
        transition_matrix = _last_dims(transition_matrices, t)
        transition_offset = _last_dims(transition_offsets, t, ndims=1)
        err = (
            smoothed_state_means[t + 1]
            - np.dot(transition_matrix, smoothed_state_means[t])
            - transition_offset
        )
        Vt1t_A = (
            np.dot(pairwise_covariances[t + 1],
                   transition_matrix.T)
        )
        res += (
            np.outer(err, err)
            + np.dot(transition_matrix,
                     np.dot(smoothed_state_covariances[t],
                            transition_matrix.T))
            + smoothed_state_covariances[t + 1]
            - Vt1t_A - Vt1t_A.T
        )

    return (1.0 / (n_timesteps - 1)) * res


def _em_initial_state_mean(smoothed_state_means):
    r"""Apply the EM algorithm to parameter `initial_state_mean`
    Maximize expected log likelihood of observations with respect to the
    initial state distribution mean `initial_state_mean`.
    .. math::
        \mu_0 = \mathbb{E}[x_0]
    """

    return smoothed_state_means[0]


def _em_initial_state_covariance(initial_state_mean, smoothed_state_means,
                                 smoothed_state_covariances):
    r"""Apply the EM algorithm to parameter `initial_state_covariance`
    Maximize expected log likelihood of observations with respect to the
    covariance of the initial state distribution `initial_state_covariance`.
    .. math::
        \Sigma_0 = \mathbb{E}[x_0, x_0^T] - mu_0 \mathbb{E}[x_0]^T
                   - \mathbb{E}[x_0] mu_0^T + mu_0 mu_0^T
    """
    x0 = smoothed_state_means[0]
    x0_x0 = smoothed_state_covariances[0] + np.outer(x0, x0)
    return (
        x0_x0
        - np.outer(initial_state_mean, x0)
        - np.outer(x0, initial_state_mean)
        + np.outer(initial_state_mean, initial_state_mean)
    )


def _em_transition_offset(transition_matrices, smoothed_state_means):
    r"""Apply the EM algorithm to parameter `transition_offset`
    Maximize expected log likelihood of observations with respect to the
    state transition offset `transition_offset`.
    .. math::
        b = \frac{1}{T-1} \sum_{t=1}^{T-1}
                \mathbb{E}[x_t] - A_{t-1} \mathbb{E}[x_{t-1}]
    """
    n_timesteps, n_dim_state = smoothed_state_means.shape
    transition_offset = np.zeros(n_dim_state)
    for t in range(1, n_timesteps):
        transition_matrix = _last_dims(transition_matrices, t - 1)
        transition_offset += (
            smoothed_state_means[t]
            - np.dot(transition_matrix, smoothed_state_means[t - 1])
        )
    if n_timesteps > 1:
        return (1.0 / (n_timesteps - 1)) * transition_offset
    else:
        return np.zeros(n_dim_state)


def _em_observation_offset(observation_matrices, smoothed_state_means,
                           observations):
    r"""Apply the EM algorithm to parameter `observation_offset`
    Maximize expected log likelihood of observations with respect to the
    observation offset `observation_offset`.
    .. math::
        d = \frac{1}{T} \sum_{t=0}^{T-1} z_t - C_{t} \mathbb{E}[x_{t}]
    """
    n_timesteps, n_dim_obs = observations.shape
    observation_offset = np.zeros(n_dim_obs)
    n_obs = 0
    for t in range(n_timesteps):
        if not np.any(np.ma.getmask(observations[t])):
            observation_matrix = _last_dims(observation_matrices, t)
            observation_offset += (
                observations[t]
                - np.dot(observation_matrix, smoothed_state_means[t])
            )
            n_obs += 1
    if n_obs > 0:
        return (1.0 / n_obs) * observation_offset
    else:
        return observation_offset
