"""Implements class originating from inputlds.py"""
import numpy as np

class DynamicalSystem(object):
    """
    Class originated from inputlds.py, which
    was the first version of the algorithm. 
    Used as a model instance for our filters.
    """
    def __init__(self,matrix_a,matrix_b,matrix_c,matrix_d, **kwargs):
        """
        Inits DynamicalSystem with four matrix args and
        adds possibility of additional keywords in arguments.

        G - matrix_a
        matrix_b - np.zeros((2,1))
        F_dash - matrix_c
        matrix_d - np.zeros((1,1))

        If a matrix_a is a number, transforms it into float 
        and makes d-state vector equal to 1.
        If a matrix_a is square y x y, set d equal to y.
        If a matrix_b is a number, transform it into float
        and set n-input vector equal to 1. 
        matrix_b can't take place in case of single numbered
        matrix_a.
        If matrix_b is a matrix, number of its columns is assigned to n.
        If matrix_c is a number, transform it into float
        and set m-observation vector equal to 1.
        matrix_c can be a number only if matrix_a is a number too.
        If matrix_c is a matrix, number of its rows is assigned to m.
        matrix_d can't be not zero number if matrix_b is a matrix.
        Number of columns of matrix_d must be equal to n.

        Args:
            matrix_a: the evolution, system, transfer or state matrix (G matrix). Shape n x n
            matrix_b: 
            matrix_c:
            matrix_d: 

        Raises:
            KeyError: in case of no additional keywords.
            Exits in case of wrong format of a matrix.
            Exits in case of not square matrix_a.
            Exits in case of having any matrix_b, but
            matrix_a is a number.
            Exits if number of rows of matrix_b isn't
            equal to d.
            Exits if matrix_c is a number, but matrix_a
            is not.
            Exits if number of columns of matrix_c is not
            equalt to d.
            Exits if matrix_b is a matrix, but matrix_d is
            not zero number.
            Exits if number of columns of matrix_d is not
            equal to n-input vector.
        """
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.matrix_c = matrix_c
        self.matrix_d = matrix_d

        try:
            self.process_noise = kwargs['process_noise']
            try:
                self.proc_std = kwargs['process_noise_std']
            except KeyError:
                print('''Error: set 'process_noise_std'.''')
                exit()
        except KeyError:
            self.process_noise = None

        try:
            self.observation_noise = kwargs['observation_noise']
            try:
                self.obs_std = kwargs['observation_noise_std']
            except KeyError:
                print('''Error: set 'observation_noise_std'.''')
                exit()

        except KeyError:
            self.observation_noise = None

        # We expect to get a function that for a time-step t_t produces a multiplier
        # to be applied to b (possibly all elements of b, element-wise)
        try:
            self.timevarying_multiplier_b = kwargs['timevarying_multiplier_b']
        except KeyError:
            self.timevarying_multiplier_b = None


            # We expect to get a function that for a time-step t_t produces a multiplier
        # to be applied to b (possibly all elements of b, element-wise)
        try:
            self.corrupt_probability = kwargs['corrupt_probability']
        except KeyError:
            self.corrupt_probability = None


            #Checking dimensions of matrix_a and setting dimension, d, of state vector
        r = self.check_input(self.matrix_a)
        if r != 400:
            if r == 1:
                self.matrix_a=float(self.matrix_a)
                self.d=1
            else:
                self.matrix_a = np.matrix(self.matrix_a)
                if self.matrix_a.shape[0] != self.matrix_a.shape[1]:
                    print("Invalid state transition operator, matrix_a")
                    exit()
                self.d=self.matrix_a.shape[0]
        else:
            print("Invalid state transition operator, matrix_a")
            exit()

        #Checking dimensions of matrix_b and setting dimension, n, of input vector
        r = self.check_input(self.matrix_b)
        if r != 400:
            if r == 1:
                self.matrix_b=float(self.matrix_b)
                self.n=1
                if self.d != 1 and self.matrix_b !=0:
                    print("Invalid operator, matrix_b")
                    exit()
            else:
                self.matrix_b = np.matrix(self.matrix_b)
                if self.matrix_b.shape[0] != self.d:
                    print("Invalid operator, matrix_b")
                    exit()
                self.n=self.matrix_b.shape[1]
        else:
            print("Invalid operator, matrix_b")
            exit()

        #Checking dimensions of matrix_c and setting dimension, m, of observation vector
        r = self.check_input(self.matrix_c)
        if r != 400:
            if r == 1:
                self.matrix_c=float(self.matrix_c)
                self.m=1
                if self.d != 1:
                    print("Invalid operator, matrix_c")
                    exit()
            else:
                self.matrix_c = np.matrix(self.matrix_c)
                if self.matrix_c.shape[1] != self.d:
                    print("Invalid operator, matrix_c")
                    exit()
                self.m=self.matrix_c.shape[0]
        else:
            print("Invalid operator, matrix_c")
            exit()

        #Checking dimensions of matrix_d
        r = self.check_input(self.matrix_d)
        if r != 400:
            if r == 1:
                self.matrix_d=float(self.matrix_d)
                if self.n != 1 and self.matrix_d != 0:
                    print("Invalid operator, matrix_d")
                    exit()
            else:
                self.matrix_d = np.matrix(self.matrix_d)
                if self.matrix_d.shape[1] != self.n:
                    print("Invalid operator, matrix_d")
                    exit()
        else:
            print("Invalid operator, matrix_d")
            exit()

    def check_input(self, operator):
        """
        Checks variable type of matrices A,B,C,D

        Args:
            operator:a number or a matrix

        Returns:
            1

        Raises:
            TypeError: an error occured if the argument is none of
            possible formats
        """
        if isinstance(operator, int) or isinstance(operator, float):
            return 1
        else:
            try:
                np.matrix(operator)
            except TypeError:
                return 400

    def solve(self, h_zero, inputs, t_t, **kwargs):
        """
        Finds the outputs of the Dynamical System. We use
        them to find the prediction errors of our filters.

        t_t must be an integer greater than 1.
        Length of h_zero array must be equal to
        self.d(number of arrays in matrix A) if matrix_a
        is matrix
        If self.n-input vector is 1(matrix_b is a number),
        self.inputs will be transformed to a columns with t_t
        size.
        If matrix_b is matrix, inputs must have n x t_t size.
        If self.process_noise has Gaussian distribution, we
        create it with size d x t_t. If it isn't of Gaussian,
        we create matrix of zeros.
        If self.observation_noise has Gaussian distribution, we
        create it with size m x t_t. If it isn't of Gaussian,
        we create matrix of zeros.
        If it's wasn't given in init, we put earlies_event_time
        to zero.

        Args:
            h_zero: 1x2 array
            inputs: array of zeros of t_t size
            t_t: integer
            kwargs: additional keywords

        Raises:
            Exits if t_t is 1 or a float.
            Exits if matrix_a is a number, but
            h_zero can't be transformed into float.
            Exits if length of h_zero isn't equal
            to d(if matrix_a is matrix).
            Exits if self.n==1, but inputs don't have
            a size of t_t. 
            Exits if matrix_b is a matrix, but inputs
            don't have n x t_t size.

        """
        if t_t == 1 or not isinstance(t_t,int):
            print("t_t must be an integer greater than 1")
            exit()

        if self.d==1:
            try:
                h_zero=float(h_zero)
            except:
                print("Something wrong with initial state.")
                exit()
        else:
            try:
                h_zero = np.matrix(h_zero, dtype=float).reshape(self.d,1)
            except:
                print("Something wrong with initial state.")
                exit()

        if self.n==1:
            try:
                self.inputs = list(np.squeeze(np.array(inputs, dtype=float).reshape(1,t_t)))
            except:
                print("Something wrong with inputs. Should be list of scalars of length %d." % (
                    t_t))
                exit()
        else:
            try:
                self.inputs = np.matrix(inputs, dtype=float)
            except:
                print("Something wrong with inputs.")
                exit()

            if self.inputs.shape[0] != self.n or self.inputs.shape[1] !=t_t:
                print("Something wrong with inputs: wrong dimension or wrong number of inputs.")
                exit()

        if str(self.process_noise).lower() == 'gaussian':
            process_noise = np.matrix(np.random.normal(loc=0,\
                 scale=self.proc_std, size=(self.d,t_t)))
        else:
            process_noise = np.matrix(np.zeros((self.d,t_t)))

        if str(self.observation_noise).lower() == 'gaussian':
            observation_noise = np.matrix(np.random.normal(loc=0,\
                 scale=self.proc_std, size=(self.m,t_t)))
        else:
            observation_noise = np.matrix(np.zeros((self.m,t_t)))

        try:
            earliest_event_time = kwargs['earliest_event_time']
        except KeyError:
            earliest_event_time = 0

        self.h_zero=h_zero
        self.outputs = []
        self.event_or_not = []
        for t in range(t_t):

            if self.n==1:
                h_zero = self.matrix_a*h_zero + self.matrix_b*self.inputs[t] + process_noise[:,t]
                y  = self.matrix_c*h_zero + self.matrix_d*self.inputs[t] + observation_noise[:,t]
                if self.timevarying_multiplier_b is not None:
                    self.matrix_b *= self.timevarying_multiplier_b(t)
            else:
                h_zero = self.matrix_a*h_zero + self.matrix_b*self.inputs[:,t] + process_noise[:,t]
                y  = self.matrix_c*h_zero + self.matrix_d*self.inputs[:,t] + observation_noise[:,t]
                if self.timevarying_multiplier_b is not None:
                    self.matrix_b = self.matrix_b.dot(self.timevarying_multiplier_b(t))

            if ((self.corrupt_probability is not None) and
                    np.random.random_sample() <= self.corrupt_probability and
                        t>earliest_event_time):
                self.event_or_not.append(True)
                y[:,0] = 100.0 * np.random.random_sample()
                self.outputs.append(y)
            else:
                self.event_or_not.append(False)
                self.outputs.append(y)
        #print(self.outputs)

#print(DynamicalSystem.check_input.__doc__)
