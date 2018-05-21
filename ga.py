#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

class SimpleGeneticAlgorithm(object):
    """ Basic Genetic Algorithm for single-objective problems
    Converted from the matlab code below:
    https://www.mathworks.com/matlabcentral/fileexchange/39021-basic-genetic-algorithm?focused=9262568&tab=function

    The version of this Genetic Algorithm is decribed in
    "Control predictivo basado en modelos mediante técnica de optimización heurística.
    Aplicación a procesos no lineales y multivariables"
    F. Xavier Blasco Ferragud, PhD Tesis 1999 (in Spanish)
    Editorial UPV. ISBN 84-699-5429-6.

    Attributes
    ----------
    bounds : numpy.ndarray
        [lb, ub]  lower (lb) and upper (up) bounds of the search space.
        each dimension of the search space requires bounds
    Objfun : function
        the name of the 0bjective function to be minimize
    max_gen : int
        Number of generation, = n_var*20+10 by default
    n_pop : int
        Size of the population, = n_var*50 by default
    alpha : int
        Parameter for linear crossover, 0 by default
    p_xovr : float
        Crossover probability, 0.9 by default
    p_mut : float
        Mutation probability, 0.1 by default
    ObjfunPar : dict
        Additional parameters of the objective function
        have to be packed in a dict, empty by default
    chrom_init : numpy.ndarray
        Initialized members of the initial population, empty by default
    xmin : numpy.ndarray
        the optimal solution
    xmingen : numpy.ndarray
        the optimal solutions at each generation
    fxmin : float
        the final minimum of the objective function
    fxmingen : numpy.ndarray
        the minimum objectives at each generation
    """
    def __init__(self, Objfun, bounds, max_gen=None, n_pop=None,
            alpha=0, p_xovr=0.9, p_mut=0.1,
            ObjfunPar=None, chrom_init=None, verbose=1):

        self.Objfun = Objfun
        self.bounds = bounds
        self.n_var = len(bounds)

        if max_gen is None:
            self.max_gen = self.n_var*20 + 10
        else:
            self.max_gen = max_gen

        if n_pop is None:
            self.n_pop = self.n_var*50
        else:
            self.n_pop = n_pop

        self.alpha = alpha
        self.p_xovr = p_xovr
        self.p_mut = p_mut

        if ObjfunPar is None:
            self.ObjfunPar = {}
        else:
            self.ObjfunPar = ObjfunPar

        if chrom_init is None:
            self.chrom_init = []
        else:
            self.chrom_init = chrom_init

        self.verbose = verbose

        self.chrom = []
        self.ObjV = []
        self.xmin = []
        self.fxmin = sys.float_info.max
        self.xmingen = np.zeros((self.max_gen, self.n_var))
        self.fxmingen = np.zeros(self.max_gen)
        self.rf = np.arange(self.n_pop)
        self.gen = 0

    def minimize(self):
        """ Main loop
        """
        # Generation counter
        gen = 0

        # Initial population
        self.chrom = self.crtrp(self.n_pop, self.bounds)   # Real codification
        # Individuals of chrom_init are randomly added in the initial population
        nind0 = len(self.chrom_init)
        if nind0 > 0:
            posicion0 = np.ceil(np.random.rand(nind0)*self.n_pop).astype(int)
            self.chrom[posicion0, :] = self.chrom_init

        while (gen < self.max_gen):
            self.gen = gen
            self.gaevolucion()
            # Increase generation counter
            self.xmingen[gen, :] = self.xmin
            self.fxmingen[gen] = self.fxmin
            gen += 1

        # Present final results
        self.garesults()


    def crtrp(self, n_pop, bounds):
        """ A random real value matrix is created coerced by upper and
        lower bounds
        """
        n_var = len(bounds)
        aux = np.random.rand(n_pop, n_var)
        ub_sub_lb = np.repeat((bounds[:, 1]-bounds[:, 0])[np.newaxis, :], n_pop, axis=0)
        lb = np.repeat((bounds[:, 0])[np.newaxis, :], n_pop, axis=0)
        chrom = ub_sub_lb*aux + lb

        return chrom

    def gaevolucion(self):
        """ One generation
        """
        chrom = self.chrom
        nind = chrom.shape[0]
        if len(self.ObjfunPar) == 0:
            ObjV = self.Objfun(chrom)
        else:
            ObjV = self.Objfun(chrom, **self.ObjfunPar)
        self.ObjV = ObjV

        # Best individual of the generation
        p = self.ObjV.argmin()
        v = self.ObjV[p]
        if v <= self.fxmin:
            self.xmin = chrom[p, :]
            self.fxmin = v

        # Next generation
        # RANKING
        fit_vec = self.ranking(self.ObjV, self.rf)
        # SELECTION
        # Stochastic Universal Sampling (SUS).
        chrom_sel = self.select(chrom, fit_vec, 1)
        # CROSSOVER
        # Uniform crossover.
        chrom_sel = self.lxov(chrom_sel, self.p_xovr, self.alpha)
        # MUTATION
        chrom = self.mutbga(chrom_sel, self.bounds, [self.p_mut, 1]) # Codificacin Real.
        # Reinsert the best individual
        chrom[int(np.round(self.n_pop/2)), :] = self.xmin
        self.chrom = chrom

        # Optional additional task required by user
        if self.verbose == 1:
            self.gaiteration()

    def ranking(self, ObjV, RFun):
        """ Ranking function

        Parameters
        ----------
        ObjV : numpy.ndarray
        RFun : numpy.ndarray
            np.array([0, 1, ... , n_pop])

        Returns
        -------
        fit_vec : numpy.ndarray
            fitness vector
        """
        if not (len(ObjV) == len(RFun)):
            raise Exception('RFun have to be of the same size than ObjV.');

        pos = np.argsort(ObjV)
        fit_vec = np.zeros(len(RFun))
        fit_vec[pos] = RFun[-1::-1]

        return fit_vec

    def select(self, chrom, fit_vec, GGAP=1.0):
        """ Selection Function

        Parameters
        ----------
        chrom : numpy.ndarray
        fit_vec : numpy.ndarray
            fitness vector
        GGAP : float

        Returns
        -------
        chrom_sel : numpy.ndarray
            selected chromosomes

        """
        # Indexes of new individuals
        indices = self.sus2(fit_vec, int(np.round(len(fit_vec)*GGAP)))

        if GGAP < 1: # there is overlap
            # Members of the population to overlap
            oldpos = np.arange(len(fit_vec))
            for k in range(len(fit_vec)):
                pos = int(np.round(np.random.rand()*len(fit_vec)+0.5))
                # exchange indexes
                oldpos[pos], oldpos[k] = oldpos[k], oldpos[pos]
            oldpos = oldpos[0:int(np.round(len(fit_vec)*GGAP))]
            chrom_sel = chrom
            chrom_sel[oldpos, :] = chrom[indices, :]
        else: # more childs than parents
            chrom_sel = chrom[indices, :]

        # Disorder the population.
        indi = np.argsort(np.random.rand(len(fit_vec)))
        chrom_sel = chrom_sel[indi, :]

        return chrom_sel

    def lxov(self, chrom_old, p_xovr=0.7, alpha=0, bounds=None):
        """ Linear crossover
        Produce a new population by linear crossover and p_xovr crossover probability

        Linear recombination.
        Parameters 'beta1' and 'beta2' are randomly obtained inside [-0.5-alpha, 0.5+alpha]
        interval
          Child1 = beta1*Parent1+(1-beta1)*Parent2
          Child2 = beta2*Parent1+(1-beta2)*Parent2

        Parameters
        ----------
        chrom_old : numpy.ndarray
        p_xovr : float
        alpha : float

        Returns
        -------
        chrom_new : numpy.ndarray

        """
        n_pop = chrom_old.shape[0]   # Number of individuals and chromosome length
        n_pairs = int(np.floor(n_pop/2))    # Number of pairs
        cruzar = np.random.rand(n_pairs)<= p_xovr    # Pairs to crossover
        chrom_new = chrom_old.copy()
        for i in range(n_pairs):
            pin = i*2
            if cruzar[i]:
                betas = np.random.rand(2)*(1+2*alpha)-(0.5+alpha)
                A = np.array([[betas[0], 1-betas[0]], [1-betas[1], betas[1]]])
                chrom_new[pin:pin+2, :] = np.dot(A, chrom_old[pin:pin+2, :])

        # Coerce points outside search space
        if bounds is not None:
            auxf1 = np.repeat((bounds[:, 0])[np.newaxis, :], n_pop, axis=0)
            auxf2 = np.repeat((bounds[:, 1])[np.newaxis, :], n_pop, axis=0)
            chrom_new[chrom_new > auxf2] = auxf2[chrom_new > auxf2]
            chrom_new[chrom_new < auxf1] = auxf1[chrom_new < auxf1]

        return chrom_new

    def mutbga(self, chrom_old, bounds, mut_opt=None):
        """ Mutation function
        Real coded mutation.
        Mutation is produced by adding a small random value

        Parameters
        ----------
        chrom_old : numpy.ndarray
            Initial population.
        bounds : numpy.ndarray
            Upper and lower bounds
        mut_opt: list
            mutation options,
            mut_opt[0] = mutation probability (0 to 1).
            mut_opt[1] = compression of the mutation value (0 to 1).
            default mut_opt[0] = 1/n_var, mut_opt[1] = 1
        """

        if mut_opt is not None:
            p_mut = mut_opt[0]
            shr = mut_opt[1]
        else:
            p_mut = 1/len(bounds)
            shr = 1.0

        n_pop = chrom_old.shape[0]
        m1 = 0.5-(1-p_mut)*0.5
        m2 = 0.5+(1-p_mut)*0.5

        aux = np.random.rand(chrom_old.shape[0], chrom_old.shape[1])
        mut_mx = np.logical_xor(aux > m2, aux < m1)
        ub_sub_lb = np.repeat(((bounds[:, 1]-bounds[:, 0])*0.5*shr)[np.newaxis, :], n_pop, axis=0)
        index = np.vstack((np.where(mut_mx)[0], np.where(mut_mx)[1])).T
        m = 20
        alpha = np.random.rand(m, len(index)) < (1.0/m)
        xx = 2**np.arange(0, -m, -1, dtype=float)
        aux2 = np.dot(xx[np.newaxis, :], alpha)
        delta = np.zeros(mut_mx.shape)
        delta[index[:, 0], index[:, 1]] = aux2
        chrom_new = chrom_old + (mut_mx*ub_sub_lb*delta)

        # Coerce points outside bounds
        auxf1 = np.repeat((bounds[:, 0])[np.newaxis, :], n_pop, axis=0)
        auxf2 = np.repeat((bounds[:, 1])[np.newaxis, :], n_pop, axis=0)
        chrom_new[chrom_new > auxf2] = auxf2[chrom_new > auxf2]
        chrom_new[chrom_new < auxf1] = auxf1[chrom_new < auxf1]

        return chrom_new

    def sus2(self, fit_vec, n_sel):
        """Stochastic Universal Sampling (SUS)

        Returns
        -------
        idx_chrom_new : numpy.ndarray
            the indices of selected chromosomes
        """
        suma = np.sum(fit_vec)
        # Position of the roulette pointers
        j = 0
        sumfit = 0
        paso = suma/n_sel # distance between pointers
        flecha = np.random.rand()*paso # offset of the first pointer
        idx_chrom_new = np.zeros(n_sel, dtype=int)
        for i in range(n_sel):
            sumfit = sumfit + fit_vec[i]
            while (sumfit >= flecha):
                idx_chrom_new[j] = i
                flecha = flecha + paso
                j += 1
        return idx_chrom_new

    def gaiteration(self):
        """ Optional user task executed at the end of each iteration
        """

        # For instance, results of the iteration
        print('------------------------------------------------')
        print('Iteration: ', self.gen)
        print('xmin: ', self.xmin)
        print('f(xmin): ', self.fxmin)

    def garesults(self):
        """ Optional user task executed when the algorithm ends
        """
        # For instance, final result
        print('------------------------------------------------')
        print('######   RESULT   #########')
        print('Objective function for xmin: ', self.fxmin)
        print('xmin: ', self.xmin)
        print('------------------------------------------------')
