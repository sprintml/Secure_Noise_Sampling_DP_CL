from decimal import *
getcontext().prec = 224 # increase precision

import math
import numpy as np
from scipy.stats import norm
from datetime import datetime
import pickle
import random
import json

import matplotlib.pyplot as plt

# see __main__ at the bottom of this script for example usage

# ----------------------------------------------------------------
# implementation of dice ensemble approximation
# ----------------------------------------------------------------

# note: we use the python decimal package with 224 places of precision (default 28) to improve the numerical stability
# of this implementation compared to a floating point number implementation. However, at higher numbers of dice and for 
# distributions with pmf values requiring very high precision representations, this implementation may still run into
# into numerical instability due to fixed precision arithmetic. 
# For a real-world implementation in sensitive applications, look up table values
# should be computed using arbitrary precision arithmetic. Performance on runtime benchmarks
# for our noise sampling protocol (performed in another section of this repo) is unaffected
# by this distinction, since our protocol has the same runtime regardless of the values in the table.

class DiceEnsemble:
    def __init__(self, T, t_pmf):
        self.T = T # table size
        self.target_pmf = t_pmf # dictionary representing the probability mass fn of the target dist
        self.n = len(t_pmf.keys()) # size of support of target dist
        self.target_support = [k for k in t_pmf.keys()] # support of target dist
        self.de_pmf = {} # dictionary representing the probability mass fn encoded by sampling from the dice ensemble
        self.ratios = {} # initialized by compute_ratios(). ratios f'(x) / f(x) where f' is the de-approximated pmf and f is the target pmf
        self.body = None # initialized by fit_pmf_decimal() and fit_pmf_float(). tables which encode the dice ensemble, dictionary with bookkeeping conventions described below
        self.em_mass_ls = None # initialized by em_masses(). probability mass of elements in each layer of the dice ensemble
        
    # body should be a dictionary - keys are names of dice, values are the dice.
        # dice are lists of size <num faces>
    # possible dice entries:
    # ("bot", None), indicates an unfilled face
    # ("placeholder", <the index of where the placeholder will be filled from>)
    # ("element", <the element>)

    # the following methods do bookkeeping to maintain the convention described above

    def set_blocks_to_element(self, num_blocks, index, element):
        inds = indices_holding_x(self.body[index], ("bot", None))
        if num_blocks > len(inds):
            print("set_blocks_to_element(num_blocks=", num_blocks, ", index=", index, ", element=", element, ") tried to allocate too many blocks.")
            print(num_blocks, " were requested for allocation but layer ", index, " has only ", len(inds), " blocks.")
            return -1
        for i in range(num_blocks):
            self.body[index][inds[i]] = ("element", element) # set the ith index holding \bot to hold the element instead

    def set_blocks_to_link(self, num_blocks, source_index, target_index):
        inds = indices_holding_x(self.body[source_index], ("bot", None))
        if num_blocks > len(inds):
            print("set_blocks_to_link(num_blocks=", num_blocks, ", source_index=", source_index, ", target_index=", target_index, ") tried to allocate too many blocks.")
            print(num_blocks, " were requested for allocation but layer ", source_index, " has only ", len(inds), " blocks.")
            return -1
        for i in range(num_blocks):
            self.body[source_index][inds[i]] = ("placeholder", target_index)
    
    def indices_holding_element(self, element, layer):
        return indices_holding_x(self.body[layer], ("element", element))
    
    def indices_holding_link(self, source_layer, target_layer):
        return indices_holding_x(self.body[source_layer], ("placeholder", target_layer))
    
    # Algorithm 4
    # l is the number of dice allowed in the ensemble
    def fit_pmf_decimal(self, l, verbose=True):
        if self.T < (2* self.n):
            print("error: this method only works if T >= 2n")
            return -1
        self.body = {}
        for i in range(1, l+1):
            self.body[i] = [("bot", None)] * int(self.T) # T entries of \bot to begin
        curr_dist = {} 
        # initialize curr_dist to f(x)
        for e in self.target_support:
            curr_dist[e] = self.target_pmf[e]
            self.de_pmf[e] = Decimal(0.0)
        curr_cond_mass = Decimal(1)
        for layer_counter in range(1, l+1):
            if verbose:
                print("layer_counter:", layer_counter)
            for e in self.target_support:
                p = curr_dist[e]
                e_blocks = (p * (self.T)) // Decimal(1)
                norm_e_mass = e_blocks / self.T
                self.set_blocks_to_element(int(e_blocks), layer_counter, e)
                curr_dist[e] = curr_dist[e] - norm_e_mass
                self.de_pmf[e] += norm_e_mass * curr_cond_mass
            if sum(curr_dist.values()) == 0:
                return
            curr_dist = normalize(curr_dist)
            inds = indices_holding_x(self.body[layer_counter], ("bot", None))
            curr_cond_mass = curr_cond_mass * len(inds) * (1/self.T) # amount of prob mass represented by next layer
            if layer_counter < l:
                self.set_blocks_to_link(len(inds), layer_counter, layer_counter+1)


    # l is the number of allowed dice in the ensemble
    # this version uses floating point numbers, runs into numerical instability quickly
    def fit_pmf_float(self, l):
        if self.T < (2* self.n):
            print("error: this method only works if T >= 2n")
            return -1

        self.body = {}
        for i in range(1, l+1):
            self.body[i] = [("bot", None)] * self.T # T entries of \bot to begin
        curr_dist = {} 
        # initialize curr_dist to f(x)
        for e in self.target_support:
            curr_dist[e] = self.target_pmf[e]
            self.de_pmf[e] = 0.0

        curr_cond_mass = 1
        for layer_counter in range(1, l+1):
            for e in self.target_support:
                p = curr_dist[e]
                e_blocks = math.floor(p * (self.T))
                print("l:", layer_counter, "e:", e, "thing:", p*self.T, " other thing:", e_blocks)
                norm_e_mass = e_blocks / self.T
                self.set_blocks_to_element(e_blocks, layer_counter, e)
                curr_dist[e] = curr_dist[e] - norm_e_mass
                self.de_pmf[e] += norm_e_mass * curr_cond_mass
            curr_dist = normalize(curr_dist)
            inds = indices_holding_x(self.body[layer_counter], ("bot", None))
            curr_cond_mass = curr_cond_mass * len(inds) * (1/self.T) # amount of prob mass represented by next layer
            if layer_counter < l:
                self.set_blocks_to_link(len(inds), layer_counter, layer_counter+1)

    def compute_ratios(self):
        for e in self.target_support:
            if self.de_pmf[e] == 0:
                self.ratios[e] = "undefined"
            else:
                self.ratios[e] = self.de_pmf[e] / self.target_pmf[e]
    
    # returns the max defined ratio, and the number of undefined ratios
    def max_ratio(self):
        u_keys = []
        r_vals = []
        for k in self.ratios.keys():
            if self.ratios[k] == "undefined":
                u_keys.append(k)
            else:
                r_vals.append(self.ratios[k])
        return max(r_vals), len(u_keys)
    
    # compute probability mass based on layers up to m
    def empirical_mass(self, m):
        em_mass = {}
        for e in self.target_support:
            em_mass[e] = Decimal(0)
        cond_mass = Decimal(1) # amount of mass held by the current layer in total
        for layer in range(1, m+1):
            for e in self.target_support:
                elem_inds = self.indices_holding_element(e, layer)
                blocks_with_e = Decimal(len(elem_inds))
                element_mass = cond_mass * (blocks_with_e / Decimal(self.T)) # mass for element in layer
                em_mass[e] = em_mass[e] + element_mass
            link_inds = self.indices_holding_link(layer, layer+1)
            blocks_with_link = Decimal(len(link_inds))
            cond_mass = cond_mass * (blocks_with_link / Decimal(self.T))
        return em_mass

    # compute the empirical mass of each layer of the dice ensemble
    # returns a list of dictionaries, where d[e] <- f'(e) for all e \in supp(f)
    def em_masses(self):
        max_layer = len(self.body)
        cond_mass = Decimal(1) # amount of mass held by the current layer in total
        em_mass_ls = []
        prev_em_mass = {}
        for e in self.target_support:
            prev_em_mass[e] = Decimal(0)
        for layer in range(1, max_layer+1):
            em_mass_ls.append(prev_em_mass)
            new_em_mass = {}
            for k in prev_em_mass.keys():
                new_em_mass[k] = Decimal(prev_em_mass[k]) # copy each element
            for e in self.target_support:
                elem_inds = self.indices_holding_element(e, layer)
                blocks_with_e = Decimal(len(elem_inds))
                element_mass = cond_mass * (blocks_with_e / Decimal(self.T)) # mass for element in layer
                new_em_mass[e] = prev_em_mass[e] + element_mass
            link_inds = self.indices_holding_link(layer, layer+1)
            blocks_with_link = Decimal(len(link_inds))
            cond_mass = cond_mass * (blocks_with_link / Decimal(self.T))
            prev_em_mass = new_em_mass
        self.em_mass_ls = em_mass_ls
        return em_mass_ls

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

def load_de(filename):
    with open(filename, "rb") as f:
        ret = pickle.load(f)
    return ret


# given a list of empirical masses and a dice ensemble
# calculate the max ratio of all f(x) / f'(x)
def empirical_ratios(em_masses, de):
    ret = []
    for layer_em_mass in em_masses:
        curr_ratios = {}
        u_count = 0
        for e in de.target_support:
            if layer_em_mass[e] == 0:
                u_count += 1
            else:
                curr_ratios[e] =  de.target_pmf[e] / layer_em_mass[e]
        temp = "undefined"
        if len(curr_ratios) != 0:
            temp = max(curr_ratios.values())
        ret.append( (temp, u_count) )
    return ret

# given a list and an element x, gives all indices of the list that hold x
def indices_holding_x(ls, x):
    inds = []
    for i in range(len(ls)):
        if ls[i] == x:
            inds.append(i)
    return inds

# given a dictionary whose values are real numbers
# normalize it
def normalize(d):
    new_dict = {}
    s = sum(d.values())
    if s != 0:
        for k in d.keys():
            new_dict[k] = d[k]/s
    return new_dict


# -----------------------
# dictionary-based sampler
# -----------------------
# converts a dice ensemble to a python dictionary for ease of accuracy/utility analysis

# pmf should be a python dictionary representing a probability mass function
# pmf[x] = p
# where x is an element of the support, and p is its probability mass
# sample dict converts it to a cdf, and rounds missing mass (resulting from e.g. floating point rounding)
# to the max weight element.
# in effect, this redirects all \bot blocks to the max weight element
def pmf_to_sample_dict(pmf):
    sample_dict = {}
    sorted_elems = [k for k, v in sorted(pmf.items(), key=lambda item: item[1])]
    ctr = 0.0
    for k in sorted_elems:
        ctr += float(pmf[k])
        sample_dict[float(k)] = ctr
    sample_dict[sorted_elems[-1]] = 1.0
    return sample_dict

def save_as_json(sample_dict, filename):
    s = json.dumps(sample_dict)
    with open(filename, "w") as f:
        f.write(s)

def load_dict_from_json(filename):
    with open(filename, "r") as f:
        s = f.read()
    d = json.loads(s)
    return d

# return a sample from a sample dict
def sample_from_dict(sample_dict):
    u = random.uniform(0,1)
    sorted_elems = [k for k, v in sorted(sample_dict.items(), key=lambda item: item[1])]
    for k in sorted_elems:
        if sample_dict[k] >= u:
            return k
    print("problem with dict_sample")
    return None


# --------------------
# modeling target distributions (discrete gaussian & mesh gaussian)
# --------------------

# algorithm 2 from https://arxiv.org/pdf/2106.02848
def dec_discretize_and_truncate(cdf, mesh_size, L, verbose=True):
    h = mesh_size # this is just notation, they use h for mesh size in the paper
    n = math.floor((L- (h/2) ) / h)
    qs = {}
    for i in range(-n, n+1):
        qs[Decimal(i) * Decimal(h)] = cdf(i*h + h/2) - cdf(i*h - h/2)
        #if verbose:
        #    print(i, qs[i])

    if verbose:
        print("beginning dec conversion")
    dec_conversion = {}
    for k in qs.keys():
        dec_conversion[Decimal(k)] = Decimal(qs[k])
    if verbose:
        print("ended dec conversion")
    s = sum(dec_conversion.values())
    trunc_mass = Decimal(1) - s
    for k in dec_conversion.keys():
        dec_conversion[k] = dec_conversion[k] / s

    # in the paper they do this:
    # Y^L = Y conditioned on |Y| <= L
    # mu = E[Y^L] - \sum{i \in [-n, n]} i*h * q_i
    # but for guassian, mu = 0
    return dec_conversion, trunc_mass

def dec_mesh_gaussian(sigma, mesh_size, L):
    def cdf(x):
        return norm.cdf(x, loc=0, scale=sigma)
    return dec_discretize_and_truncate(cdf, mesh_size, L)

def mg_find_trunc_mass(sigma, mesh_size, target_mass):
    step = mesh_size * 1000
    UPPER_LIMIT = 5000
    for i in range(0,UPPER_LIMIT):
        trunc_mass = 2*(1-norm.cdf(i*step, scale=sigma))
        if trunc_mass < target_mass:
            return dec_mesh_gaussian(sigma, mesh_size, i*step)
    print("mg_find_trunc_mass() with params:", sigma, mesh_size, target_mass, " was unsuccessful.")
    return -1

# coarse search for L which will give less than the target amount of truncated mass
def mg_find_L_param(sigma, mesh_size, target_mass):
    step = mesh_size * 1000
    UPPER_LIMIT = 5000
    for i in range(0,UPPER_LIMIT):
        trunc_mass = 2*(1-norm.cdf(i*step, scale=sigma))
        if trunc_mass < target_mass:
            return i*step
    return -1


# discrete gaussian 
def dg_num(x, mu, sigma):
    num = - (x-mu)**2
    denom = 2 * sigma ** 2
    #print(num)
    #print(denom)
    return np.exp(-( ((x - mu)**2 ) / (2*(sigma**2)) )) 

# r is range parameter, number where we chop the tail in the simulation
def dg_approx_denom(mu, sigma, r):
    sum = 0
    for i in range(r, 0, -1): # sum for tails, not including 0
        sum += 2 * dg_num(i, mu, sigma)
    
    sum += dg_num(0, mu, sigma) # add in 0
    
    return sum

def dg_approx_pdf(x, mu=0, sigma=1, r=100):
    return dg_num(x, mu, sigma) / dg_approx_denom(mu, sigma, r)

def generate_fast_dg_approx_pdf(mu, sigma, r):
    C = dg_approx_denom(mu, sigma, r)
    def f(x):
        return dg_num(x, mu, sigma) / C
    return f

# set r > L
def generate_fast_dg_dict(mu, sigma, r, L):
    C = dg_approx_denom(mu, sigma, r)
    def f(x):
        return dg_num(x, mu, sigma) / C
    d = {}
    ts = 0
    for i in range(-r, r+1):
        if i >= -L and i <= L:
            d[i] = f(i)
        ts += f(i)
    s = sum(d.values())
    truncation_penalty = ts - s
    to_delete = []
    for k in d.keys():
        d[k] = d[k]/s # normalize after truncation
        if d[k] == 0:
            to_delete.append(k)

    for k in to_delete: # remove 0 probability elements
        del d[k]
    return d, truncation_penalty

def dec_dg(mu, sigma, r, L):
    qs, truncation_penalty_1 = generate_fast_dg_dict(mu, sigma, r, L)
    dec_conversion = {}
    for k in qs.keys():
        dec_conversion[Decimal(k)] = Decimal(qs[k])
    s = sum(dec_conversion.values())
    trunc_mass = Decimal(1) - s
    for k in dec_conversion.keys():
        dec_conversion[k] = dec_conversion[k] / s

    return dec_conversion, truncation_penalty_1, trunc_mass


# ----------------------
# Truncation of discrete distributions
# ----------------------

# remove all elements from a pmf with mass under tau
# count the total mass removed
# renormalize and return the updated pmf
def min_mass_filter(pmf_dict, tau):
    truncated_pmf_dict = {}

    # type check for Decimals, should work with Decimals or floats
    if type( next(iter(pmf_dict.values())) ) == type(Decimal(0)):
        mass_loss = Decimal(0)
        tau = Decimal(tau)
    else:
        mass_loss = 0

    for k in pmf_dict.keys():
        if pmf_dict[k] < tau:
            mass_loss += pmf_dict[k]
        else:
            truncated_pmf_dict[k] = pmf_dict[k]
    
    # convert to Decimal for accurate normalization
    dec_conversion = {}
    for k in truncated_pmf_dict.keys():
        dec_conversion[k] = Decimal(truncated_pmf_dict[k])

    # normalize
    s = sum(dec_conversion.values())
    for k in dec_conversion.keys():
        dec_conversion[k] = dec_conversion[k] / s
    
    return dec_conversion, mass_loss

# first generate dg pdf
# then filter out small mass elements
# then convert to Decimal
# then renormalize
def mass_filter_dg(mu, sigma, r, L, tau):
    pmf_dict, dg_approx_mass_loss = generate_fast_dg_dict(mu, sigma, r, L)
    renorm_pmf_dict, mass_filter_mass_loss = min_mass_filter(pmf_dict, tau)
    return renorm_pmf_dict, dg_approx_mass_loss, mass_filter_mass_loss


#given target truncation mass and sigma, return the truncation filter required to get that target mass
def find_trunc_filter_dg(target_mass, sigma, r=2**20, L=2**19):
    pmf_dict, dg_approx_mass_loss = generate_fast_dg_dict(0, sigma, r, L)
    ls = sorted([v for v in pmf_dict.values()])
    max_val = 0
    sum = 0
    for v in ls:
        if sum >= target_mass:
            return max_val
        sum += v
        max_val = v
    print("problem")
    return -1

def target_trunc_mass_dg(mu, sigma, r, L, target_mass):
    pmf_dict, dg_approx_mass_loss = generate_fast_dg_dict(mu, sigma, r, L)
    tau = find_trunc_filter_dg(target_mass, sigma, r, L)
    renorm_pmf_dict, mass_filter_mass_loss = min_mass_filter(pmf_dict, tau)
    return renorm_pmf_dict, mass_filter_mass_loss, tau


# ------------------------
# plots
# ------------------------

def plot_empirical_ratios(em_masses, de, title=None):
    ls = empirical_ratios(em_masses, de)
    xs = []
    ys = []
    # only plot if all ratios defined
    # (number of undefined ratios is second entry in tuples for empirical_ratios() )
    for i in range(len(ls)):
        if ls[i][1] == 0:
            xs.append(i)
            ys.append(ls[i][0])

    plt.plot(xs, ys)
    if title:
        plt.title(title)

    plt.xlabel("Number of Tables (size " + str(int(de.T)) + ")")
    plt.ylabel("max(approx_prob / target_prob)")
    plt.show()

def plot_table_size_vs_sd(sec_param = 64):
    sigmas = [i for i in range(1, 2101, 100)]
    ret = []
    i = 0
    for s in sigmas:
        print(i+1, "/", len(sigmas))
        pmf_dict, _, _ = target_trunc_mass_dg(0, s, 2**20, 2**19, 2**-sec_param)
        ret.append(len(pmf_dict) * 2)
        i += 1
    plt.plot(sigmas, ret)
    for i in range(6, 17):
        plt.axhline(2**i, linestyle="--")
    plt.yscale("log", base=2)
    plt.ylabel("Necessary Table Size")
    plt.xlabel("Standard Deviation")
    plt.title("Table Sizes for Discrete Gaussian Distributions")

    plt.show()



# --------------------------------
# constructing dice ensembles
# --------------------------------

def decimal_test_toy():
    toy_example = {}
    toy_example[1] = Decimal('0.51')
    toy_example[3] = Decimal('0.25')
    toy_example[5] = Decimal('0.24')
    a = DiceEnsemble(Decimal(10), toy_example)
    a.fit_pmf_decimal(3)
    print("actual pmf: ", toy_example)
    print("de pmf: ", a.de_pmf)
    print("body: ", a.body)
    return a


def decimal_test_mesh_gaussian_01():
    mg, trunc_mass = dec_mesh_gaussian(1, 4E-04, 4)
    print("number of elements in support:", len(mg))
    a = DiceEnsemble(Decimal(2*len(mg)), mg)
    a.fit_pmf_decimal(2)
    a.compute_ratios()
    print("max ratio for 2: ", a.max_ratio())
    return a

def decimal_test_mesh_gaussian_small():
    mg, trunc_mass = dec_mesh_gaussian(1, 5E-03, 2)
    print("number of elements in support:", len(mg))
    a = DiceEnsemble(Decimal(2*len(mg)), mg)
    a.fit_pmf_decimal(4)
    a.compute_ratios()
    print("max ratio for 2: ", a.max_ratio())
    return a

def empirical_ratios_mg(sigma=0.5, mesh_size=4E-04, L=2.0):
    pmf, _ = dec_mesh_gaussian(sigma, mesh_size, L)
    de = DiceEnsemble(Decimal(2*len(pmf)), pmf)
    de.fit_pmf_decimal(40)
    de.save("data/de_10k_40d.p")
    de.compute_ratios()
    print("max ratio: ", de.max_ratio())
    return de

def discrete_gaussian_test(sigma=1.0, T=10000, n_dice=40, filename=None, save=False):
    target_pmf, tp1, tp2 = dec_dg(0, sigma, 10000, 20) 
    de = DiceEnsemble(Decimal(T), target_pmf)
    print("fitting pmf")
    de.fit_pmf_decimal(n_dice)
    print("computing ratios")
    de.compute_ratios()
    print("computing empirical masses")
    ems = de.em_masses()
    print("done.")
    if save:
        if not filename:
            f = "data/dg_de_test_default"
        else:
            f = filename
        print("saving to: " + f + ".p")
        de.save(f + ".p")
        approx_pmf = ems[39]
        sample_dict = pmf_to_sample_dict(approx_pmf)
        save_as_json(sample_dict, f + ".json")
    return de


# constructs dice ensemble approximations of discrete gaussian distributions with varying parameters
def dea_discrete_gaussian(sigma=1.0, n_dice=64, filename=None, save=False):
    target_pmf, tp1, tp2 = target_trunc_mass_dg(0, sigma, 2**20, 2**19, 2**-64)
    T = len(target_pmf)*2
    de = DiceEnsemble(Decimal(T), target_pmf)
    print("fitting pmf")
    de.fit_pmf_decimal(n_dice)
    print("computing ratios")
    de.compute_ratios()
    print("computing empirical masses")
    ems = de.em_masses()
    print("done.")
    if save:
        if not filename:
            f = "data/dg_de_default"
        else:
            f = filename
        print("saving to: " + f + ".p")
        de.save(f + ".p")
        approx_pmf = ems[39]
        sample_dict = pmf_to_sample_dict(approx_pmf)
        save_as_json(sample_dict, f + ".json")
    return de




if __name__ == '__main__':
    # examples of how to use the functions in this script:
    ndice = 64
    dice_ensemble = dea_discrete_gaussian(sigma=1.0, n_dice=ndice) # run Algorithm 4 for a discrete gaussian at varying parameters
    sample_dict = pmf_to_sample_dict(dice_ensemble.em_mass_ls[ndice-1])
    de_samples = [sample_from_dict(sample_dict) for i in range(10)] # sample 10 values from the dice ensemble
    print("samples from discrete gaussian dice ensemble with sigma=1.0: ", de_samples)
    print("values in first lookup table: ", dice_ensemble.body[1])
    print("values in last lookup table: ", dice_ensemble.body[64])
    print("--------------------------------")
    print("dice ensemble for a simple toy distribution:")
    decimal_test_toy() # run algorithm 4 on a simple toy distribution
    #plot_table_size_vs_sd() # uncomment to see plot of required table size vs standard deviation for the discrete gaussian