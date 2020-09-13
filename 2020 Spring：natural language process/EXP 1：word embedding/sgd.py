#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

import pickle
import glob
import random
import numpy as np
import os
import os.path as op
from datetime import datetime

from utils.config import config as cfg  
from utils.summary import LogSummary
from utils.utils import mkdirs

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    save_dir = os.path.join(cfg.save_path, cfg.exp_name)
    for f in glob.glob(os.path.join(save_dir, "saved_params_*.npy")):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        params_file = os.path.join(save_dir, "saved_params_%d.npy" % st)
        state_file = os.path.join(save_dir, "saved_state_%d.pickle" % st)
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    save_dir = os.path.join(cfg.save_path, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    params_file = "saved_params_%d.npy" % iter
    params_save_path = os.path.join(save_dir, params_file)
    state_save_path = os.path.join(save_dir, "saved_state_%d.pickle" % iter)
    
    np.save(params_save_path, params)
    with open(state_save_path, "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = cfg.ANNEAL_EVERY

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    log_dir = os.path.join(cfg.log_dir, datetime.now().strftime('%b%d_%H-%M-%S_') + cfg.exp_name)
    logger = LogSummary(log_dir)

    for iter in range(start_iter + 1, iterations + 1):
        # You might want to print the progress every few iterations.

        loss = None
        ### YOUR CODE HERE (~2 lines)

        loss, gradient = f(x)
        x = x - cfg.LR * gradient

        ### END YOUR CODE

        x = postprocessing(x)

        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter (%d / %d) : %.4f - %.4f" % (iter, iterations, exploss, loss))

        if iter % cfg.log_frequency == 0:
            logger.write_scalars({ 'loss': loss }, tag='train', n_iter=iter)

        if iter % cfg.SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    print("-" * 40)
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    print("-" * 40)
    assert abs(t1) <= 1e-6

    print("-" * 40)
    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    print("-" * 40)
    assert abs(t2) <= 1e-6

    print("-" * 40)
    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    print("-" * 40)
    assert abs(t3) <= 1e-6

    t4 = sgd(quad, np.random.randn(10,10) * 100, 0.01, 1000, PRINT_EVERY=100)
    print("-" * 40)
    t4_loss, _ = quad(t4)
    print("test 4 result:", t4_loss)
    print("-" * 40)
    assert abs(t3) <= 1e-6

    print("-" * 40)
    print("ALL TESTS PASSED")
    print("-" * 40)


if __name__ == "__main__":
    sanity_check()
