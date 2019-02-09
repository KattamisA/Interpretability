from __future__ import print_function
from functions.saliency.saliency_utils import get_smoothed_gradients
import numpy as np


# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, smoothgrad, baseline = None, steps=50,
                         cuda=False, magnitude=False, absolute=False):

    if baseline is None:
        baseline = 0 * inputs

    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]

    if smoothgrad is False:
        grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
        avg_grads = np.average(grads[:-1], axis=0)
        avg_grads = np.transpose(avg_grads, (1, 2, 0))
    else:
        grads = get_smoothed_gradients(scaled_inputs, model, target_label_idx, predict_and_gradients, cuda=cuda,
                                       magnitude=magnitude)
        avg_grads = np.average(grads[:-1], axis=0)

    if absolute:
        integrated_grad = np.abs(inputs - baseline) * avg_grads
    else:
        integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad


def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps,
                                         num_random_trials, cuda, smoothgrad=False, magnitude=False, absolute=False):
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, smoothgrad,
                                               baseline=255.0 * np.random.random(inputs.shape), steps=steps, cuda=cuda,
                                               magnitude=magnitude, absolute=absolute)
        all_intgrads.append(integrated_grad)
        print('the trial number is: [{:>2}/{}]'.format(i+1, num_random_trials), end='\r')
    avg_intgrads = np.average(all_intgrads, axis=0)
    return avg_intgrads
