
import numpy as np
import matplotlib.pyplot as plt
print("hello world")
label = 1 
theta = 0
feature_vector = np.array([1,2,3]) 
theta_0 = 0
def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    print('single update')
    next_theta = np.add(current_theta, np.dot(label, feature_vector))
    next_theta_0 = current_theta_0 + label
    tup1 = (next_theta, next_theta_0)
    return tup1
perceptron_single_step_update(feature_vector, label, theta, theta_0)