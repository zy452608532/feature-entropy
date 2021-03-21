import numpy as np


def calculate_feature_entropy(birth_point_array):

    """
        Calculate the feature entropy of the birth points of feature maps with respect to a single unit.

        Parameters
        ----------
        birth_point_array: 1-D vector.
            The birth point vector resulted from feature maps with respect to a single unit.

        Returns
        -------
        (feature_entropy, selective_rate, ineffectiveness): tuple.

        feature_entropy: float.
            The feature entropy, where zeros are ruled out.

        selective_rate: float. Between 0 and 1.
            The percentage of zeros.

        ineffectiveness: float.
            The quantitative assessment of ineffectiveness of the given unit.

    """

    zero_num = np.count_nonzero(birth_point_array == 0)
    selective_rate = zero_num/ len(birth_point_array)
    birth_point_array = birth_point_array[birth_point_array != 0]  # remove zeros
    total_num_without0 = len(birth_point_array)
    unique, counts = np.unique(birth_point_array, return_counts=True)
    counts_ratio = counts / total_num_without0
    log_counts_ratio = np.log(counts_ratio)
    feature_entropy = -1 * sum(counts_ratio * log_counts_ratio)

    if selective_rate != 1: 
        ineffectiveness = feature_entropy / (1- selective_rate)
    else: 
        ineffectiveness = feature_entropy

    return (feature_entropy, selective_rate, ineffectiveness)


def unit_importance_rank(bp_array):

    """
        Rank the importance of units within a given layer.

        Parameters
        ----------
        bp_array: 2-D array. [bp Distribution, channels]
            The birth point array resulted from feature maps within the given layer.

        Returns
        -------
        importance_rank: list.
            Unit importance rank, where each element denotes the index of unit.

    """    


    bp_list = [bp_array[:, i] for i in range(bp_array.shape[1])]
    H = np.asarray(list(map(calculate_feature_entropy, bp_list)))
    dH = H[:, 2]
    dH[np.absolute(dH) < 1e-1] = 100*np.log(100)
    importance_rank = dH.argsort()[::-1]
    importance_rank = importance_rank.tolist()
    importance_rank.reverse()
    # print(dH[importance_rank[-30:]])

    return importance_rank