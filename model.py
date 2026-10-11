"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    y = np.asarray(labels)                           # y: (N,) class labels
    if y.size == 0:                             # empty node: define impurity as 0
        return 0.0
    _, counts = np.unique(y, return_counts=True)  # counts: (K,) n_k for each distinct label
    p = counts / y.size                         # p: (K,) class fractions p_k
    return float(1.0 - np.sum(p ** 2))

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    features = np.asarray(features)                       # features: (N, F) data
    labels = np.asarray(labels)                           # labels: (N,) class labels
    mask = features[:, feature_index] <= threshold        # mask: (N,) True -> row goes left
    return features[mask], labels[mask], features[~mask], labels[~mask]

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    parent = np.asarray(parent_labels)                 # parent: (N_P,) labels before the split
    left = np.asarray(left_labels)                     # left: (N_L,) labels in the left child
    right = np.asarray(right_labels   )                   # right: (N_R,) labels in the right child

    n = parent.size                             # n: N_P, number of parent samples
    if n == 0:
        return 0.0

    w_l = left.size / n                         # w_l: fraction of samples going left
    w_r = right.size / n                        # w_r: fraction of samples going right

    child_impurity = w_l * impurity(left) + w_r * impurity(right)  # size-weighted child impurity
    return float(impurity(parent) - child_impurity)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    features = np.asarray(features)                 # features: (N, d) data
    labels = np.asarray(labels)                     # labels: (N,) class labels

    best = {'feature_index': None, 'threshold': None, 'score': 0.0}   # running best, score starts at 0

    for fi in feature_indices:                      # fi: column index j being searched
        u = np.unique(features[:, fi])              # u: sorted distinct values of column fi
        if u.size < 2:                              # constant column: no threshold exists
            continue
        thresholds = (u[:-1] + u[1:]) / 2           # midpoints t_k between consecutive values

        for t in thresholds:                        # t: candidate threshold
            lf, ll, rf, rl = split_dataset(features, labels, fi, t)
            if ll.size == 0 or rl.size == 0:        # skip empty-sided splits
                continue
            s = split_score(labels, ll, rl)         # s: gain Delta G for this candidate
            if s > best['score']:                   # strictly better than the running best
                best = {'feature_index': int(fi), 'threshold': float(t), 'score': float(s)}

    return best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    y = np.asarray(labels)                              # y: (N,) labels at this node
    pure = np.unique(y).size <= 1                  # pure: True if at most one distinct class
    too_deep = depth >= max_depth                  # too_deep: depth limit reached
    too_small = y.size < min_samples_split         # too_small: not enough samples to split
    return bool(pure or too_deep or too_small)

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

