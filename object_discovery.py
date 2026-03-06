# """
# Main functions for applying Normalized Cut.
# Code adapted from LOST: https://github.com/valeoai/LOST
# """

# import torch
# import torch.nn.functional as F
# import numpy as np
# from scipy.linalg import eigh
# from scipy import ndimage

# def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
#     """
#     Implementation of NCut Method.
#     Inputs
#       feats: the pixel/patche features of an image
#       dims: dimension of the map from which the features are used
#       scales: from image to map scale
#       init_image_size: size of the image
#       tau: thresold for graph construction
#       eps: graph edge weight
#       im_name: image_name
#       no_binary_graph: ablation study for using similarity score as graph edge weight
#     """
#     cls_token = feats[0,0:1,:].cpu().numpy() 

#     feats = feats[0,1:,:]
#     feats = F.normalize(feats, p=2)
#     A = (feats @ feats.transpose(1,0)) 
#     A = A.cpu().numpy()
#     if no_binary_graph:
#         A[A<tau] = eps
#     else:
#         A = A > tau
#         A = np.where(A.astype(float) == 0, eps, A)
#     d_i = np.sum(A, axis=1)
#     D = np.diag(d_i)
  
#     # Print second and third smallest eigenvector 
#     _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
#     eigenvec = np.copy(eigenvectors[:, 0])

#     # Using average point to compute bipartition 
#     second_smallest_vec = eigenvectors[:, 0]
#     avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
#     bipartition = second_smallest_vec > avg
    
#     seed = np.argmax(np.abs(second_smallest_vec))

#     if bipartition[seed] != 1:
#         eigenvec = eigenvec * -1
#         bipartition = np.logical_not(bipartition)
#     bipartition = bipartition.reshape(dims).astype(float)

#     # predict BBox
#     pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
#     mask = np.zeros(dims)
#     mask[cc[0],cc[1]] = 1

#     return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

# def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
#     """
#     Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
#     """
#     w_featmap, h_featmap = dims
#     objects, num_objects = ndimage.label(bipartition) 
#     cc = objects[np.unravel_index(seed, dims)]
    

#     if principle_object:
#         mask = np.where(objects == cc)
#        # Add +1 because excluded max
#         ymin, ymax = min(mask[0]), max(mask[0]) + 1
#         xmin, xmax = min(mask[1]), max(mask[1]) + 1
#         # Rescale to image size
#         r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
#         r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
#         pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
#         # Check not out of image size (used when padding)
#         if initial_im_size:
#             pred[2] = min(pred[2], initial_im_size[1])
#             pred[3] = min(pred[3], initial_im_size[0])
        
#         # Coordinate predictions for the feature space
#         # Axis different then in image space
#         pred_feats = [ymin, xmin, ymax, xmax]

#         return pred, pred_feats, objects, mask
#     else:
#         raise NotImplementedError


import torch
import torch.nn.functional as F
import numpy as np
#from scipy.linalg.decomp import eig
import scipy
from scipy.linalg import eigh
from scipy import ndimage
#from sklearn.mixture import GaussianMixture
#from sklearn.cluster import KMeans

def find_best_ncut_threshold(eigenvec, W, d_i, n_thresholds=40):
    """
    Find the threshold that minimises the normalised-cut energy (Shi & Malik 2000).

    NCut(A,B) = cut(A,B)/assoc(A,V)  +  cut(A,B)/assoc(B,V)

    where
      cut(A,B)   = sum of W[i,j] for i in A, j in B
      assoc(A,V) = sum of W[i,j] for i in A, all j  = sum of d_i[i] for i in A

    We sweep `n_thresholds` candidate values uniformly between
    min(eigenvec) and max(eigenvec) and pick the cheapest cut.
    """
    lo, hi = eigenvec.min(), eigenvec.max()
    thresholds = np.linspace(lo, hi, n_thresholds + 2)[1:-1]  # exclude endpoints

    best_thresh = thresholds[0]
    best_energy = np.inf

    assoc_total = d_i.sum()  # == assoc(V, V)

    for t in thresholds:
        mask_A = eigenvec >= t   # boolean, shape (N,)
        mask_B = ~mask_A

        assoc_A = d_i[mask_A].sum()
        assoc_B = assoc_total - assoc_A

        if assoc_A == 0 or assoc_B == 0:
            continue  # degenerate cut – skip

        # cut(A,B) = sum_{i in A, j in B} W[i,j]
        # Efficient: cut = assoc_A - sum_{i in A, j in A} W[i,j]
        W_AA = W[np.ix_(mask_A, mask_A)].sum()
        cut_AB = assoc_A - W_AA

        energy = cut_AB / assoc_A + cut_AB / assoc_B

        if energy < best_energy:
            best_energy = energy
            best_thresh = t

    return best_thresh


def ncut(feats, dims, scales, init_image_size, tau=0, eps=1e-5,
         im_name='', no_binary_graph=False,
         spatial_weight=0.0, spatial_sigma=6.0):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
      spatial_weight: scalar alpha that weights the Gaussian spatial affinity term
      spatial_sigma: sigma (in patch units) for the Gaussian: exp(-||i-j||^2 / sigma^2)
    """
    # main_tokencut.py passes feats as (1, N+1, D): batch + CLS + patch tokens
    cls_token = feats[0, 0:1, :].cpu().numpy()
    feats = feats[0, 1:, :]          # (N, D) — remove batch dim and CLS token
    feats = feats.transpose(0, 1)    # (D, N) — expected by the rest of this function

    feats = F.normalize(feats, p=2, dim=0)   # L2-normalise each patch column
    A = (feats.transpose(0, 1) @ feats)      # (N, D) @ (D, N) = (N, N) cosine sim
    A = A.cpu().numpy()
    A = A + 0.1*(A@A)

    # --- Optional spatial affinity term ---
    if spatial_weight > 0:
        print("spatial_weight", spatial_weight)
        feat_h, feat_w = dims
        rows = np.arange(feat_h).repeat(feat_w)
        cols = np.tile(np.arange(feat_w), feat_h)
        dr = (rows[:, None] - rows[None, :]).astype(np.float32)
        dc = (cols[:, None] - cols[None, :]).astype(np.float32)
        dist2 = dr ** 2 + dc ** 2
        spatial_kernel = np.exp(-dist2 / (spatial_sigma ** 2))
        A = A + spatial_weight * spatial_kernel

    A_raw = A.copy()
    if no_binary_graph:
        A[A < tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)

    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]

    # ---- Bipartition via NCut energy minimisation ----
    best_t = find_best_ncut_threshold(second_smallest_vec, A, d_i, n_thresholds=50)
    bipartition = second_smallest_vec >= best_t
    # # old — mean threshold
    # avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    # bipartition = second_smallest_vec > avg

    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox (single principal CC containing the seed)
    pred, _, objects, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:])
    mask = np.zeros(dims)
    mask[cc[0], cc[1]] = 1

    # mask = torch.from_numpy(bipartition).to('cpu')
    # mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()

    # Return signature matches main_tokencut.py:
    # pred, objects, foreground, seed, bins, eigenvector
    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]


    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError