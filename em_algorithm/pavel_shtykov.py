from scipy.signal import fftconvolve
from scipy.special import softmax
import numpy as np

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    def nice_conv(a, b):
        return fftconvolve(a, np.flip(b), mode="valid", axes=(0, 1))

    F = F[:, :, None]
    bg_part = (X - B[:, :, None]) ** 2 / (2 * s ** 2)
    bg_part_sum = np.sum(bg_part, axis=(0, 1))
    bg_inter_face = nice_conv(bg_part, np.ones_like(F))

    face_part = (
        nice_conv(X ** 2, np.ones_like(F))  # X^2
        - 2 * nice_conv(X, F)               # - 2 * X * F
        + np.sum(F ** 2)                    # + F^2 = (X - F)^2
    ) / (2 * s ** 2)

    ret = (
        - (bg_part_sum + face_part - bg_inter_face)
        - X.shape[0] * X.shape[1] * np.log(np.sqrt(2 * np.pi) * s)
    )

    return ret
    

def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    
    L = calculate_log_probability(X, F, B, s) + np.log(A + 1e-20)[:, :, None]
    if use_MAP:
        L = L[q[0, :], q[1, :], np.arange(q.shape[-1])].sum()
    else:
        L = (
            np.einsum("ijk,ijk->ij", L, q) 
            - np.einsum("ijk,ijk->ij", np.log(q, where=(q!=0)), q)
        ).sum()

    return L



def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    pXd_ThetaA = calculate_log_probability(X, F, B, s) + np.log(A + 1e-20)[:, :, None]
    if use_MAP:
        ret = np.array(np.unravel_index(
            indices=pXd_ThetaA.reshape(-1, pXd_ThetaA.shape[-1]).argmax(0),
            shape=pXd_ThetaA.shape[:2]
        ))
    else:
        ret = softmax(pXd_ThetaA, (0, 1))

    return ret



def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    if use_MAP:
        H, W, K = X.shape

        F = np.zeros((h, w))
        B = np.zeros((H, W))
        A = np.zeros((H - h + 1, W - w + 1))

        B_denum = np.zeros_like(B)
        B_ones = np.ones_like(B)
        F_ones = np.ones_like(F)
        for i in range(K):
            q_h, q_w = q[:, i]

            A[q_h, q_w] += 1 / K

            F += X[q_h:q_h+h, q_w:q_w+w, i] / K

            B += X[:, :, i]
            B[q_h:q_h+h, q_w:q_w+w] -= X[q_h:q_h+h, q_w:q_w+w, i]
            B_denum += B_ones
            B_denum[q_h:q_h+h, q_w:q_w+w] -= F_ones

        B = np.divide(B, B_denum, where=(B_denum != 0))

        F_part = B_part = 0
        for i in range(K):
            q_h, q_w = q[:, i]

            F_part += ((X[q_h:q_h+h, q_w:q_w+w, i] - F) ** 2).sum()
            B_part += (
                ((X[:, :, i] - B) ** 2).sum()
                - ((X[q_h:q_h+h, q_w:q_w+w, i] - B[q_h:q_h+h, q_w:q_w+w]) ** 2).sum()
            )
        
        s = np.sqrt((F_part + B_part) / np.prod(X.shape))
    else:
        A = q.mean(axis=2)

        F = fftconvolve(X, np.flip(q), mode="valid")[:, :, 0] / X.shape[2]

        q_d_k_not_face = 1 - fftconvolve(q, np.ones_like(F)[:, :, None], mode="full")
        num = X * q_d_k_not_face
        denum = q_d_k_not_face
        B = (num / (denum.sum(2)[:, :, None] + 1e-8)).sum(2)

        X_wo_face = (
            fftconvolve(X ** 2, np.ones_like(F)[:, :, None], mode="valid")
            - 2 * fftconvolve(X, np.flip(F[:, :, None]), mode='valid')
            + np.sum(F ** 2)
        ) 
        F_part = (q * X_wo_face).sum()
        B_part = ((X - B[:, :, None]) ** 2 * q_d_k_not_face).sum()
        s = np.sqrt((F_part + B_part) / np.prod(X.shape))

    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    H, W, _ = X.shape
    if F is None:
        F = np.random.random((h, w)) * 255
    if s is None:
        s = np.random.random() * 511
    if B is None:
        B = np.random.random((H, W)) * 255
    if A is None:
        A = np.ones([H - h + 1, W - w + 1])
        A /= np.sum(A)

    Ls = []
    Ls.append(-np.inf)
    for _ in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        Ls.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP))
        if Ls[-1] - Ls[-2] < tolerance:
            break

    return F, B, s, A, np.array(Ls)[1:]


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    all_res = [
        list(run_EM(X, h, w, F=None, B=None, s=None, A=None, 
                tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP))
        for _ in range(n_restarts)
    ]
    best_res = max(all_res, key=lambda t: t[-1][-1])
    best_res[-1] = best_res[-1][-1]

    return best_res