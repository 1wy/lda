#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
import numpy as np
cimport numpy as np
# from libc.stdlib cimport malloc, free

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil

cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)

cdef int searchsorted(double [:] x, int length, double value) nogil:
    cdef int imin = 0, imax = length
    cdef int imid
    if x[imax-1] <= value:
        return imax - 1
    while imin < imax:
        imid = imin + ((imax-imin)>>1)
        if x[imid] < value:
            imin = imid + 1
        else:
            imax = imid
    return imin

cpdef double loglikelihood(int [:,:] mat_doc_topic, int [:,:] mat_topic_word,int [:] n_m, int [:] n_k, double alpha, double beta) nogil:

    cdef int M = mat_doc_topic.shape[0], K = mat_doc_topic.shape[1]
    cdef int V = mat_topic_word.shape[1]
    cdef int m, k, v

    cdef double ll = 0
    cdef double lg_alpha = lgamma(alpha)
    cdef double lg_beta = lgamma(beta)

    ll += M*lgamma(alpha*K)
    for m in range(M):
        ll -= lgamma(alpha*K + n_m[m])
        for k in range(K):
            if mat_doc_topic[m,k] > 0:
                ll += lgamma(alpha+mat_doc_topic[m,k]) - lg_alpha

    ll += K*lgamma(beta*V)
    for k in range(K):
        ll -= lgamma(beta*V + n_k[k])
        for v in range(V):
            if mat_topic_word[k,v] > 0:
                ll += lgamma(beta+mat_topic_word[k,v]) - lg_beta
    return ll

cpdef sample_topic(int [:] Wm, int [:] Wv, int [:] Wz, int [:,:] mat_doc_topic, int [:,:] mat_topic_word, int [:] n_k, double alpha, double beta, double [:] rands):
    cdef int M = mat_doc_topic.shape[0], K = mat_doc_topic.shape[1]
    cdef int V = mat_topic_word.shape[1]
    cdef int m, i, v, z, k, z_new
    cdef int N = Wm.shape[0], N_rands = rands.shape[0]
    cdef double p_z_cum = 0, r = 0
    cdef double [:] p_z_cum_arr = np.zeros(K)
    # cdef double* p_z_cum_arr = <double*> malloc(K * sizeof(double))
    with nogil:
        for i in range(N):
            m = Wm[i]
            v = Wv[i]
            z = Wz[i]

            dec(mat_doc_topic[m, z])
            dec(mat_topic_word[z, v])
            dec(n_k[z])

            p_z_cum = 0
            for k in range(K):
                p_z_cum += (mat_doc_topic[m,k]+alpha) * (mat_topic_word[k,v]+beta) / (n_k[k]+beta*V)
                p_z_cum_arr[k] = p_z_cum

            r = rands[i % N_rands] * p_z_cum
            z_new = searchsorted(p_z_cum_arr, K, r)

            Wz[i] = z_new
            inc(mat_doc_topic[m, z_new])
            inc(mat_topic_word[z_new, v])
            inc(n_k[z_new])
        # free(p_z_cum_arr)
