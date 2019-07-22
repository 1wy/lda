

import numpy as np
from _lda import loglikelihood, sample_topic
import time

class LDA:

	def __init__(self, n_topics, n_iter=2000, alpha=0.1, beta=0.01, rand_seed=0, log_step=10):

		self.K = n_topics
		self.n_iter = n_iter
		self.alpha = alpha
		self.beta = beta
		self.log_step = log_step
		self.rand_seed = rand_seed
		self.rand_state = np.random.RandomState(rand_seed)
		self.rands = self.rand_state.rand(1024 ** 2 // 8)

	def initialize(self, X):
		self.M, self.V = X.shape
		self.mat_doc_topic = np.zeros((self.M, self.K), dtype=np.intc)
		self.mat_topic_word = np.zeros((self.K, self.V), dtype=np.intc)

		self.Wm, self.Wv = np.nonzero(X)
		self.W_cnt = X[self.Wm, self.Wv]
		self.Wm = np.repeat(self.Wm, self.W_cnt).astype(np.intc) # doc index of the word
		self.Wv = np.repeat(self.Wv, self.W_cnt).astype(np.intc) # word index of the word
		self.N = len(self.Wm)
		self.Wz = np.zeros(self.N, dtype=np.intc) # topic of the word

		for i in range(self.N):
			z_i = i % self.K
			self.mat_doc_topic[self.Wm[i],z_i] += 1
			self.mat_topic_word[z_i, self.Wv[i]] += 1
			self.Wz[i] = z_i

		self.n_m = np.sum(self.mat_doc_topic, axis=1).astype(np.intc)
		self.n_k = np.sum(self.mat_topic_word, axis=1).astype(np.intc)
		pass


	def fit(self, X):
		self.initialize(X)
		random_state = np.random.RandomState(self.rand_seed)
		start_time = time.time()
		for iter in range(self.n_iter):
			random_state.shuffle(self.rands)
			if iter % self.log_step == 0:
				ll = loglikelihood(self.mat_doc_topic, self.mat_topic_word, self.n_m, self.n_k, self.alpha, self.beta)
				print('%d / %d loglikelihood: %.0f' % (iter, self.n_iter, ll))
			sample_topic(self.Wm, self.Wv, self.Wz, self.mat_doc_topic, self.mat_topic_word, self.n_k, self.alpha, self.beta, self.rands)
		end_time = time.time()
		print(end_time - start_time)
		ll = loglikelihood(self.mat_doc_topic, self.mat_topic_word, self.n_m, self.n_k, self.alpha, self.beta)
		print('%d / %d loglikelihood: %.0f' % (self.n_iter, self.n_iter, ll))

		self.mat_doc_topic = (self.mat_doc_topic + self.alpha).astype(float)
		self.mat_doc_topic /= self.mat_doc_topic.sum(axis=1, keepdims=True)
		self.mat_topic_word = (self.mat_topic_word + self.beta).astype(float)
		self.mat_topic_word /= self.mat_topic_word.sum(axis=1, keepdims=True)

