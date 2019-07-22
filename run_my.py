import numpy as np
from mylda import LDA
import lda.datasets
X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

model = LDA(n_topics=20, n_iter=1500, rand_seed=1)
model.fit(X)  # model.fit_transform(X) is also available

n_top_words = 8
for i, topic_dist in enumerate(model.mat_topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))