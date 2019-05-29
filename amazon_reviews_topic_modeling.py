### https://www.analyticsvidhya.com/blog/2018/10/mining-online-reviews-topic-modeling-lda/
import time

import pandas as pd
pd.set_option('display.max_colwidth', 200)

import nltk
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim import corpora

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
# import seaborn as sns

import spacy
spacy_md = spacy.load('en_core_web_md', disable=['ner', 'parser'])

class TopicModeling(object):

	def __init__(self):
		self.stopwords = stopwords.words('english')

	def readData(self):
		### data szie = rows - 20472, columns - 9
		df = pd.read_json('Automotive_5.json', lines=True)
		print("\n df --- ",df.shape, df.head())
		print("\n columns - ", df.columns)
		# print("\n reviewText - ", df['reviewText'])
		return df

	def preProcessing(self, df):
		df['reviewText'] = df['reviewText'].str.replace("[^a-zA-Z#]", " ")
		df['reviewText'] = df['reviewText'].apply(self.removeStopwords)
		df['reviewText'] = df['reviewText'].apply(self.getLemma)
		reviews = list(df['reviewText'])
		print("\n len(reviews) --- ", len(reviews))
		print("\n reviews_2 --- ", reviews[:3])
		return reviews


	def removeStopwords(self, sent):
		return " ".join([word for word in word_tokenize(sent) if word not in self.stopwords and len(word) > 2]).lower()

	def getLemma(self,sent):
		return [token.lemma_ for token in spacy_md(sent) if token.pos_ in ['NOUN', 'ADJ']]

	def freq_words(self, x):
		all_words = " ".join([text for text in x])
		all_words = all_words.split()
		print("\n all_words in corpus --- ", len(all_words), all_words[:12])
		fdist = FreqDist(all_words)
		# print("\n fdist --- ", fdist)

		words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
		print("\n words counts in corpus --- \n ", words_df.head())

		# selecting top 20 most frequent words
		d = words_df.nlargest(columns='count', n = 20)
		print("\n d --- ", d)
		# plt.figure(figsize=(20,5))
		# ax = sns.barplot(data=d, x='word', y = 'count')
		# ax.set(ylabel='count')
		# plt.show()

	def dataAnalysis(self, df, reviews):
		self.freq_words(df)

	def processTopics(self, topic_string):
		# '0.028*"cable" + 0.017*"car" + 0.012*"dust" + 0.010*"good" + 0.010*"voltage" + 0.010*"load" + 0.009*"truck" + 0.009*"resistance" + 0.009*"bed" + 0.008*"gauge"'
		return [(topic.split('*')[0], topic.split('*')[1]) for topic in topic_string.replace('"', '').split('+')]

	def TrainTopicModel(self, reviews):
		print("\n reviews to model --- ", reviews)
		dictionary = corpora.Dictionary(reviews)
		print("\n dictionary ...")
		doc_term_matrix = [dictionary.doc2bow(lst) for lst in reviews]
		LDA = gensim.models.ldamodel.LdaModel
		lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=10, chunksize=1000, passes=50, random_state=42)
		topics = lda_model.print_topics()
		print("\n topics --- ", topics)
		new_topics = []
		new_topics = [(tpl[0], self.processTopics(tpl[1])) for tpl in topics]
		print("\n new_topics --- ", new_topics)

		### visualize topics
		# vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
		# print(vis)

	def main(self):
		start_time = time.time()
		df = self.readData()
		print("\n read time --- ", time.time() - start_time)
		reviews = self.preProcessing(df)
		print("\n preProcessing time --- ", time.time() - start_time)
		self.dataAnalysis(df, reviews)
		print("\n data analysis time --- ", time.time() - start_time)
		self.TrainTopicModel(reviews)
		print("\n model training time --- ", time.time() - start_time)


"""
These topics learned can be used :
1. to give tags to the sentence / question on quora, stackoverflow
2. can be used as a features while training the machine learning models
3. can be used in  a recommendation systems. but how ?
"""
if __name__ == '__main__':
	obj = TopicModeling()
	obj.main()