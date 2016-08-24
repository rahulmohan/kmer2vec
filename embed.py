import sys
sys.setrecursionlimit(30000)
import os
import time
import theano
import itertools
import tabix
import matplotlib
matplotlib.use('Agg')
import glob
import argparse
import numpy as np
#np.random.seed(1201)
from keras.layers.advanced_activations import ELU
import pickle, cPickle
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from sklearn.metrics import fbeta_score, make_scorer, log_loss, v_measure_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import homogeneity_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier, LassoCV
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from glove import Glove, Corpus
from joblib import Parallel, delayed
from operator import itemgetter
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout,MaxoutDense,TimeDistributedDense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib
from keras.models import model_from_json, model_from_yaml
from auprg import auPRG
#import pyximport; pyximport.install()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
#from one_hot_encode import one_hot_encode
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr, pearsonr
from keras import regularizers
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.cluster import AffinityPropagation, Birch, AgglomerativeClustering
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
#from keras.layers import Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from scipy.stats import rankdata
from deeplift import keras_conversion as kc
from prg import prg
from seya.layers.variational import VariationalDense
from sklearn.decomposition import PCA
from scipy.stats import rankdata
from sklearn.preprocessing import PolynomialFeatures
from deeplift.blobs import MxtsMode
import hdbscan
import copy
import math
import gensim,logging
from word2veckeras.word2veckeras import Word2VecKeras as Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def auprg_score(y_true, y_prob):
	min_val = math.pi
	max_val = 1
	rescaled_probs = (max_val*(y_prob-min_val))/((max_val-min_val)*y_prob)
	recall, precision, thresholds = precision_recall_curve(y_true,rescaled_probs)
	precision_gain = (precision - math.pi)/((1-math.pi)*precision)
	recall_gain = (recall - math.pi)/((1-math.pi)*recall)
	auprg = auc(recall_gain, precision_gain)
	return auprg

class TrainEmbedding(object):
	def __init__(self, fasta_file, window_size, embedding_size, kmer_size):
		self.fasta = fasta_file
		self.window_size = window_size
		self.embedding_size = embedding_size
		self.k = kmer_size
		self.kmers = []
	def chunks(self, iterable, size):
		it = iter(iterable.upper())
		chunk = "".join(list(itertools.islice(it,size)))
		while chunk:
			yield chunk
			chunk = "".join(list(itertools.islice(it,size)))
	def compute_nonoverlapping_kmers(self):
		line_num = 0
		with open(self.fasta) as infile:
			for line in infile:
				if (line_num % 10000 == 0):
					print line_num
				self.kmers.append(list(self.chunks(line.strip("\n").upper(),self.k)))
				line_num = line_num + 1
	def train_embedding(self, out_name):
		self.compute_nonoverlapping_kmers()
		corpus_model = Corpus()
		corpus_model.fit(self.kmers, window=self.window_size)
		print('Dict size: %s' % len(corpus_model.dictionary))
		print('Collocations: %s' % corpus_model.matrix.nnz)
		glove = Glove(no_components=self.embedding_size, learning_rate=0.05)
		glove.fit(corpus_model.matrix, epochs=50, no_threads=210, verbose=True)
		glove.add_dictionary(corpus_model.dictionary)
		glove.save(out_name)
	def train_word2vec(self, out_name):
		self.compute_nonoverlapping_kmers()
		model = Word2Vec(self.kmers, size=self.embedding_size, alpha=0.025, sg=1, window=self.window_size, min_count=50, workers=8, iter=1)
		model.train(self.kmers)
		model.save(out_name)

class CellTypeEmbedding(object):
	# Reads all of the cell type specific DNase GloVe models
	# and computes the combined embedding for each DNA sequence
	def __init__(self, fasta, embedding_size, kmer_size):
		self.fasta = fasta
		self.embedding_size = embedding_size
		self.kmer_size = kmer_size
	# Compute kmers for a single DNA sequence
	def compute_kmers(self, seq, kmer_size):
		it = iter(seq)
		win = [it.next() for cnt in xrange(kmer_size)]
		yield "".join(win)
		for e in it:
			win[:-1] = win[1:]
			win[-1] = e
			yield "".join(win)
	# Compute embeddings without any distance weighting
	def compute_embedding(self):
		models=glob.glob("/mnt/lab_data/kundaje/projects/snpbedding/dnase_peaks/celltype_specific_models/*.model")
		num_lines = sum(1 for line in open(self.fasta))
		embeddings = np.zeros((num_lines, self.embedding_size*len(models)))
		all_kmer_sentences = []
		print("Computing kmers..")
		with open(self.fasta) as infile:
			for line in infile:
				kmers = list(self.compute_kmers(line.strip("\n").upper(),self.kmer_size))
				all_kmer_sentences.append(kmers)
		print("Computing embedding..")
		for i in range(len(models)):
			print("On model " + str(i))
			model = pickle.load(open(models[i],"rb"))
			kmer_vectors = model['word_vectors']
			seen_kmers = model['dictionary'].keys()
			for j in range(len(all_kmer_sentences)):
				try:
					kmer_idxs = itemgetter(*all_kmer_sentences[j])(model['dictionary'])
					embedding = np.mean(kmer_vectors[kmer_idxs,:],axis=0)
					embeddings[j,(i*20):((i*20)+20)] = embedding
				except:
					continue
		return embeddings

class RoadmapEmbedding(object):
	# Trains DNase embedding 
	def __init__(self, signal_files, num_samples, num_features):
		self.signal_files = signal_files
		self.num_samples = num_samples
		self.num_features = num_features
	# Load in features
	def load_features(self):
		all_features = np.zeros((self.num_samples, self.num_features*len(self.signal_files)))
		for i in range(len(self.signal_files)):
			print("Processing " + str(self.signal_files[i]))
			features = np.memmap(self.signal_files[i],shape=(self.num_samples,self.num_features),dtype="float32")
			all_features[:,(i*self.num_features):((i*self.num_features)+self.num_features)] = features
		print("Features shape: " + str(all_features.shape))
		return all_features
	# rank normalize features
	def rank_normalize(self,features):
		print("Rank normalizing features")
		normalized_features = np.copy(features)
		for i in range(features.shape[1]):
			ranks = rankdata(features[:,i])
			normalized_features[:,i] = ranks/float(features.shape[0])
		return normalized_features
	# train autoencoder
	def train_sparse_autoencoder(self, features):
		input_img = Input(shape=(features.shape[1],))
		# encoder
		encoded = Dense(512, activation='relu')(input_img)
		encoded = Dense(256, activation='relu')(encoded)
		# decoder
		decoded = Dense(512, activation='relu')(encoded)
		decoded = Dense(features.shape[1], activation='linear')(decoded)
		# compile and train model
		autoencoder = Model(input=input_img, output=decoded)
		autoencoder.compile(optimizer='rmsprop', loss="mse")
		autoencoder.fit(features, features, shuffle=True, validation_split=0.2, nb_epoch=50, batch_size=100)
		encoder = Model(input=input_img, output=encoded)
		cPickle.dump(encoder,open("dnase_sparse_autoencoder.pkl","wb"))

class Embedding(object):
	# Takes in a GloVe model and a file of DNA sequences
	# and computes the embedding for each sequence
	# using linear or exponential averaging of each 
	# overlapping kmer in each sequence
	def __init__(self, fasta, model, embedding_size, kmer_size):
		self.fasta = fasta
		self.model = model
		self.embedding_size = embedding_size
		self.kmer_size = kmer_size
	# Compute kmers for a single DNA sequence
	def compute_kmers(self, seq, kmer_size):
		it = iter(seq)
		win = [it.next() for cnt in xrange(kmer_size)]
		yield "".join(win)
		for e in it:
			win[:-1] = win[1:]
			win[-1] = e
			yield "".join(win)
	# Compute chunks
	def chunks(self, iterable, size):
		it = iter(iterable.upper())
		chunk = "".join(list(itertools.islice(it,size)))
		while chunk:
			yield chunk
			chunk = "".join(list(itertools.islice(it,size)))
	# Compute non-overlapping kmers
	def compute_nonoverlapping_kmers(self):
		line_num = 0
		kmers = []
		with open(self.fasta) as infile:
			for line in infile:
				if (line_num % 10000 == 0):
					print line_num
				kmers.append(list(self.chunks(line.strip("\n").upper(),self.kmer_size)))
				line_num = line_num + 1
		return kmers
	# Compute raw embedding with no sum
	def full_embedding(self, seq_len):
		num_lines = sum(1 for line in open(self.fasta))
		embeddings = np.zeros((num_lines, self.embedding_size * (seq_len/self.kmer_size)))
		model = pickle.load(open(self.model,"rb"))
		kmer_vectors = model['word_vectors']
		seen_kmers = model['dictionary'].keys()
		kmers = self.compute_nonoverlapping_kmers()
		for i in range(len(kmers)):
			kmers[i].pop()		
			try:
				kmer_idxs = itemgetter(*kmers[i])(model['dictionary'])
				embedding = (kmer_vectors[kmer_idxs,:]).flatten()
				embeddings[i,:] = embedding
			except:
				continue
		return embeddings
	# Compute 3D embedding for LSTM/CNN
	def positional_embedding(self, seq_len):
		num_lines = sum(1 for line in open(self.fasta))
		embeddings = np.zeros((num_lines,(seq_len-self.kmer_size),self.embedding_size))
		model = pickle.load(open(self.model,"rb"))
		kmer_vectors = model['word_vectors']
		seen_kmers = model['dictionary'].keys()
		kmers = self.compute_nonoverlapping_kmers()
		for i in range(len(kmers)):
			kmers[i].pop()		
			try:
				kmer_idxs = itemgetter(*kmers[i])(model['dictionary'])
				embeddings[i,:,:] = kmer_vectors[kmer_idxs,:]
			except:
				continue
		return embeddings
	# Linear average of kmers embeddings - each kmer
	# gets equal weight when computing the average
	def linear_average(self):
		num_lines = sum(1 for line in open(self.fasta))
		embeddings = np.zeros((num_lines, self.embedding_size))
		model = pickle.load(open(self.model,"rb"))
		kmer_vectors = model['word_vectors']
		seen_kmers = model['dictionary'].keys()
		line_num = 0
		with open(self.fasta) as infile:
		    for line in infile:
		        if (line_num % 10000 == 0): print line_num
		        kmers = list(self.compute_kmers(line.strip("\n").upper(),self.kmer_size))
		        try:
		        	kmer_idxs = itemgetter(*kmers)(model['dictionary'])
		        except:
		        	kmers_to_keep = []
		        	for i in range(len(kmers)):
		        		if kmers[i] in seen_kmers:
		        			kmers_to_keep.append(kmers[i])
		        	kmer_idxs = itemgetter(*kmers_to_keep)(model['dictionary'])
		        embedding = kmer_vectors[(kmer_idxs),:]
		        average = np.mean(embedding,axis=0)
		        embeddings[line_num,:] = average
		        line_num = line_num + 1
		return embeddings
	# Linear average, max, and min of kmers embeddings
	def linear_mean_max_min(self):
		num_lines = sum(1 for line in open(self.fasta))
		embeddings = np.zeros((num_lines, self.embedding_size * 2))
		model = pickle.load(open(self.model,"rb"))
		kmer_vectors = model['word_vectors']
		seen_kmers = model['dictionary'].keys()
		line_num = 0
		with open(self.fasta) as infile:
		    for line in infile:
		        if (line_num % 10000 == 0): print line_num
		        kmers = list(self.compute_kmers(line.strip("\n").upper(),self.kmer_size))
		        try:
		        	kmer_idxs = itemgetter(*kmers)(model['dictionary'])
		        except:
		        	kmers_to_keep = []
		        	for i in range(len(kmers)):
		        		if kmers[i] in seen_kmers:
		        			kmers_to_keep.append(kmers[i])
		        	kmer_idxs = itemgetter(*kmers_to_keep)(model['dictionary'])
		        embedding = kmer_vectors[(kmer_idxs),:]
		        average = np.mean(embedding,axis=0)
		        max_embedding = np.max(embedding,axis=0)
		        min_embedding = np.min(embedding,axis=0)
		        features = np.concatenate((average, max_embedding))
		        embeddings[line_num,:] = features
		        line_num = line_num + 1
		return embeddings
	# Exponentially weighted average based on distance from center of region
	def exponential_average(self, half_max):
		num_lines = sum(1 for line in open(self.fasta))
		embeddings = np.zeros((num_lines, self.embedding_size))
		model = pickle.load(open(self.model,"rb"))
		kmer_vectors = model['word_vectors']
		seen_kmers = model['dictionary'].keys()
		# create exponential weights
		right_distance = np.array(range(494))
		left_distance = np.array(range(500))[::-1]
		right_weights = 2**-(right_distance / half_max)
		left_weights = 2**-(left_distance / half_max)
		weights = np.concatenate((left_weights,right_weights))
		line_num = 0
		with open(self.fasta) as infile:
		    for line in infile:
		        if (line_num % 10000 == 0): print line_num
		        kmers = list(self.compute_kmers(line.strip("\n").upper(),self.kmer_size))
		        try:
		        	kmer_idxs = itemgetter(*kmers)(model['dictionary'])
		        except:
		        	kmers_to_keep = []
		        	for i in range(len(kmers)):
		        		if kmers[i] in seen_kmers:
		        			kmers_to_keep.append(kmers[i])
		        	kmer_idxs = itemgetter(*kmers_to_keep)(model['dictionary'])
		        embedding = kmer_vectors[(kmer_idxs),:]
		        try:
		        	average = np.average(embedding,axis=0,weights=weights)
		        except:
		        	continue
		        embeddings[line_num,:] = average
		        line_num = line_num + 1
		return embeddings

class Clustering(object):
	# Cluster embeddings and use clusters as features
	def __init__(self, model):
		self.model = pickle.load(open(model,"rb"))
		self.word_centroid_map = {}
		self.num_centroids = 0
		self.num_clusters = 0
	# Train K-means model on word embeddings
	def create_clusters(self):
		kmer_vectors = self.model['word_vectors']
		kmer2idxs = self.model['dictionary']
		ordered_vocab = []
		for i in range(len(kmer2idxs)):
			word_idx = np.where(np.array(kmer2idxs.values()) == i)[0][0]
			word = kmer2idxs.keys()[word_idx]
			ordered_vocab.append(word)
		self.num_clusters = 50
		kmeans_clustering = KMeans(n_clusters=self.num_clusters,n_jobs=-1,verbose=1)
		idx = kmeans_clustering.fit_predict( kmer_vectors )
		self.word_centroid_map = dict(zip(ordered_vocab, idx))
		pickle.dump(self.word_centroid_map,open("sequence_k=6_window=10_size=50.clusters","wb"))
	# Create bag of centroids features for a single sequence
	def create_bag_of_centroids(self,word_list):
		self.num_centroids = max(self.word_centroid_map.values()) + 1
		bag_of_centroids = np.zeros((self.num_centroids,), dtype="float32")
		for word in word_list:
			if word in self.word_centroid_map:
				index = self.word_centroid_map[word]
				bag_of_centroids[index] += 1
		return bag_of_centroids
	# Print out clusters
	def print_clusters(self,num_to_print):
		for cluster in range(num_to_print):
			print "\nCluster %d" % cluster
			words = []
			for i in range(0,len(self.word_centroid_map.values())):
				if (self.word_centroid_map.values()[i] == cluster):
					words.append(self.word_centroid_map.keys()[i])
			print words

	# Compute kmers for a single DNA sequence
	def compute_kmers(self, seq, kmer_size):
		it = iter(seq)
		win = [it.next() for cnt in xrange(kmer_size)]
		yield "".join(win)
		for e in it:
			win[:-1] = win[1:]
			win[-1] = e
			yield "".join(win)
	# Convert fasta file with sequence to features
	def fasta2features(self, fasta, kmer_size):
		num_lines = sum(1 for line in open(fasta))
		all_features = np.zeros((num_lines,self.num_clusters))
		line_num = 0
		with open(fasta) as infile:
			for line in infile:
				kmers = self.compute_kmers(line,kmer_size)
				features = self.create_bag_of_centroids(kmers)
				all_features[line_num,:] = features
				line_num = line_num + 1
		return all_features

class SupervisedCNN(object):
	# Trains supervised CNN on raw sequence
	def __init__(self, pos_train, neg_train, pos_test, neg_test):
		self.pos_train = pos_train
		self.neg_train = neg_train
		self.pos_test = pos_test
		self.neg_test = neg_test
	# One hot encode fasta sequences
	def encode_sequence_conv(self):
		self.pos_train = one_hot_encode(np.loadtxt(self.pos_train,dtype="str"))
		self.neg_train = one_hot_encode(np.loadtxt(self.neg_train,dtype="str"))
		self.pos_test = one_hot_encode(np.loadtxt(self.pos_test,dtype="str"))
		self.neg_test = one_hot_encode(np.loadtxt(self.neg_test,dtype="str"))
	# Create training and test sets
	def create_train_test(self):
		self.x_train = np.concatenate((self.pos_train, self.neg_train))
		self.y_train = np.concatenate((np.ones((len(self.pos_train))),np.zeros((len(self.neg_train)))))
		self.x_test = np.concatenate((self.pos_test, self.neg_test))
		self.y_test = np.concatenate((np.ones((len(self.pos_test))),np.zeros((len(self.neg_test)))))
	# Train CNN
	def train_cnn_model(self):
		self.encode_sequence_conv()
		self.create_train_test()
		model = Sequential()
		model.add(Convolution2D(32,4,20,input_shape=(1, self.x_train.shape[2], self.x_train.shape[3])))
		model.add(MaxPooling2D(pool_size=(1,20)))
		model.add(Flatten())
		model.add(Dense(256,activation="relu"))
		model.add(Dense(128,activation="relu"))
		model.add(Dense(64,activation="relu"))
		model.add(Dense(1,activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer="adam")
		best_auc = 0
		best_auprg = 0
		for epoch in range(20):
			model.fit(self.x_train,self.y_train, nb_epoch=1, show_accuracy=True, verbose=1)
			preds = model.predict(self.x_test)
			rocauc = roc_auc_score(self.y_test,preds)
			auprg = auPRG(self.y_test,preds)
			print rocauc,auprg
			if rocauc > best_auc:
				best_auc = rocauc
			if auprg > best_auprg:
				best_auprg = auprg
		return best_auc, best_auprg

class Classifier(object):
	# Trains supervised fine-tuning models on top of embedding 
	def __init__(self, pos_train, neg_train, pos_test, neg_test):
		self.pos_train = pos_train
		self.neg_train = neg_train
		self.pos_test = pos_test
		self.neg_test = neg_test
		self.x_train = []
		self.x_test = []
		self.y_train = []
		self.y_test = []
		self.embedding_model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=50.model"
	# Create training and test sets
	def create_train_test(self):
		self.x_train = np.concatenate((self.pos_train, self.neg_train))
		self.y_train = np.concatenate((np.ones((len(self.pos_train))),np.zeros((len(self.neg_train)))))
		self.x_test = np.concatenate((self.pos_test, self.neg_test))
		self.y_test = np.concatenate((np.ones((len(self.pos_test))),np.zeros((len(self.neg_test)))))
		self.x_train,self.y_train = shuffle(self.x_train,self.y_train)
		self.x_test,self.y_test = shuffle(self.x_test,self.y_test)
		scaler=StandardScaler()
		scaler.fit(self.x_train)
		self.x_train = scaler.transform(self.x_train)
		self.x_test = scaler.transform(self.x_test)
	# Train single layer ReLU neural net
	def train_relu_model(self):
		self.create_train_test()
		model = Sequential()
		model.add(Dense(2048,activation="relu",input_shape=(self.x_train.shape[1],)))
		model.add(Dense(1024,activation="relu"))
		model.add(Dense(512,activation="relu"))
		model.add(Dense(1,activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer="adam")
		best_auc = 0
		best_auprg = 0
		best_epoch = 0
		num_epochs = 50
		best_model = []
		for epoch in range(num_epochs):
			model.fit(self.x_train,self.y_train, nb_epoch=1, batch_size=1000, show_accuracy=True, verbose=1, class_weight={0:1,1:10})
			preds = model.predict(self.x_test)
			rocauc = roc_auc_score(self.y_test,preds)
			auprg = auPRG(self.y_test,preds)
			print "DNN AUC: " + str(rocauc)
			print "DNN auPRG: " + str(auprg)
			if rocauc > best_auc:
				best_auc = rocauc
				best_epoch = epoch
			if auprg > best_auprg:
				best_auprg = auprg
		return best_auc, best_auprg, preds, model
	# Get feature importance from DNN using DeepLIFT
	def deeplift(self,model,data):
		deeplift_model = kc.convert_sequential_model(model, mxts_mode=MxtsMode.DeepLIFT)
		target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
		target_contribs = target_contribs_func(task_idx=0, input_data_list=[data],batch_size=200, progress_update=10000)
		target_contribs = np.array(target_contribs)
		return target_contribs
	# Get kmer importance
	def kmer_importance(self, embedding_model, model, out_file_name):
		out_file = open(out_file_name,"wb")
		embedding_model = pickle.load(open(embedding_model,"rb"))
		kmers = embedding_model['dictionary'].keys()
		kmer_idxs = embedding_model['dictionary'].values()
		kmer_vectors = embedding_model['word_vectors']
		true_pos_idxs = np.where(self.y_test == 1)[0]
		true_pos_embedding = self.x_test[true_pos_idxs,:]
		deeplift_scores = self.deeplift(model, true_pos_embedding)
		avg_deeplift_scores = np.mean(np.array(deeplift_scores),axis=0)
		print avg_deeplift_scores.shape, kmer_vectors.shape
		contribs = kmer_vectors.dot(avg_deeplift_scores)
		ranked_contribs = np.argsort(contribs.flatten())
		for i in range(len(ranked_contribs)-1,-1,-1):
			kmer_idx = ranked_contribs[i]
			score = contribs[kmer_idx]
			kmer = kmers[np.where(kmer_idxs==kmer_idx)[0]]
			out_file.write(kmer + "\t" + str(score) + "\n")
		print("Finished kmer importance computation")
		out_file.close()
	# Simple baseline summing embeddings
	def sum_embedding(self):
		self.create_train_test()
		sum_x_test = np.sum(self.x_test,axis=1)
		rocauc = roc_auc_score(self.y_test,sum_x_test)
		auprg = auPRG(self.y_test,sum_x_test)
		print "Sum AUC: " + str(roc_auc_score(self.y_test,sum_x_test))
		print "Sum auPRG: " + str(auPRG(self.y_test,sum_x_test))
		return rocauc,auprg
	# Train AdaBoost
	def train_adaboost_model(self):
		self.create_train_test()
		model = AdaBoostClassifier(n_estimators=200)
		model.fit(self.x_train,self.y_train)
		preds = model.predict_proba(self.x_test)
		rocauc = roc_auc_score(self.y_test,preds[:,1])
		auprg = auPRG(self.y_test,preds[:,1])
		return rocauc,auprg,preds,model
	# Train GBM
	def train_gradient_boosting_model(self):
		self.create_train_test()
		model = GradientBoostingClassifier(n_estimators=30, learning_rate=1.0, max_depth=1, random_state=0)
		model.fit(self.x_train,self.y_train)
		preds = model.predict_proba(self.x_test)
		rocauc = roc_auc_score(self.y_test,preds[:,1])
		auprg = auPRG(self.y_test,preds[:,1])
		return rocauc,auprg,preds,model
	# Train LASSO
	def train_lasso_model(self):
		self.create_train_test()
		model = LogisticRegression(C=0.05)
		model.fit(self.x_train,self.y_train)
		preds = model.predict_proba(self.x_test)
		rocauc = roc_auc_score(self.y_test,preds[:,1])
		prg_curve = prg.create_prg_curve(self.y_test,preds[:,1], create_crossing_points=True)
		rocauc = roc_auc_score(self.y_test,preds[:,1])
		auprg = prg.calc_auprg(prg_curve)
		print np.sum(self.y_test)/float(len(self.y_test))
		return rocauc,auprg,preds,model
	# Train SVM
	def train_svm_model(self):
		self.create_train_test()
		model = SVC(C=2,probability=True, class_weight={0:1,1:25})
		model.fit(self.x_train,self.y_train)
		preds = model.predict_proba(self.x_test)
		rocauc = roc_auc_score(self.y_test,preds[:,1])
		auprg = auPRG(self.y_test,preds[:,1])
		return rocauc,auprg
	# Train ensemble
	def train_ensemble_model(self):
		self.create_train_test()
		clf1 = RandomForestClassifier(n_estimators=500,max_depth=1,n_jobs=512,class_weight="balanced")
		clf2 = SGDClassifier(class_weight="balanced",loss="log")
		clf3 = GaussianNB()
		clf4 = AdaBoostClassifier()
		clf5 = KNeighborsClassifier()
		print("Training ensemble model..")
		model = VotingClassifier(estimators=[('rf', clf1), ('gnb', clf3), ('svm',clf2), ('ada',clf4), ('nn',clf5)], voting='soft')
		model.fit(self.x_train,self.y_train)
		preds = model.predict_proba(self.x_test)
		rocauc = roc_auc_score(self.y_test,preds[:,1])
		auprg = auPRG(self.y_test,preds[:,1])
		return rocauc,auprg

class EIGEN(object):
	# Evaluate EIGEN
	def __init__(self, pos_train, neg_train, pos_test, neg_test):
		self.pos_scores = []
		self.neg_scores = []
	def evaluate_from_files(self,pos_file,neg_file):
		self.pos_scores = np.loadtxt(pos_file,dtype="str")[:,-1].astype("float")
		self.neg_scores = np.loadtxt(neg_file,dtype="str")[:,-3].astype("float")
		pos_labels = np.ones((len(self.pos_scores)))
		neg_labels = np.zeros((len(self.neg_scores)))
		eigen_preds = np.concatenate((self.pos_scores, self.neg_scores))
		labels = np.concatenate((pos_labels, neg_labels))
		rocauc = roc_auc_score(labels,eigen_preds)
		auprg = auPRG(labels,eigen_preds)
		print rocauc,auprg


class Regressor(object):
	# Trains supervised fine-tuning models on top of embedding 
	def __init__(self, train_embedding, train_targets, test_embedding, test_targets):
		self.x_train = train_embedding
		self.y_train = train_targets
		self.x_test = test_embedding
		self.y_test = test_targets
	# Train single layer ReLU neural net
	def train_relu_model(self):
		model = Sequential()
		model.add(Dense(2048,activation="relu",input_shape=(self.x_train.shape[1],)))
		model.add(Dense(1,activation="linear"))
		model.compile(loss='mse', optimizer="adam")
		for epoch in range(10):
			model.fit(self.x_train,self.y_train, nb_epoch=1, show_accuracy=True, verbose=1)
			preds = model.predict(self.x_test)
			spearman = spearmanr(self.y_test, preds)
			test_pearson = pearsonr(self.y_test, preds[:,0])
			print spearman,test_pearson
		return model
	# Train deepLIFT model
	def deeplift(self,model):
		deeplift_model = kc.convert_sequential_model(model, mxts_mode=MxtsMode.DeepLIFT)
		target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-1)
		target_contribs = target_contribs_func(task_idx=0, input_data_list=[self.x_test],batch_size=200, progress_update=10000)
		target_contribs = np.array(target_contribs)
		return target_contribs

from gruln import GRULN
def precision_at_recall_threshold(labels, predictions, recall_threshold):
    precision, recall = precision_recall_curve(labels, predictions)[:2]
    return 100 * precision[np.searchsorted(recall - recall_threshold, 0)]

def recall_at_fdr(y_true, y_score, recall_cutoff=0.1):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fdr = 1-recall
    cutoff_index = next(i for i, x in enumerate(fdr) if x > recall_cutoff)
    return precision[cutoff_index-1]

class RandomClassifier(object):
	# Trains supervised fine-tuning models on top of embedding 
	# with random splits with pos and neg set
	def __init__(self, pos, neg):
		self.pos = pos
		self.neg = neg
		self.x_train = []
		self.x_test = []
		self.y_train = []
		self.y_test = []
	# Create training and test sets
	def create_train_test(self):
		pos_labels = np.ones((len(self.pos)))
		neg_labels = np.zeros((len(self.neg)))
		self.data = np.concatenate((self.pos,self.neg))
		self.labels = np.concatenate((pos_labels,neg_labels))
		self.data, self.labels = shuffle(self.data, self.labels)
	# Train GRU that takes into account positional information
	def train_gru(self,x_train, x_test, y_train, y_test):
		model = Sequential()
		model.add(GRULN(15,return_sequences=False, input_shape=(x_train.shape[1],x_train.shape[2])))
		model.add(Dense(1, activation="sigmoid"))
		optimizer = RMSprop(clipnorm=0.001)
		model.compile(loss='binary_crossentropy', optimizer=optimizer)
		best_auc = 0
		best_auprg = 0
		best_preds = []
		for epoch in range(20):
			model.fit(x_train,y_train, nb_epoch=1, show_accuracy=False, verbose=1)
			preds = model.predict_proba(x_test)
			rocauc = roc_auc_score(y_test,preds)
			auprg = auPRG(y_test,preds)
			print rocauc,auprg
			if rocauc > best_auc:
				best_auc = rocauc
			if auprg > best_auprg:
				best_auprg = auprg
				best_preds = np.copy(preds)
		return best_auc, best_auprg, best_preds
	# Train CNN
	def train_cnn(self,x_train, x_test, y_train, y_test):
		x_train = np.reshape(x_train, (len(x_train),1,x_train.shape[1],x_train.shape[2]))
		x_test = np.reshape(x_test, (len(x_test),1,x_test.shape[1],x_test.shape[2]))
		model = Sequential()
		model.add(Convolution2D(10,10,10,input_shape=(1,x_train.shape[2],x_train.shape[3])))
		model.add(MaxPooling2D(pool_size=(5,5)))
		model.add(Flatten())
		model.add(Dense(256, activation="relu"))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer="adam")
		best_auc = 0
		best_auprg = 0
		best_preds = []
		for epoch in range(20):
			model.fit(x_train,y_train, nb_epoch=1, show_accuracy=False, verbose=1, class_weight={0:1,1:15})
			preds = model.predict_proba(x_test)
			rocauc = roc_auc_score(y_test,preds)
			auprg = auPRG(y_test,preds)
			print rocauc,auprg
			if rocauc > best_auc:
				best_auc = rocauc
				if auprg > best_auprg:
					best_auprg = auprg
					best_preds = np.copy(preds)
		return best_auc, best_auprg, best_preds

	# Get feature importance from DNN using DeepLIFT
	def deeplift(self,model,data):
		deeplift_model = kc.convert_sequential_model(model, mxts_mode=MxtsMode.DeepLIFT)
		target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
		target_contribs = target_contribs_func(task_idx=0, input_data_list=[data],batch_size=200, progress_update=10000)
		target_contribs = np.array(target_contribs)
		return target_contribs
	# deeplift based kmer importance
	def kmer_importance(self, embedding_model, model, x_test, y_test, preds, out_file_name):
		out_file = open(out_file_name,"wb")
		embedding_model = pickle.load(open(embedding_model,"rb"))
		kmers = embedding_model['dictionary'].keys()
		kmer_idxs = embedding_model['dictionary'].values()
		kmer_vectors = embedding_model['word_vectors']
		true_pos_idxs = np.where((preds[:,0] > 0.5) & (y_test == 1))[0]
		print len(true_pos_idxs)
		true_pos_embedding = x_test[true_pos_idxs,:]
		deeplift_scores = self.deeplift(model, true_pos_embedding)
		avg_deeplift_scores = np.mean(np.array(deeplift_scores),axis=0)
		print avg_deeplift_scores.shape, kmer_vectors.shape
		contribs = kmer_vectors.dot(avg_deeplift_scores)
		ranked_contribs = np.argsort(contribs.flatten())
		for i in range(len(ranked_contribs)-1,-1,-1):
			kmer_idx = ranked_contribs[i]
			score = contribs[kmer_idx]
			kmer = kmers[np.where(kmer_idxs==kmer_idx)[0]]
			out_file.write(kmer + "\t" + str(score) + "\n")
		print("Finished kmer importance computation")
		out_file.close()
	# Train single layer ReLU neural net
	def train_relu_model(self,x_train, x_test, y_train, y_test):
		model = Sequential()
		model.add(Dense(2048,activation="relu",input_shape=(x_train.shape[1],)))
		model.add(Dense(1024,activation="relu"))
		model.add(Dense(512,activation="relu"))
		model.add(Dense(1,activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer="adam")
		best_auc = 0
		best_auprg = 0
		best_preds = []
		best_model = []
		for epoch in range(20):
			model.fit(x_train,y_train, nb_epoch=1, batch_size=20, show_accuracy=True, verbose=1, class_weight={0:1,1:3})
			preds = model.predict_proba(x_test)
			rocauc = roc_auc_score(y_test,preds)
			prg_curve = prg.create_prg_curve(y_test,preds, create_crossing_points=True)
			prg.plot_prg(prg_curve)
			precision_at_10_recall = recall_at_fdr(y_test, preds)
			precision, recall = precision_recall_curve(y_test, preds)[:2]
			auprg = auPRG(y_test,preds)
			if rocauc > best_auc:
				best_auc = rocauc
				best_preds = copy.deepcopy(preds)
				best_model = copy.deepcopy(model)
			if auprg > best_auprg:
				best_auprg = auprg
		print best_auc, best_auprg
		return best_auc, best_auprg, best_preds, best_model
	# sum test
	def sum_embedding(self, x_train, x_test, y_train, y_test):
		sum_x_test = np.sum(x_test,axis=1)
		rocauc = roc_auc_score(y_test,sum_x_test)
		auprg = auPRG(y_test,sum_x_test)
		return rocauc,auprg,sum_x_test
	# Train AdaBoost
	def train_adaboost_model(self, x_train, x_test, y_train, y_test):
		self.create_train_test()
		model = GradientBoostingClassifier(n_estimators=30,learning_rate=0.1,max_depth=3)
		model.fit(x_train,y_train)
		preds = model.predict_proba(x_test)
		prg_curve = prg.create_prg_curve(y_test,preds[:,1], create_crossing_points=True)
		rocauc = roc_auc_score(y_test,preds[:,1])
		auprg = prg.calc_auprg(prg_curve)
		recall, precision, _ = precision_recall_curve(y_test, preds[:,1])
		auprc = auc(recall, precision, reorder=True)
		calibrated_classifier = CalibratedClassifierCV(base_estimator=model,cv="prefit",method="sigmoid")
		calibrated_classifier.fit(np.concatenate((x_train,x_test)),np.concatenate((y_train,y_test)))
		calibrated_probs = calibrated_classifier.predict_proba(x_test)
		return rocauc,auprg,calibrated_probs[:,1],model
	# Train ensemble model
	def train_ensemble_model(self, x_train, x_test, y_train, y_test):
		self.create_train_test()
		clf1 = RandomForestClassifier(n_estimators=500,max_depth=1,n_jobs=512,class_weight="balanced")
		clf2 = SGDClassifier(class_weight="balanced",loss="log")
		clf3 = GaussianNB()
		clf4 = AdaBoostClassifier()
		clf5 = KNeighborsClassifier()
		print("Training ensemble model..")
		model = VotingClassifier(estimators=[('rf', clf1), ('gnb', clf3), ('svm',clf2), ('ada',clf4), ('nn',clf5)], voting='soft')
		model.fit(x_train,y_train)
		preds = model.predict_proba(x_test)
		rocauc = roc_auc_score(y_test,preds[:,1])
		auprg = auPRG(y_test,preds[:,1])
		return rocauc,auprg
	# Train logistic regression
	def train_log_reg(self, x_train, x_test, y_train, y_test):
		scorer = make_scorer(log_loss)
		#model = LogisticRegressionCV(scoring=scorer,class_weight="balanced")
		model = LogisticRegression(C=1,class_weight="balanced",penalty='l2')
		model.fit(x_train,y_train)
		preds = model.predict_proba(x_test)
		prg_curve = prg.create_prg_curve(y_test,preds[:,1], create_crossing_points=True)
		rocauc = roc_auc_score(y_test,preds[:,1])
		auprg = prg.calc_auprg(prg_curve)
		recall, precision, _ = precision_recall_curve(y_test, preds[:,1])
		auprc = auc(recall, precision, reorder=True)
		return rocauc,auprg
	# train SVM
	def train_svm(self, x_train, x_test, y_train, y_test):
		model = SVC(C=0.5,probability=True,class_weight="balanced")
		model.fit(x_train,y_train)
		preds = model.predict_proba(x_test)
		prg_curve = prg.create_prg_curve(y_test,preds[:,1], create_crossing_points=True)
		rocauc = roc_auc_score(y_test,preds[:,1])
		auprg = prg.calc_auprg(prg_curve)
		return rocauc,auprg,preds[:,1]
	# run cross val
	def run_kfold_cross_val(self, average_across_folds=True):
		self.create_train_test()
		skf = StratifiedKFold(self.labels, n_folds=5)
		embedding_model="/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=75.model"
		all_labels = []
		all_preds = []
		aucs = []
		auprgs = []
		fold_num = 0
		for train_index, test_index in skf:
			X_train, X_test = self.data[train_index], self.data[test_index]
			y_train, y_test = self.labels[train_index], self.labels[test_index]
			scaler=StandardScaler().fit(X_train)
			X_train = scaler.transform(X_train)
			X_test = scaler.transform(X_test)
			auc,auprg = self.train_relu_model(X_train,X_test,y_train,y_test)
			print("AUC: "+str(auc) + ", AUPRG: "+str(auprg))
			#if (auprg > 0.75):
			#	self.kmer_importance(embedding_model, model, X_test, y_test, preds, "dsqtls_kmer_importance.txt")
			aucs.append(auc)
			auprgs.append(auprg)
			fold_num = fold_num + 1
		if average_across_folds:
			print("Stratified k-fold cross val, AUC: " + str(np.mean(aucs)), ", AUPRG: " + str(np.mean(auprgs)))
		else:
			all_labels=np.concatenate(all_labels)
			all_preds=np.concatenate(all_preds)
			prg_curve = prg.create_prg_curve(all_labels,all_preds, create_crossing_points=True)
			rocauc = roc_auc_score(all_labels,all_preds)
			auprg = prg.calc_auprg(prg_curve)
			print("Stratified k-fold cross val, AUC: " + str(rocauc), ", AUPRG: " + str(auprg))

class DeepLIFT(object):
	# Run DeepLIFT on DNNs
	def __init__(self, model_yaml, model_weights, data):
		self.model_yaml = model_yaml
		self.model_weights = model_weights
		self.data = data
	# Get deepLIFT scores
	def compute_importance_scores(self):
		model = model_from_yaml(open(self.model_yaml,"rb"))
		model.load_weights(self.model_weights)
		deeplift_model = kc.convert_sequential_model(model, mxts_mode=MxtsMode.DeepLIFT)
		target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
		target_contribs = target_contribs_func(task_idx=0, input_data_list=[self.data],batch_size=200,progress_update=10000)
		target_contribs = np.array(target_contribs)
		return target_contribs

def run_cross_validation():
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/low_quality_variant_task/folds/"
	dnase_model = "/mnt/lab_data/kundaje/projects/snpbedding/sequence_embedding_asdhs_glove_k=7_window=12_size=256.model"
	conservation_model = "/mnt/lab_data/kundaje/projects/snpbedding/sequence_embedding_GERP_conserved_glove_k=7_window=12_size=256.model"
	path, dirs, files = os.walk(splits).next()
	cv_stats = {'auc':[],'auprg':[],'deeplift_auc':[],'deeplift_auprg':[]}
	num_examples = []
	all_labels = []
	all_preds = []
	aucs = []
	auprgs = []
	embedding_size = 256
	kmer_size = 7
	for i in range(5):
		pos_train_file = splits + dirs[i] + "/pos_train.fa"
		neg_train_file = splits + dirs[i] + "/neg_train.fa"
		pos_test_file = splits + dirs[i] + "/pos_test.fa"
		neg_test_file = splits + dirs[i] + "/neg_test.fa"
		pos_train_1 = Embedding(pos_train_file,dnase_model,embedding_size,kmer_size).linear_average()
		neg_train_1 = Embedding(neg_train_file,dnase_model,embedding_size,kmer_size).linear_average()
		pos_test_1 = Embedding(pos_test_file,dnase_model,embedding_size,kmer_size).linear_average()
		neg_test_1 = Embedding(neg_test_file,dnase_model,embedding_size,kmer_size).linear_average()
		pos_train_2 = Embedding(pos_train_file,conservation_model,embedding_size,kmer_size).linear_average()
		neg_train_2 = Embedding(neg_train_file,conservation_model,embedding_size,kmer_size).linear_average()
		pos_test_2 = Embedding(pos_test_file,conservation_model,embedding_size,kmer_size).linear_average()
		neg_test_2 = Embedding(neg_test_file,conservation_model,embedding_size,kmer_size).linear_average()

		pos_train = np.concatenate((pos_train_1, pos_train_2),axis=1)
		neg_train = np.concatenate((neg_train_1, neg_train_2),axis=1)
		pos_test = np.concatenate((pos_test_1, pos_test_2),axis=1)
		neg_test = np.concatenate((neg_test_1, neg_test_2),axis=1)

		num_test_examples = len(pos_test)+len(neg_test)
		classifier = Classifier(pos_train_1,neg_train_1,pos_test_1,neg_test_1)
		auc, auprg, labels, preds = classifier.train_relu_model()
		cv_stats['auc'].append(auc)
		cv_stats['auprg'].append(auprg)
		num_examples.append(num_test_examples)
		all_labels.append(labels)
		all_preds.append(preds)
		aucs.append(auc)
		auprgs.append(auprg)
		print("Fold " + str(i) + ", AUC: " + str(auc) + ", auPRG: " + str(auprg))
	print("Finished cross validation")
	print("Cross validation - AUC: " + str(np.mean(aucs)) + ", auPRG: " + str(np.mean(auprgs)))
	'''
	all_labels = np.concatenate(all_labels)
	all_preds = np.concatenate(all_preds)
	print all_labels.shape,all_preds.shape

	prg_curve = prg.create_prg_curve(all_labels,all_preds, create_crossing_points=True)
	rocauc = roc_auc_score(all_labels,all_preds)
	auprg = prg.calc_auprg(prg_curve)
	print rocauc,auprg

	cv_auc = np.average(cv_stats['auc'],weights=np.array(num_examples))
	cv_auprg = np.average(cv_stats['auprg'],weights=np.array(num_examples))
	print("Cross validation - AUC: " + str(cv_auc) + ", auPRG: " + str(cv_auprg))
	'''

def run_supervised_baseline():
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/dsQTL_deltaSVM_task/folds/"
	path, dirs, files = os.walk(splits).next()
	cv_stats = {'auc':[],'auprg':[]}
	for i in range(len(dirs)):
		pos_train_file = splits + dirs[i] + "/pos_train.fa"
		neg_train_file = splits + dirs[i] + "/neg_train.fa"
		pos_test_file = splits + dirs[i] + "/pos_test.fa"
		neg_test_file = splits + dirs[i] + "/neg_test.fa"
		classifier = SupervisedCNN(pos_train_file,neg_train_file,pos_test_file,neg_test_file)
		auc, auprg = classifier.train_cnn_model()
		cv_stats['auc'].append(auc)
		cv_stats['auprg'].append(auprg)
		print("Fold " + str(i) + ", AUC: " + str(cv_stats['auc'][i]) + ", auPRG: " + str(cv_stats['auprg'][i]))
	cv_auc = np.mean(cv_stats['auc'])
	cv_auprg = np.mean(cv_stats['auprg'])
	print("Cross validation - AUC: " + str(cv_auc) + ", auPRG: " + str(cv_auprg))

def cluster_embeddings():
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=50.model"
	clusters = Clustering(model)
	clusters.create_clusters()
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/dsQTL_deltaSVM_task/folds/"
	path, dirs, files = os.walk(splits).next()
	cv_stats = {'auc':[],'auprg':[]}
	for i in range(len(dirs)):
		pos_train_file = splits + dirs[i] + "/pos_train.fa"
		neg_train_file = splits + dirs[i] + "/neg_train.fa"
		pos_test_file = splits + dirs[i] + "/pos_test.fa"
		neg_test_file = splits + dirs[i] + "/neg_test.fa"
		pos_train = clusters.fasta2features(pos_train_file, 6)
		neg_train = clusters.fasta2features(neg_train_file, 6)
		pos_test = clusters.fasta2features(pos_test_file, 6)
		neg_test = clusters.fasta2features(neg_test_file, 6)
		pos_train_embedding = Embedding(pos_train_file,model,50,6).linear_average()
		neg_train_embedding = Embedding(neg_train_file,model,50,6).linear_average()
		pos_test_embedding = Embedding(pos_test_file,model,50,6).linear_average()
		neg_test_embedding = Embedding(neg_test_file,model,50,6).linear_average()
		pos_train_all = np.concatenate((pos_train, pos_train_embedding),axis=1)
		neg_train_all = np.concatenate((neg_train, neg_train_embedding),axis=1)
		pos_test_all = np.concatenate((pos_test, pos_test_embedding),axis=1)
		neg_test_all = np.concatenate((neg_test, neg_test_embedding),axis=1)
		classifier = Classifier(pos_train_all,neg_train_all,pos_test_all,neg_test_all)
		auc, auprg = classifier.train_gradient_boosting_model()
		cv_stats['auc'].append(auc)
		cv_stats['auprg'].append(auprg)
		print("Fold " + str(i) + ", AUC: " + str(cv_stats['auc'][i]) + ", auPRG: " + str(cv_stats['auprg'][i]))
	cv_auc = np.mean(cv_stats['auc'])
	cv_auprg = np.mean(cv_stats['auprg'])
	print("Cross validation - AUC: " + str(cv_auc) + ", auPRG: " + str(cv_auprg))

def use_alternate_allele():
	model = "/mnt/lab_data/kundaje/projects/snpbedding/sequence_embedding_asdhs_glove_k=7_window=12_size=256.model"
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/causal_snps_high_quality/alt_allele_test/"
	path, dirs, files = os.walk(splits).next()
	cv_stats = {'auc':[],'auprg':[]}
	for i in range(1):
		pos_ref = splits + "/causal_snps_1000bp_ref.fa"
		neg_ref = splits + "/ld_snps_1000bp_ref.fa"
		pos_alt = splits + "/causal_snps_1000bp_alt.fa"
		neg_alt = splits + "/ld_snps_1000bp_alt.fa"
		pos_ref_embedding = Embedding(pos_ref,model,256,7).linear_average()
		neg_ref_embedding = Embedding(neg_ref,model,256,7).linear_average()
		pos_alt_embedding = Embedding(pos_alt,model,256,7).linear_average()
		neg_alt_embedding = Embedding(neg_alt,model,256,7).linear_average()
		pos_delta_embedding = np.abs(pos_ref_embedding - pos_alt_embedding)
		neg_delta_embedding = np.abs(neg_ref_embedding - neg_alt_embedding)
		pos_embedding = np.concatenate((pos_ref_embedding, pos_alt_embedding),axis=1)
		neg_embedding = np.concatenate((neg_ref_embedding, neg_alt_embedding),axis=1)
		classifier = RandomClassifier(pos_embedding,neg_embedding)
		auc, auprg, preds = classifier.train_relu_model()
		cv_stats['auc'].append(auc)
		cv_stats['auprg'].append(auprg)
		print("Fold " + str(i) + ", AUC: " + str(cv_stats['auc'][i]) + ", auPRG: " + str(cv_stats['auprg'][i]))
	cv_auc = np.mean(cv_stats['auc'])
	cv_auprg = np.mean(cv_stats['auprg'])
	print("Cross validation - AUC: " + str(cv_auc) + ", auPRG: " + str(cv_auprg))

def train_regression():
	folder = "/mnt/lab_data/kundaje/projects/snpbedding/bedFiles/"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=128.model"
	train_sequences = folder + "allele-specific-DHS-1000bp-train.fa"
	test_sequences = folder + "allele-specific-DHS-1000bp-test.fa"
	train_embedding = Embedding(train_sequences,model,128,6).linear_average()
	test_embedding = Embedding(test_sequences,model,128,6).linear_average()
	train_targets = np.loadtxt(folder+"allele-specific-DHS-targets-train.txt",dtype="str").astype("float")
	test_targets = np.loadtxt(folder+"allele-specific-DHS-targets-test.txt",dtype="str").astype("float")
	regressor = Regressor(train_embedding, train_targets, test_embedding, test_targets)
	model = regressor.train_relu_model()
	#regressor.deeplift(model)

def run_single_fold():
	folder = "../../rmohan99/CRISPR-prediction/TSS_screen/"
	model = "../../rmohan99/CRISPR-prediction/TSS_screen/sequence_embedding_crko_tss_regions_glove_k=6_window=10_size=50.model"
	pos_train_file = folder + "pos_train.fa"
	neg_train_file = folder + "neg_train.fa"
	pos_test_file = folder + "pos_test.fa"
	neg_test_file = folder + "neg_test.fa"
	pos_train = Embedding(pos_train_file,model,50,6).linear_average()
	neg_train = Embedding(neg_train_file,model,50,6).linear_average()
	pos_test = Embedding(pos_test_file,model,50,6).linear_average()
	neg_test = Embedding(neg_test_file,model,50,6).linear_average()
	classifier = Classifier(pos_train,neg_train,pos_test,neg_test)
	auc, auprg, deeplift_auc, deeplift_auprg = classifier.train_relu_model()

import matplotlib.pyplot as plt
from tsne import bh_sne
def cluster_snps():
	snps_sequence = "/mnt/lab_data/kundaje/projects/snpbedding/snp_clustering/dbSNP_10K_1000bp.fa"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=128.model"
	snps_embedding = Embedding(snps_sequence,model,128,6).linear_average()
	#tsne_embedding = bh_sne(snps_embedding, perplexity=100)
	#plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1])
	#plt.savefig("snpbedding/tsne_embedding_noclusters.png")
	#print("Done with T-SNE")
	def hbdscan():
		clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
		cluster_labels = clusterer.fit_predict(snps_embedding)
		cluster_num = np.unique(cluster_labels)
		np.savetxt("cluster_labels.txt",cluster_labels,fmt="%i")
		for i in range(len(cluster_num)):
			num_points_in_cluster = len(np.where(cluster_labels == cluster_num[i])[0])
			print cluster_num[i], num_points_in_cluster
	def dbscan():
		dbscan = DBSCAN(eps=0.8)
		cluster_labels = dbscan.fit_predict(tsne_embedding)
		cluster_num = np.unique(cluster_labels)
		print cluster_num
		plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=cluster_labels)
		plt.savefig("snpbedding/tsne_embedding.png")
		np.savetxt("cluster_labels.txt",cluster_labels,fmt="%i")
	def birch():
		birch = Birch(branching_factor=5, n_clusters=None, threshold=0.62,compute_labels=True)
		cluster_labels = birch.fit_predict(snps_embedding)
		np.savetxt("cluster_labels.txt",cluster_labels,fmt="%i")
		cluster_num = np.unique(cluster_labels)
		print cluster_num
		plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=cluster_labels)
		plt.savefig("snpbedding/tsne_embedding.png")
	def kmeans():
		kmeans = KMeans(n_clusters=6,verbose=1,n_jobs=200,n_init=20)
		cluster_labels = kmeans.fit_predict(tsne_embedding)
		cluster_num = np.unique(cluster_labels)
		for i in range(len(cluster_num)):
			num_points_in_cluster = len(np.where(cluster_labels == cluster_num[i])[0])
			print cluster_num[i], num_points_in_cluster
		cluster_labels[cluster_labels == 5] = 2
		cluster_labels[cluster_labels == 2] = 2
		cluster_labels[cluster_labels == 3] = 2
		cluster_labels[cluster_labels == 4] = 2
		np.savetxt("cluster_labels.txt",cluster_labels,fmt="%i")
		plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=cluster_labels)
		plt.savefig("snpbedding/tsne_embedding.png")
	def spectral_clustering():
		spectral = SpectralClustering(n_clusters=3)
		cluster_labels = spectral.fit_predict(tsne_embedding)
		print np.unique(cluster_labels)
		print silhouette_score(snps_embedding, cluster_labels)
		np.savetxt("cluster_labels.txt",cluster_labels,fmt="%i")
		plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=cluster_labels)
		plt.savefig("snpbedding/tsne_embedding.png")
	def agglomerative_clustering():
		ac = AgglomerativeClustering(n_clusters=3)
		cluster_labels = ac.fit_predict(tsne_embedding)
		np.savetxt("cluster_labels.txt",cluster_labels,fmt="%i")
		plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=cluster_labels)
		plt.savefig("snpbedding/tsne_embedding.png")
	def affinityprop():
		af = AffinityPropagation(damping=0.9, verbose=True)
		cluster_labels = af.fit_predict(snps_embedding)
		cluster_num = np.unique(cluster_labels)
		np.savetxt("cluster_labels.txt",cluster_labels,fmt="%i")
		print len(np.unique(cluster_labels))
		#plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=cluster_labels)
		#plt.savefig("snpbedding/tsne_embedding.png")
	affinityprop()

def visualize_embedding():
	split = "/mnt/lab_data/kundaje/projects/snpbedding/causal_snps_high_quality/folds/split_15/"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=50.model"
	pos_train_file = split + "/pos_train.fa"
	neg_train_file = split + "/neg_train.fa"
	pos_train_embedding = Embedding(pos_train_file,model,50,6).exponential_average(100)
	neg_train_embedding = Embedding(neg_train_file,model,50,6).exponential_average(100)
	data = np.concatenate((pos_train_embedding, neg_train_embedding))
	pos_labels = np.ones((len(pos_train_embedding)))
	neg_labels = np.zeros((len(neg_train_embedding)))
	labels = np.concatenate((pos_labels, neg_labels))
	tsne_embedding = bh_sne(data, perplexity=5)
	plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=labels)
	plt.savefig("snpbedding/causal_snps_tsne.png")

from annotate_variants import *
def annotate_clusters():
	cluster_labels = np.loadtxt("cluster_labels.txt")
	regions = np.loadtxt("snp_clustering/dbSNP_50K.bed",dtype="str")
	exon_enrichment = np.zeros((len(cluster_labels),3))
	intron_enrichment = np.zeros((len(cluster_labels),3))
	dnase_enrichment = np.zeros((len(cluster_labels),3))
	states_enrichment = np.zeros((len(cluster_labels),3))
	for i in range(len(cluster_labels)):
		try:
			chr_num = int(regions[i][0].split("chr")[1])
			location = int(regions[i][1])
		except:
			continue
		exon_overlap = float(overlaps_exon(chr_num,location))
		intron_overlap = float(overlaps_intron(chr_num,location))
		dnase_overlap = float(overlaps_DNase(chr_num,location))
		states_overlap = float(overlaps_states(chr_num,location,["14_TssBiv","15_EnhBiv"]))

		exon_enrichment[i, cluster_labels[i]] = exon_overlap
		intron_enrichment[i, cluster_labels[i]] = intron_overlap
		dnase_enrichment[i, cluster_labels[i]] = dnase_overlap
		states_enrichment[i, cluster_labels[i]] = states_overlap

	cluster_freqs = [len(np.where(cluster_labels==0)[0]),len(np.where(cluster_labels==1)[0]),len(np.where(cluster_labels==2)[0])]
	exon_frequencies = np.divide(np.sum(exon_enrichment,axis=0), np.array(cluster_freqs))
	intron_frequencies = np.divide(np.sum(intron_enrichment,axis=0), np.array(cluster_freqs))
	dnase_frequencies = np.divide(np.sum(dnase_enrichment,axis=0), np.array(cluster_freqs))
	states_frequencies = np.divide(np.sum(states_enrichment,axis=0), np.array(cluster_freqs))

	print exon_frequencies[0], exon_frequencies[1], exon_frequencies[2]
	print intron_frequencies[0], intron_frequencies[1], intron_frequencies[2]
	print dnase_frequencies[0], dnase_frequencies[1], dnase_frequencies[2]
	print states_frequencies[0], states_frequencies[1], states_frequencies[2]

def train_celltype_specific_embedding():
	dnase_files = glob.glob("/mnt/lab_data/kundaje/projects/snpbedding/dnase_peaks/new_dnase_fastas/*.fa")
	window_size = 10
	embedding_size = 20
	kmer_size = 6
	for dnase_file in dnase_files:
		out_file = (dnase_file.split("/")[-1].split(".")[0])+"_glove_sequence_embedding_window="+str(window_size)+"_size="+str(embedding_size)+"_k="+str(kmer_size)+".model"
		glove = TrainEmbedding(dnase_file, window_size, embedding_size, kmer_size)
		glove.train_embedding("dnase_peaks/celltype_specific_models/"+out_file)
		print("Saved " + str(out_file))

def train_embedding_model():
	sequence_file = "all_conserved_regions_high_quality.fa"
	window_size = 12
	embedding_size = 256
	kmer_size = 7
	glove = TrainEmbedding(sequence_file, window_size, embedding_size, kmer_size)
	glove.train_embedding("sequence_embedding_GERP_conserved_glove_k=7_window=12_size=256.model")

def train_word2vec_model():
	sequence_file = "allele_specific_DNase_SNPs_150bp.fa"
	window_size = 10
	embedding_size = 2
	kmer_size = 7

	model = gensim.models.Word2Vec(kmers, size=512, alpha=0.025, sg=1, window=12, min_count=50, workers=16, iter=1)
	model.train(kmers)
	model.save("sequence_embedding_dbSNP_skipgram_context=150bp_k=6_window=12_size=512.model")

def run_cross_validation_multicelltypes():
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/qtl_challenge_em_vars/folds/"
	path, dirs, files = os.walk(splits).next()
	cv_stats = {'auc':[],'auprg':[],'deeplift_auc':[],'deeplift_auprg':[]}
	num_examples = []
	embedding_size = 75
	kmer_size = 6
	for i in range(len(dirs)):
		pos_train_file = splits + dirs[i] + "/pos_train.fa"
		neg_train_file = splits + dirs[i] + "/neg_train.fa"
		pos_test_file = splits + dirs[i] + "/pos_test.fa"
		neg_test_file = splits + dirs[i] + "/neg_test.fa"
		pos_train = CellTypeEmbedding(pos_train_file,embedding_size,kmer_size).compute_embedding()
		neg_train = CellTypeEmbedding(neg_train_file,embedding_size,kmer_size).compute_embedding()
		pos_test = CellTypeEmbedding(pos_test_file,embedding_size,kmer_size).compute_embedding()
		neg_test = CellTypeEmbedding(neg_test_file,embedding_size,kmer_size).compute_embedding()
		num_test_examples = len(pos_test)+len(neg_test)
		num_examples.append(num_test_examples)
		classifier = Classifier(pos_train,neg_train,pos_test,neg_test)
		auc, auprg = classifier.train_svm_model()
		cv_stats['auc'].append(auc)
		cv_stats['auprg'].append(auprg)
		print("Fold " + str(i) + ", AUC: " + str(cv_stats['auc'][i]) + ", auPRG: " + str(cv_stats['auprg'][i]))
	cv_auc = np.average(cv_stats['auc'],weights=np.array(num_examples))
	cv_auprg = np.average(cv_stats['auprg'],weights=np.array(num_examples))
	print("Cross validation - AUC: " + str(cv_auc) + ", auPRG: " + str(cv_auprg))

def train_dnase_embedding():
	signal_files = ["allele_specific_DNase_SNPs.25.dnase.f32","allele_specific_DNase_SNPs.100.dnase.f32",
					"allele_specific_DNase_SNPs.500.dnase.f32","allele_specific_DNase_SNPs.2500.dnase.f32",
					"allele_specific_DNase_SNPs.5000.dnase.f32","allele_specific_DNase_SNPs.10000.dnase.f32"]
	num_samples = 100000
	num_features = 51
	embedder = RoadmapEmbedding(signal_files,num_samples,num_features)
	features = embedder.load_features()
	features = embedder.rank_normalize(features)
	embedder.train_sparse_autoencoder(features)

def train_roadmap_embedding():
	signal_files = ["labels.500k.25.noE117.noE118.f32","labels.500k.100.noE117.noE118.f32",
					"labels.500k.500.noE117.noE118.f32","labels.500k.2500.noE117.noE118.f32",
					"labels.500k.5000.noE117.noE118.f32","labels.500k.10000.noE117.noE118.f32"]
	num_samples = 100000
	num_features = 1008
	embedder = RoadmapEmbedding(signal_files,num_samples,num_features)
	features = embedder.load_features()
	features = embedder.rank_normalize(features)
	embedder.train_sparse_autoencoder(features)

def evaluate_dnase_embedding():
	pos_signal_files = ["dsQTL_deltaSVM_task/dnase_files/pos_set_dnase.25.dnase.f32","dsQTL_deltaSVM_task/dnase_files/pos_set_dnase.100.dnase.f32","dsQTL_deltaSVM_task/dnase_files/pos_set_dnase.500.dnase.f32",
						"dsQTL_deltaSVM_task/dnase_files/pos_set_dnase.2500.dnase.f32","dsQTL_deltaSVM_task/dnase_files/pos_set_dnase.5000.dnase.f32","dsQTL_deltaSVM_task/dnase_files/pos_set_dnase.10000.dnase.f32"]
	neg_signal_files = ["dsQTL_deltaSVM_task/dnase_files/neg_set_dnase.25.dnase.f32","dsQTL_deltaSVM_task/dnase_files/neg_set_dnase.100.dnase.f32","dsQTL_deltaSVM_task/dnase_files/neg_set_dnase.500.dnase.f32",
						"dsQTL_deltaSVM_task/dnase_files/neg_set_dnase.2500.dnase.f32","dsQTL_deltaSVM_task/dnase_files/neg_set_dnase.5000.dnase.f32","dsQTL_deltaSVM_task/dnase_files/neg_set_dnase.10000.dnase.f32"]
	
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=50.model"
	pos_seq_file = "/mnt/lab_data/kundaje/projects/snpbedding/dsQTL_deltaSVM_task/pos_set_150bp.fa"
	neg_seq_file = "/mnt/lab_data/kundaje/projects/snpbedding/dsQTL_deltaSVM_task/neg_set_150bp.fa"
	pos_seq_embedding = Embedding(pos_seq_file,model,50,6).linear_average()
	neg_seq_embedding = Embedding(neg_seq_file,model,50,6).linear_average()

	num_pos_examples = 574
	num_neg_examples = 27735

	pos_embedder = RoadmapEmbedding(pos_signal_files,num_pos_examples,51)
	pos_features = pos_embedder.load_features()
	pos_features = pos_embedder.rank_normalize(pos_features)

	neg_embedder = RoadmapEmbedding(neg_signal_files,num_neg_examples,51)
	neg_features = neg_embedder.load_features()
	neg_features = neg_embedder.rank_normalize(neg_features)

	autoencoder = cPickle.load(open("dnase_sparse_autoencoder.pkl"))
	pos_embedding = autoencoder.predict(pos_features)
	neg_embedding = autoencoder.predict(neg_features)

	pos_combined_embedding = np.concatenate((pos_features, pos_seq_embedding),axis=1)
	neg_combined_embedding = np.concatenate((neg_features, neg_seq_embedding),axis=1)

	classifier = RandomClassifier(pos_combined_embedding,neg_combined_embedding)
	classifier.run_kfold_cross_val()

def similarity_query(model, word_vectors, word_vec, number):
	dst = (np.dot(model['word_vectors'], word_vec)/ np.linalg.norm(model['word_vectors'], axis=1)/ np.linalg.norm(word_vec))
	word_ids = np.argsort(-dst)
	return [(model['inverse_dictionary'][x], dst[x]) for x in word_ids[:number] if x in model['inverse_dictionary']]

def most_similar(model, word, number=5):
	try:
		word_idx = model['dictionary'][word]
	except KeyError:
		raise Exception('Word not in dictionary')
	return similarity_query(model, model['word_vectors'], model['word_vectors'][word_idx], number)[1:]

from scipy import spatial
def motif_similarity(model, motif1, motif2):
	word1_idx = model['dictionary'][motif1]
	word2_idx = model['dictionary'][motif2]
	word1_embedding = model['word_vectors'][word1_idx]
	word2_embedding = model['word_vectors'][word2_idx]
	cosine_similarity = 1 - spatial.distance.cosine(word1_embedding, word2_embedding)
	print cosine_similarity

def cluster_kmers():
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=75.model"
	clusters = Clustering(model)
	clusters.create_clusters()
	clusters.print_clusters(50)

def find_similar_kmers():
	model_file = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=75.model"
	model = pickle.load(open(model_file,"rb"))
	fox_motif = "AATATT"
	gata_motif = "GATAAG"
	myc_motif = "CACGTG"
	zbtb33 = "TCCTGC"
	tcf7l2 = "CTTTGA"
	tata = "ATAAAA"
	nfkb = "ACTTCC"
	motifs = most_similar(model, nfkb, number=10)
	print motifs

def use_random_split():
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/validated_snps/"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/sequence_embedding_asdhs_glove_k=7_window=12_size=256.model"
	pos_file = splits + "oreganno_combined_all_causal_snps_1000bp.fa"
	neg_file = splits + "matched_neg_examples_merged.fa"
	pos_embedding = Embedding(pos_file,model,256,7).linear_average()
	neg_embedding = Embedding(neg_file,model,256,7).linear_average()
	classifier = RandomClassifier(pos_embedding,neg_embedding)
	classifier.run_kfold_cross_val()

def use_random_split_dnase():
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/dsQTL_deltaSVM_task/"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_chromatin_E116_snps_glove_k=6_window=10_size=512.model"
	pos_file = splits + "pos_set_150bp.fa"
	neg_file = splits + "neg_set_150bp.fa"
	pos_seq_embedding = Embedding(pos_file,model,512,6).linear_average()
	neg_seq_embedding = Embedding(neg_file,model,512,6).linear_average()
	pos_dnase_embedding = DeepLIFT("sequence_embedding_relu_dnase_model.yaml","sequence_embedding_relu_dnase_model.h5",pos_seq_embedding).compute_importance_scores()
	neg_dnase_embedding = DeepLIFT("sequence_embedding_relu_dnase_model.yaml","sequence_embedding_relu_dnase_model.h5",neg_seq_embedding).compute_importance_scores()
	pos_combined = np.concatenate((pos_seq_embedding, pos_dnase_embedding),axis=1)
	neg_combined = np.concatenate((neg_seq_embedding, neg_dnase_embedding),axis=1)
	classifier = RandomClassifier(pos_combined,neg_combined)
	classifier.run_kfold_cross_val()

def use_random_split_alt_allele():
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/causal_snps_high_quality/alt_allele_test/"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/sequence_embedding_asdhs_glove_k=7_window=12_size=5.model"
	
	pos_file_ref = splits + "causal_snps_19bp_ref.fa"
	pos_file_alt = splits + "causal_snps_19bp_alt.fa"
	neg_file_ref = splits + "ld_snps_19bp_ref.fa"
	neg_file_alt = splits + "ld_snps_19bp_alt.fa"

	pos_ref = Embedding(pos_file_ref,model,5,7).linear_average()
	neg_ref = Embedding(neg_file_ref,model,5,7).linear_average()
	pos_alt = Embedding(pos_file_alt,model,5,7).linear_average()
	neg_alt = Embedding(neg_file_alt,model,5,7).linear_average()

	print pos_ref[0]
	print pos_alt[0]

	#pos_ref = DeepLIFT("sequence_embedding_relu_dnase_model.yaml","sequence_embedding_relu_dnase_model.h5",pos_ref).compute_importance_scores()
	#neg_ref = DeepLIFT("sequence_embedding_relu_dnase_model.yaml","sequence_embedding_relu_dnase_model.h5",neg_ref).compute_importance_scores()
	#pos_alt = DeepLIFT("sequence_embedding_relu_dnase_model.yaml","sequence_embedding_relu_dnase_model.h5",pos_alt).compute_importance_scores()
	#neg_alt = DeepLIFT("sequence_embedding_relu_dnase_model.yaml","sequence_embedding_relu_dnase_model.h5",neg_alt).compute_importance_scores()

	pos_combined = np.concatenate((pos_ref, pos_alt),axis=1)
	neg_combined = np.concatenate((neg_ref, neg_alt),axis=1)
	classifier = RandomClassifier(pos_combined,neg_combined)
	classifier.run_kfold_cross_val()

def use_predefined_split():
	splits = "/mnt/lab_data/kundaje/projects/snpbedding/validated_snps/"
	embedding_model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=75.model"
	
	pos_train_file = splits + "pos_train.fa"
	neg_train_file = splits + "neg_train.fa"
	pos_test_file = splits + "pos_test.fa"
	neg_test_file = splits + "neg_test.fa"

	pos_train = Embedding(pos_train_file,embedding_model,75,6).linear_average()
	neg_train = Embedding(neg_train_file,embedding_model,75,6).linear_average()
	pos_test = Embedding(pos_test_file,embedding_model,75,6).linear_average()
	neg_test = Embedding(neg_test_file,embedding_model,75,6).linear_average()

	x_train = np.concatenate((pos_train, neg_train))
	y_train = np.concatenate((np.ones((len(pos_train))),np.zeros((len(neg_train)))))
	x_test = np.concatenate((pos_test, neg_test))
	y_test = np.concatenate((np.ones((len(pos_test))),np.zeros((len(neg_test)))))

	x_train,y_train = shuffle(x_train,y_train)
	x_test,y_test = shuffle(x_test,y_test)
	classifier = RandomClassifier(pos_train, neg_train)
	auc, auprg, preds, model = classifier.train_relu_model(x_train, x_test, y_train, y_test)
	classifier.kmer_importance(embedding_model, model, x_test, y_test, preds, "validated_causal_snps_kmer_importance.txt")

from keras.models import model_from_json, model_from_yaml
def validate_tert():
	model = Sequential()
	model.add(Dense(1024, activation="relu", input_shape=(50,)))
	model.add(Dense(512,activation="relu"))
	model.add(Dense(256,activation="relu"))
	model.add(Dense(164, activation="sigmoid"))
	model.load_weights("dnase_predictor/dnase_predictor_dnn.h5")

	embedding_model = "/mnt/lab_data/kundaje/projects/snpbedding/sequence-embeddings/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=50.model"
	TERT_embeddings = Embedding("dnase_predictor/HBB_regulatory_regions.fa",embedding_model,50,6).linear_average()
	TERT_predictions = model.predict(TERT_embeddings)
	task_idx = np.argmax(np.mean(TERT_predictions))
	
	deeplift_model = kc.convert_sequential_model(model, mxts_mode=MxtsMode.DeepLIFT)
	target_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)
	target_contribs = target_contribs_func(task_idx=task_idx, input_data_list=[TERT_embeddings],batch_size=1, progress_update=10000)
	target_contribs = np.mean(np.array(target_contribs),axis=0)

	embedding_model = pickle.load(open(embedding_model,"rb"))
	kmers = embedding_model['dictionary'].keys()
	kmer_idxs = embedding_model['dictionary'].values()
	kmer_vectors = embedding_model['word_vectors']
	contribs = kmer_vectors.dot(target_contribs.T)
	ranked_contribs = np.argsort(contribs.flatten())
	out_file = open("hbb_kmer_importances.txt","wb")

	for i in range(len(ranked_contribs)-1,-1,-1):
		kmer_idx = ranked_contribs[i]
		score = contribs[kmer_idx]
		kmer = kmers[np.where(kmer_idxs==kmer_idx)[0]]
		out_file.write(kmer + "\t" + str(score) + "\n")
	print("Finished kmer importance computation")
	out_file.close()

def compute_kmers(seq, kmer_size):
	it = iter(seq)
	win = [it.next() for cnt in xrange(kmer_size)]
	yield "".join(win)
	for e in it:
		win[:-1] = win[1:]
		win[-1] = e
		yield "".join(win)

def sequence_processor(seq,k):
    kmers=[]
    for i in range(0,len(seq),k):
        kmers.append(seq[i:i+k])
    return kmers

def chunks(iterable,size):
    it = iter(iterable.upper())
    chunk = "".join(list(itertools.islice(it,size)))
    while chunk:
        yield chunk
        chunk = "".join(list(itertools.islice(it,size)))

def chunks_to_list(iterable, size):
    return list(chunks(iterable,size))

def compute_nonoverlapping_kmers(fasta, k):
    all_kmers = Parallel(n_jobs=16)(delayed(sequence_processor)(seq,k) for seq in fasta)
    return all_kmers

def per_base_scoring(kmer_importance_file, fasta_file, out_file):
	kmer_importances = np.loadtxt(kmer_importance_file,dtype="str")
	kmer_to_importance_map = dict(zip(kmer_importances[:,0], kmer_importances[:,1]))
	fasta = np.loadtxt(fasta_file,dtype="str")
	all_scores = np.zeros((len(fasta),len(fasta[0])/7))
	sequence_num = 0
	for sequence in fasta:
		kmers = sequence_processor(sequence.upper(), 7)
		edge = [sequence.upper()[len(sequence)-8:]]*7
		kmers = kmers.pop()
		#assert len(kmers) == len(sequence)
		scores = itemgetter(*kmers)(kmer_to_importance_map)    
		try:
			all_scores[sequence_num,:] = scores
		except:
			continue
		sequence_num = sequence_num + 1
	np.save(out_file,all_scores)

def score_sequences():
	kmer_importance_file = "validated_causal_snps_kmer_importance.txt"
	fasta_file = "validated_snps/all_snps.fa"
	out_file = "validated_causal_snps_deeplift_scores.npy"
	per_base_scoring(kmer_importance_file, fasta_file, out_file)

from scipy.signal import savgol_filter

def plot_deeplift_scores():
	scores = np.load("validated_causal_snps_deeplift_scores.npy").mean(axis=0)
	scores = savgol_filter(np.abs(scores), 101, 3)
	
	plt.plot(scores, label="deepLIFT kmer scores") 
	plt.grid(True)
	plt.title("deepLIFT scores from embedding on causal variants")
	plt.legend()
	plt.savefig("kmer2vec/causal_snps_deeplift.png")


def plot_with_chipseq():
	scores = np.load("CRISPRi_Enh_Element_correctly_predicted_pos_scores.npy")
	chipseq = np.load("MAX_GM12878_ChIPSeq_postest.npy")[:,0,0,:].mean(axis=0)

	scaler = StandardScaler()
	chipseq = scaler.fit_transform(chipseq)
	scores = savgol_filter(scores[0,:], 61, 3)
	scores = MinMaxScaler(feature_range=(-2,4)).fit_transform(scores)

	#plt.plot(chipseq, label="MAX TF ChIP-Seq signal")
	plt.plot(scores, label="deepLIFT kmer scores") 
	plt.grid(True)
	plt.title("Comparing deepLIFT scores from embedding to ChIP-Seq signal")
	plt.legend()
	plt.savefig("snpbedding/crispr_enh_elements.png")

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import SpectralEmbedding, Isomap

def tsne_embeddings():
	pos_file = "/mnt/lab_data/kundaje/projects/snpbedding/snp_clustering/pos_set_504bp.fa"
	neg_file = "/mnt/lab_data/kundaje/projects/snpbedding/snp_clustering/neg_set_504bp.fa"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=75.model"
	pos_embedding = Embedding(pos_file,model,75,6).linear_average()
	neg_embedding = Embedding(neg_file,model,75,6).linear_average()
	pos_labels = np.ones((len(pos_embedding)))
	neg_labels = np.zeros((len(neg_embedding)))
	data = np.concatenate((pos_embedding,neg_embedding))
	labels = np.concatenate((pos_labels,neg_labels))
	data_2d = SpectralEmbedding(n_components=2).fit_transform(data)
	plt.scatter(data_2d[np.where(labels == 1),0],data_2d[np.where(labels == 1),1],c="green",s=35,label="ORegAnno SNPs")
	plt.scatter(data_2d[np.where(labels == 0),0],data_2d[np.where(labels == 0),1],c="red",s=35,label="Random LD SNPs")
	plt.grid(True)
	plt.title("Spectral Dimensionality Reduction on Embedding")
	plt.legend()
	plt.savefig("snpbedding/causal_snps_spectral.png")

from scipy.stats import mannwhitneyu
def enhancer_clustering():
	pos = "enhancers/regions_enh_E034.fa"
	neg = "enhancers/regions_enh_E062.fa"
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=128.model"
	pos_embedding = Embedding(pos,model,128,6).linear_average()
	neg_embedding = Embedding(neg,model,128,6).linear_average()
	pos_embedding_sum = np.sum(pos_embedding, axis=1)
	neg_embedding_sum = np.sum(neg_embedding, axis=1)
	print mannwhitneyu(pos_embedding_sum, neg_embedding_sum)

	pos_labels = np.ones((len(pos_embedding)))
	neg_labels = np.zeros((len(neg_embedding)))
	labels = np.concatenate((pos_labels,neg_labels))
	data = np.concatenate((pos_embedding,neg_embedding))
	data_2d = PCA().fit_transform(data)
	plt.scatter(data_2d[np.where(labels == 1),0],data_2d[np.where(labels == 1),1],c="green",s=35,label="E071 enhancers")
	plt.scatter(data_2d[np.where(labels == 0),0],data_2d[np.where(labels == 0),1],c="red",s=35,label="E062 enhancers")
	plt.grid(True)
	plt.title("Spectral Dimensionality Reduction on Enhancer Embeddings")
	plt.legend()
	plt.savefig("snpbedding/gm12878_vs_k562_enhancers.png")


from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

def predict_enh_clusters():
	model = "/mnt/lab_data/kundaje/projects/snpbedding/models/sequence_embedding_dnase_snps_glove_k=6_window=10_size=128.model"
	enh_clusters = Embedding("all_enhancer_clusters.fa",model,128,6).linear_average()
	enh_cluster_labels = np.loadtxt("all_enhancer_cluster_labels.bed")
	x_train,x_test,y_train,y_test = train_test_split(enh_clusters,enh_cluster_labels,test_size=0.2)
	classifier = OneVsRestClassifier(LinearSVC(C=10))
	classifier.fit(x_train,y_train)
	y_test = label_binarize(y_test,classes=np.unique(y_test))
	preds = classifier.predict_proba(x_test)
	print roc_auc_score(y_test, preds)

def compute_cluster_enrichments():
	cluster_labels = np.loadtxt("cluster_labels.txt")
	tf_binding_overlaps = np.loadtxt("snp_clustering/dbSNP_10K_tss_overlaps.bed")
	label_nums = np.unique(cluster_labels)
	for cluster_num in label_nums:
		idxs = np.where(cluster_labels == cluster_num)[0]
		enrichment = np.mean(tf_binding_overlaps[idxs]) / float(np.mean(tf_binding_overlaps))
		print cluster_num, enrichment

import pyximport; pyximport.install()
from one_hot_encode import one_hot_encode

def score_basset_model():
	dnase_model_yaml = "dnase_predictor/dnase_predictor_basset_model.yaml"
	dnase_model_weights = "dnase_predictor/dnase_predictor_basset_model.h5"
	model = model_from_yaml(open(dnase_model_yaml))
	model.load_weights(dnase_model_weights)

	pos_sequences = one_hot_encode(np.loadtxt("low_quality_variant_task/matched_pos_examples_600bp.fa",dtype="str"))
	neg_sequences = one_hot_encode(np.loadtxt("low_quality_variant_task/matched_neg_examples_600bp.fa",dtype="str"))
	combined_sequences = np.concatenate((pos_sequences, neg_sequences))
	labels = np.concatenate((np.ones((len(pos_sequences))), np.zeros((len(neg_sequences)))))

	basset_preds = model.predict(combined_sequences)
	sum_preds = np.max(basset_preds,axis=1)
	print roc_auc_score(labels, sum_preds)

def train_igr():
	x_train = np.load("x_train_igr.npy")
	y_train = np.load("y_train_igr.npy")
	x_test = np.load("x_test_igr.npy")
	y_test = np.load("y_test_igr.npy")
	scaler=StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	print x_train.shape

	classifier = RandomClassifier(x_train, x_test)
	auc, auprg, preds, model = classifier.train_relu_model(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
	#run_cross_validation()
	train_igr()
	#plot_deeplift_scores()
	#eigen = EIGEN()
	#eigen.evaluate_from_files("/mnt/lab_data/kundaje/projects/snpbedding/GWAS_EIGEN/GWAScatalog_EIGEN_regulatory.txt","/mnt/lab_data/kundaje/projects/snpbedding/GWAS_EIGEN/GWAStag_EIGEN_regulatory.txt")
