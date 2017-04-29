import pandas as pd 
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt 
import sklearn 
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import sys
import pickle
class KFold_Regression:
	def __init__(self, name, data_path, pred_length):
		self.name = name
		self.pred_length = pred_length
		# Read in csv file store as static data_frame (do not alter this dataframe)
		self.data_frame = pd.read_csv(data_path)
		# Will convert targets to log diff
		self.targets = None
		# regressor_models will be (key,iterable) pair. Key=num(regressors), iterable=list of combos of regressors
		self.regressor_models = {}

		self.pred_l = []
		for i in range(1, self.pred_length+1):
			self.pred_l.append(i)
		
		self.columns = self.pred_l + ['intercept'] + list(self.data_frame.columns.values)[2:]
		
		self.results = pd.DataFrame(columns=self.columns)

	def generate_combos(self):
		# grab regressor vals
		header = list(self.data_frame.columns.values)[2:]
		# for all length combos
		for i in range(1, len(header)+1):
			regressor_combos = itertools.combinations(header, r=i)
			#hash length combo and iterable
			self.regressor_models[i] = regressor_combos

	def get_data(self, regressors):
		#collect specific columns (leave out last because of log diff)
		sub_frame = self.data_frame[regressors][:-1]

		#add targets to specific data_frame returned
		sub_frame['Target'] = self.targets

		sub_frame_clean = self.clean(sub_frame)

		return sub_frame_clean

	def get_targets(self):
		log_gdp = []
		gdp = self.data_frame['Iceland']
		# collect time now (t) and time next (t_1) take log diff
		for t, t_1 in zip(gdp, gdp[1:]):
			log_gdp.append(np.log(t_1) - np.log(t))

		#convert to numpy array and set to targets (only have to do this once)
		self.targets = np.reshape(log_gdp, (len(log_gdp), 1))

	def clean(self, data):
		return data.dropna()

	def regress(self, data, normalize=False, njobs=1):
		#collect targets from data
		target = data['Target']
		#print(type(target))

		#remove targets from regressors
		X = data.drop('Target', axis=1)

		#create regression object
		lm = LinearRegression()

		#calculate scores with cross validation kfold=k
		lm.fit(X, target)

		return lm, X

	def mass_regress(self):
		total = np.power(2, len(self.columns) - self.pred_length - 1)
		print("total runs: " + str(total))
		#for each length type combination
		run=1
		for key, values in self.regressor_models.items():
			#for each combination 
			for each_model in values:
				#get data
				each_model = list(each_model)

				data = self.get_data(each_model)
				#get score

				lm, X= self.regress(data)


				prediction = lm.predict(X)[:self.pred_length]


				intercept = [lm.intercept_]

				coefficients = lm.coef_.tolist()

				columns = self.pred_l + ['intercept'] + each_model

				results = prediction.tolist() + intercept

				results = results + coefficients

				print(results)

				#add data for model to epsilon matrix
				#self.eps_m.append(scores.tolist())

				model_df = pd.DataFrame([results], columns=columns)
				#add to results table
				self.results = pd.concat([self.results,model_df])

				progress = (run/total)*100
				sys.stdout.write("Progress: {}% \r".format(progress))
				sys.stdout.flush()
				run = run + 1

		self.results.to_csv('data/'+str(self.name)+'.csv', columns=self.columns)

		#return self.eps_m



	def get_weights(self, array_like):
		e = self.get_matrix(array_like)

		e_t = e.transpose()

		S = e_t * e

		return self.minimize(S)



	def minimize(self, matrix):
		S_i = matrix.getI()
	
		m = matrix.shape[1]
		
		one_m = self.get_matrix(np.ones(m))

		return (S_i * one_m) / (one_m.transpose() * S_i * one_m)




	def get_matrix(self, array_like):
		return np.matrix(array_like).transpose()







		
		



def main():
	data_frame = KFold_Regression('results', 'data/data.csv', 8)
	
	data_frame.generate_combos()
	
	data_frame.get_targets()

	eps = data_frame.mass_regress()
'''
	pickle.dump( eps, open( "data/results/8_fold.pkl", "wb" ) )

	eps = pickle.load( open( "data/results/8_fold.pkl", "rb" ) )

	weight_v = data_frame.get_weights(eps)


	norm = [float(i)/sum(weight_v) for i in weight_v]

	norm_w = list(np.array(norm).reshape(-1,))

	print(norm_w)

'''
	