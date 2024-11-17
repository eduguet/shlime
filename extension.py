"""
The experiment MAIN for COMPAS.
 * Run the file and the COMPAS experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   both LIME and SHAP explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
 	    1: [('length_of_stay', 0.002592352559948153), ('unrelated_column_one', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to SHAP yielding only 0 shapley values 
"""
import warnings

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from copy import deepcopy

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_compas_data(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
features = [c for c in X]

race_indc = features.index('race')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

print (features)
###
## The models f and psi for COMPAS.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

# the display model with two unrelated features
# class innocuous_model_psi_two:
# 	def predict_proba(self, X):
# 		A = np.where(X[:,unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
# 		B = np.where(X[:,unrelated_indcs1] > 0, params.positive_outcome, params.negative_outcome)
# 		preds = np.logical_xor(A, B).astype(int)
# 		return one_hot_encode(preds)
#
##
###

def experiment_main():
	"""
	Run through experiments for LIME/SHAP on compas using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
	ss = StandardScaler().fit(xtrain)
	xtrain = ss.transform(xtrain)
	xtest = ss.transform(xtest)

    # ------------------  SHLIME EXPERIEMENTS SETUP ------------------ #
	
    print ('---------------------')
    print ("Beginning SHLIME COMPAS Experiments....")
	print ('---------------------')

    # Generate LIME and SHAP perturbations seperately

    # Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'), features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], feature_names=features, perturbation_multiplier=30)
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=adv_lime.get_column_names(), categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'),features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], discretize_continuous=False)
                                               
	lime_explanations = []
	for i in range(xtest.shape[0]):
		lime_explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

    # Train adversarial model for SHAP 
    background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features)
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	shap_explanations = adv_kerenel_explainer.shap_values(xtest)

    # TODO: Combine explanations by multiplying them?


    # TODO: Display results
    

if __name__ == "__main__":
	experiment_main()

