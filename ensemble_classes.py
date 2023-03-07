import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RfRegressor:
    def __init__(self, L, U, features):
        self.L = L
        self.U = U
        self.features = features
        self.rf = RandomForestRegressor(n_estimators=U.shape[1])
        self.rf.fit(self.L.loc[:, self.features].values, self.L.loc[:, 'y'].values)
        
    def update_model(self, ind):  
        obs_df = self.U.loc[[ind], :]
        self.L = pd.concat([self.L, obs_df])
        self.U = self.U.drop(ind, axis=0)
        self.rf.fit(self.L.loc[:, self.features].values, self.L.loc[:, 'y'].values)
        
    def predict_on_L(self):
        return self.rf.predict(self.L.loc[:, self.features].values)
    
    def predict_on_U(self):
        return self.rf.predict(self.U.loc[:, self.features].values)
    
    def get_mse_on_L(self):
        y_hat = self.predict_on_L()
        y_true = self.L.loc[:, 'y'].values
        
        return np.sum(np.subtract(y_hat, y_true)**2)     
        
    def get_model_uncertainty(self, ind):
        x = self.U.loc[[ind], self.features].values
        predictions = []
        
        for tree in self.rf.estimators_:
            predictions.append(tree.predict(x))
        
        return np.var(predictions)
    
    
class Ensemble:
    def __init__(self, dataframes, seed=0, init_size=5):
        self.models = []
        ind = dataframes[0].index
        self.rng = np.random.default_rng(seed)
        
        init_inds = []
        while len(init_inds) < init_size:
            init_inds = np.unique(self.rng.integers(0, len(ind), size=init_size))
            
        self.init_inds = ind[init_inds]
        
        for df in dataframes:
            L = df.loc[self.init_inds, :]
            df = df.drop(self.init_inds, axis=0)
            self.models.append(RfRegressor(L, df, df.columns[:-1]))
            
        self.weights = [1.0/len(self.models)]*len(self.models)
               
    def predict_and_update_models(self, ind):
        #First, update each model to include ind in L
        for i in range(len(self.models)):
            self.models[i].update_model(ind)
        
        mse = []
        #Gather predictions from each model on L
        predictions = np.zeros((len(self.models), self.models[0].L.shape[0]))
        for i, m in enumerate(self.models):
            predictions[i] = self.weights[i]*m.predict_on_L()
            mse.append(1/m.get_mse_on_L())
            
        #Update the weights as fraction of the total MSE
        total_mse = np.sum(mse)
        
        for i in range(len(self.weights)):
            self.weights[i] = mse[i]/total_mse
            
        #Compute the final y_pred values as the sum of predictions * weight
        return np.sum(predictions, axis=0) 
            
    def sample_uncertainty(self):
        uc_mat = np.zeros((len(self.models), self.models[0].U.shape[0]))
        for i, m in enumerate(self.models):
            for j, ind in enumerate(m.U.index):
                uc_mat[i, j] = m.get_model_uncertainty(ind)
                       
        best_ind = np.argmax(np.mean(uc_mat, axis=0))
        
        return self.models[0].U.index[best_ind]
    
    def random_sample(self):
        ind = self.rng.integers(0, len(self.models[0].U.index))
        return self.models[0].U.index[ind]
    
    def iterate(self, mode='uc'):
        
        if mode == 'uc':
            ind = self.sample_uncertainty()
        else:
            ind = self.random_sample()
        predictions = self.predict_and_update_models(ind)
        
        return predictions, ind, self.models[0].U.empty, self.models[0].U.shape
    
    def predict_unobserved(self):
        predictions = np.zeros((len(self.models), self.models[0].U.shape[0]))
        for i, m in enumerate(self.models):
            predictions[i] = self.weights[i]*m.predict_on_U()
        
        return np.sum(predictions, axis=0), self.models[0].U.index