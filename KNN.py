import numpy as np
import warnings
warnings.filterwarnings('ignore')

class KNN():
    def __init__(self, k, metric):
        self.k = k
        self.metric = metric
        if(self.metric == 'cosine'):
            self.order = -1
        else:
            self.order = 1
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    def distance_matrix(self, A, B):
        if(self.metric == 'eucledian'):
            A_norm = np.square(np.linalg.norm(A, axis = 1)).reshape(A.shape[0],1) * np.ones(shape=(1,B.shape[0]))
            B_norm = np.square(np.linalg.norm(B, axis = 1)) * np.ones(shape=(A.shape[0],1))
            A_M_B = np.matmul(A,B.T)
            d_matrix = A_norm + B_norm - 2*A_M_B
            d_matrix = np.sqrt(d_matrix)

        if(self.metric == 'cosine'):
            from sklearn.metrics.pairwise import cosine_similarity
            d_matrix = cosine_similarity(A,B)


        if(self.metric == 'manhattan'):
            from scipy.spatial.distance import cdist
            d_matrix = cdist(A, B, metric='cityblock')
            d_matrix

        if(self.metric == 'edit'):
            d_matrix = np.matmul(A,B.T)
            d_matrix= (784-d_matrix)/2

        return d_matrix
    
    def predict(self, X_test):
        from scipy import stats
        self.similarity_matrix = self.distance_matrix(self.X_train, X_test)
        predictions = np.zeros((X_test.shape[0]))
        for i in range(X_test.shape[0]):
            predictions[i] = stats.mode(self.Y_train[(np.argsort(self.similarity_matrix[:,i]))[::self.order][0:self.k]])[0].item()

        return predictions
    
    