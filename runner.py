import os
import pickle
from surprise import evaluate
from surprise import print_perf
from surprise import Reader
from surprise import Dataset
from SVD_A import SVDplusplus_A
from SVD_A import SVD_A
import pandas as pd
#used to test the time of execution by commented out since accuracy is the main focus of project
# import time
# start_time = time.time()

#reeading the input file a pandas dataframe
df = pd.read_csv('/Users/shervinee/Downloads/ratings_Amazon_Instant_Video.csv')

#sampling the dataset
samp_df = df.sample(n = 500)

#removing indices from dataframe
samp_df.to_csv("sampeled_run.csv", index=False)

#now we get a file path used for surprise
file_path = os.path.expanduser('sampeled_run.csv')

#building a reader for our dataset file
rdr = Reader(line_format='user item rating timestamp', sep=',')

#dataset would be saved on memory
data = Dataset.load_from_file(file_path, reader = rdr)

#the number of folds for cross validation would be given with callign the split function on data
data.split(n_folds = 5)

#the two implemented aproaches in SVD_A would be called
algo1 = SVDplusplus_A(n_factors=11,n_iter = 40)
algo2 = SVD_A(n_factors = 11, n_iter = 20)

#the evaluations of the two algorithms would be handled giving the data and evaluation metric 
perf = evaluate(algo1, data, measures=['rmse'])
perf = evaluate(algo2, data, measures=['rmse'])


# print time.time() - start_time
