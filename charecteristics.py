import pandas as pd

df = pd.read_csv('/Users/shervinee/Downloads/ratings_Musical_Instruments.csv', names = ['user','item','ratings','timestamp'])

num_user= len(set(df['user']))
num_item= len(set(df['item']))

print num_user
print num_item

df = pd.read_csv('/Users/shervinee/Downloads/ratings_Amazon_Instant_Video.csv', names = ['user','item','ratings','timestamp'])

num_user= len(set(df['user']))
num_item= len(set(df['item']))

print num_user
print num_item
