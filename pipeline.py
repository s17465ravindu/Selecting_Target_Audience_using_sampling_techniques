# -*- coding: utf-8 -*-
"""Pipeline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DHOPxt8vbcdXu9335ycpzkXZtCGR9D-O
"""
import pandas as pd
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from sklearn.preprocessing import StandardScaler
import random


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'hhttps://docs.google.com/spreadsheets/d/1baKBXDs_PiIIYyUFmq0UOCBcMwq6DWji/edit?usp=sharing&ouid=100934670427194582945&rtpof=true&sd=true'
id = '1baKBXDs_PiIIYyUFmq0UOCBcMwq6DWji'
print (id)

downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('sales_df_completed_uc.xlsx')  
st_df1 = pd.read_excel('sales_df_completed_uc.xlsx')

st_df2= st_df1[st_df1['respond_to_discount'] == 1]

len(st_df2)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def simple_random_sampling(data, sample_sizes):
    samples = []
    population_mean = data['total_discount_received'].mean()
    for size in sample_sizes:
        srs_sample = data.sample(size, random_state=42)
        srs_mean = srs_sample['total_discount_received'].mean()
        srs_se = np.std(srs_sample['total_discount_received'], ddof=1) / np.sqrt(size)
        srs_absolute_error = abs(population_mean - srs_mean)
        samples.append(['Simple Random Sampling', size, srs_absolute_error, srs_se])
    return samples

def stratified_sampling(data, sample_sizes):
    samples = []
    data['Customer_Strata'] = np.where(data['Gender_M'] == 1, 1, 0)
    split = StratifiedShuffleSplit(n_splits=1, random_state=42)
    population_mean = data['total_discount_received'].mean()
    for size in sample_sizes:
        for train_index, test_index in split.split(data, data['Customer_Strata']):
            str_sample = data.iloc[test_index].sort_values(by='cust_id').head(size)
        str_mean = str_sample['total_discount_received'].mean()
        str_se = np.std(str_sample['total_discount_received'], ddof=1) / np.sqrt(size)
        str_absolute_error = abs(population_mean - str_mean)
        samples.append(['Stratified Sampling', size, str_absolute_error, str_se])
    return samples

def systematic_sampling(data, sample_sizes):
    samples = []
    population_mean = data['total_discount_received'].mean()
    for size in sample_sizes:
        step = len(data) // size
        start = np.random.randint(0, step)
        indices = np.arange(start, len(data), step)
        sys_sample = data.iloc[indices]
        sys_mean = sys_sample['total_discount_received'].mean()
        sys_se = np.std(sys_sample['total_discount_received'], ddof=1) / np.sqrt(size)
        sys_absolute_error = abs(population_mean - sys_mean)
        samples.append(['Systematic Sampling', size, sys_absolute_error, sys_se])
    return samples

def cluster_sampling(data,sample_sizes):
  samples = []
  cluster1=data.loc[data['Region_Midwest'] == 1]
  cluster2=data.loc[data['Region_Northeast'] == 1]
  cluster3=data.loc[data['Region_South'] == 1]
  cluster4=data.loc[data['Region_West'] == 1]

  clusters = []
  clusters.append(cluster1)
  clusters.append(cluster2)
  clusters.append(cluster3)
  clusters.append(cluster4)

  random.seed(42)
  selected_cluster = random.choice(clusters)
  population_mean = data['total_discount_received'].mean()

  for size in sample_sizes:
    cluster_sample = selected_cluster.sample(size, random_state=42)
    cluster_mean = cluster_sample['total_discount_received'].mean()
    cluster_se = np.std(cluster_sample['total_discount_received'], ddof=1) / np.sqrt(size)
    cluster_absolute_error = abs(population_mean - cluster_mean)
    samples.append(['Cluster Sampling', size, cluster_absolute_error, cluster_se])
  return samples
    

def sampling_pipeline(data, sample_sizes):
    srs_samples = simple_random_sampling(data, sample_sizes)
    str_samples = stratified_sampling(data, sample_sizes)
    sys_samples = systematic_sampling(data, sample_sizes)
    cls_samples = cluster_sampling(data, sample_sizes)

    # Combine results into DataFrame
    train_set = srs_samples + str_samples + sys_samples + cls_samples
    df = pd.DataFrame(train_set, columns=['Sampling Technique', 'Sample Size', 'Absolute Error', 'Standard Error'])
    return df

"""### Calculate Sample Sizes ###"""

import math

# Set the given values
z_scores = [1.645, 1.96, 2.576] # z-score values for 90%, 95%, and 99% confidence intervals
margin_of_error = 0.05
population_proportion = 0.5
population_size = 9521

# Calculate the sample size for each confidence interval
sample_sizes = []
for z in z_scores:
    sample_size = (((z**2) * population_proportion * (1 - population_proportion)) / (margin_of_error**2)) / (1+ ((z**2) * population_proportion * (1 - population_proportion))/((margin_of_error**2)*population_size))
    sample_size = math.ceil(sample_size) # Round up to the nearest integer
    sample_sizes.append(sample_size)

print("Sample sizes needed for 90%, 95%, and 99% confidence intervals:", sample_sizes)

df = st_df2
result_table = sampling_pipeline(df, sample_sizes)
print(result_table)

'''0  36
1  115
2  63
3  178
4  126
5  101
6  21
7  197
8  27
9  230
10 132
11 77'''