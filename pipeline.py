

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.cluster import KMeans
import math
import urllib.request
import streamlit as st

#main data set
url = "https://www.dropbox.com/s/r20nh6xean1kurq/sales.csv"
response = urllib.request.urlopen(url)
sales_dataset = response.read()

st.set_page_config(layout="wide")
st.title('Identify Target Audience Using Sampling Techniques')
#data = pd.read_csv('sales_df_completed_uc.csv')
#st_df1  = pd.DataFrame(data)

file = st.file_uploader("Upload CSV", type="csv")

if file is not None:
    # Use pandas to read the file contents into a DataFrame
    data = pd.read_csv(file)
    st_df1  = pd.DataFrame(data)
    
    population_mean = st_df1['total_discount_received'].mean()
    
    st.markdown("")
    see_data = st.expander('You can click here to see the dataset first 👉')
    with see_data:
        st.dataframe(data=st_df1)
    st.text('')
    
    st.subheader("General Information about Dataset")
    row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))

    with row2_1:
        tot_records = sales_dataset.shape[0]
        Total_Records = "📉 No of Records: " +  str(tot_records) + " Records"
        st.markdown(Total_Records)
    with row2_2:
        no_of_variables_in_df = len(sales_dataset.columns)
        no_of_var_df = "🔢 No of Variables: " + str(no_of_variables_in_df) + " Variables"
        st.markdown(no_of_var_df)
        
    #with row3_1:
     
    #st.sidebar.title("Findings")
    #type_of_finding = st.sidebar.selectbox("Select one",('EDA', 'Clustering', 'Sampling Techniques'))   
        
    
    agree = st.checkbox('Apply Sampling Techniques')

    if agree:
        def simple_random_sampling(data, sample_sizes):
            np.random.seed(42)
            samples = []
            for size,confidence_interval in sample_sizes:
                indices = random.sample(range(len(data)),size)
                srs_sample = data.iloc[indices]
                srs_mean = srs_sample['total_discount_received'].mean()
                srs_sd = np.std(srs_sample['total_discount_received'], ddof=1) 
                srs_se = srs_sd / np.sqrt(size)
                srs_absolute_error = abs(population_mean - srs_mean)
                samples.append(['Simple Random Sampling',confidence_interval, size, srs_absolute_error, srs_se])
            return samples

        def stratified_sampling(data, sample_sizes):
            np.random.seed(42)
            data2 = data
            samples = []
            data2.loc[data2['Gender_M'] == 1, 'Customer_Strata'] = 1
            data2.loc[data2['Gender_M'] != 1, 'Customer_Strata'] = 0

            for size,confidence_interval in sample_sizes:
                split = StratifiedShuffleSplit(n_splits=1, test_size=size)
                for x, y in split.split(data2, data2['Customer_Strata']):
                    str_sample = data2.iloc[y].sort_values(by='cust_id')
                    str_mean = str_sample['total_discount_received'].mean()
                    str_se = np.std(str_sample['total_discount_received'], ddof=1) / np.sqrt(size)
                    str_absolute_error = abs(population_mean - str_mean)
                samples.append(['Stratified Sampling',confidence_interval, size, str_absolute_error, str_se])
            return samples 

        def systematic_sampling(data, sample_sizes):
            np.random.seed(42)
            samples = []
            for size,confidence_interval in sample_sizes:
                step = len(data) // size
                start = np.random.randint(0, step)
                indices = np.arange(start, len(data), step = step)
                sys_sample = data.iloc[indices]
                sys_mean = sys_sample['total_discount_received'].mean()
                sys_sd = np.std(sys_sample['total_discount_received'], ddof=1) 
                sys_se = sys_sd / np.sqrt(size)
                sys_absolute_error = abs(population_mean - sys_mean)
                samples.append(['Systematic Sampling',confidence_interval, size, sys_absolute_error, sys_se])
            return samples
            

        def cluster_sampling(data, sample_sizes):
            samples = []

            cluster1=data.loc[data['cluster'] == 0]
            cluster2=data.loc[data['cluster'] == 1]
            cluster3=data.loc[data['cluster'] == 3]

            clusters = []
            clusters.append(cluster1)
            clusters.append(cluster2)
            clusters.append(cluster3)

            random.seed(42)
            selected_cluster = random.choice(clusters)

            for size,confidence_interval in sample_sizes:
                indices = random.sample(range(len(selected_cluster)), size)
                cluster_sample = selected_cluster.iloc[indices]
                cluster_mean = cluster_sample['total_discount_received'].mean()
                cluster_se = np.std(cluster_sample['total_discount_received'], ddof=1) / np.sqrt(size)
                cluster_absolute_error = abs(population_mean - cluster_mean)
                samples.append(['Cluster Sampling', confidence_interval,size, cluster_absolute_error, cluster_se])
            return samples


        def sampling_pipeline(data, sample_sizes):
            srs_samples = simple_random_sampling(data, sample_sizes)
            str_samples = stratified_sampling(data, sample_sizes)
            sys_samples = systematic_sampling(data, sample_sizes)
            cls_samples = cluster_sampling(data, sample_sizes)

            # Combine results into DataFrame
            train_set = srs_samples  + sys_samples + cls_samples + str_samples
            df = pd.DataFrame(train_set, columns=['Sampling Technique','Confidence Interval','Sample Size', 'Absolute Error', 'Standard Error'])
            return df



        # Calculate the sample size for each confidence interval
        sample_sizes = []
        # Set the given values
        z_scores = [(1.645, '90%'), (1.96, '95%'), (2.576, '99%')]
        margin_of_error = 0.05
        population_proportion = 0.5
        population_size = len(st_df1) #9521

        for z, confidence_interval in z_scores:
            sample_size = (((z**2) * population_proportion * (1 - population_proportion)) / (margin_of_error**2)) / (1+ ((z**2) * population_proportion * (1 - population_proportion))/((margin_of_error**2)*population_size))
            sample_size = math.ceil(sample_size) # Round up to the nearest integer
            sample_sizes.append((sample_size, confidence_interval))

        #print("Sample sizes needed for 90%, 95%, and 99% confidence intervals:", sample_sizes)

        df = st_df1
        result_table = sampling_pipeline(df, sample_sizes)
        #st.write(result_table)
        st.table(result_table)
