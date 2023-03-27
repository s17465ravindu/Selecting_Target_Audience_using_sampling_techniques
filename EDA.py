import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import json
import requests
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

#link = 'https://drive.google.com/open?id=1G0KdR1AxGkEAic-1VCaNi7NUK2YdkupQ'
#fluff, id = link.split('=')
id = '100934670427194582945'
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('sales_df_clean.csv')  
sales_df = pd.read_csv('sales_df_clean.csv')
