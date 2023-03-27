import streamlit as st
import numpy as np
import pandas as pd
import pickle

selected_day = st.selectbox(
    "Select 👇", ('Age distribution of completed orders','category'))

with st.expander("About the #30DaysOfStreamlit"):
    st.markdown(
        """
    The **#30DaysOfStreamlit** is a coding challenge designed to help you get started in building Streamlit apps.
    
    Particularly, you'll be able to:
    - Set up a coding environment for building Streamlit apps
    - Build your first Streamlit app
    - Learn about all the awesome input/output widgets to use for your Streamlit app
    """
    )

