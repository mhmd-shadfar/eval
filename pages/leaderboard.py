import streamlit as st
import pandas as pd


# show the dataframe 

df = pd.read_csv("df.csv")

st.write(df)

