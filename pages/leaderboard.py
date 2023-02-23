import streamlit as st
import pandas as pd


# show the dataframe 

df = pd.read_csv("/home/mohamad/tmp/code_eval/df.csv")

st.write(df)

