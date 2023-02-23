import streamlit as st
import pandas as pd
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

@st.cache_data()
def get_data():
    return []

#upload file
uploaded_file = st.file_uploader("Choose a file")

if st.button("Add row"):

    eval_env = gym.make("CartPole-v1")
    model = PPO.load(uploaded_file)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    # save the data in dfl.csv
    open("pages/df.csv", "a").write(f"{mean_reward},{std_reward}\n")
    st.write(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    get_data().append({"mean_reward": mean_reward, "std_reward": std_reward})

st.write(pd.DataFrame(get_data()))