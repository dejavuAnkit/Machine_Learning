from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


model = ChatOpenAI()

st.header("Translation Check")


user_input = st.text_input("Likho Kuch bhi...")

if st.button("Summrize"):
    result = model.invoke(user_input)
    st.write(result.content)