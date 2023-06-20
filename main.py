import os
from key import openai_key
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.chains import SequentialChain

os.environ['OPENAI_API_KEY'] = openai_key
st.title("LangChain")
input_text = st.text_input("Enter your text here")

first_input_prompt = PromptTemplate(
    input_variables=['word'],
    template="tell me the meaning of the word {word}"

)


llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt,
                 verbose=True, output_key='word_meaning')

second_input_prompt = PromptTemplate(
    input_variables=['word_meaning'],
    template="give me 5 sentense of the word {word_meaning}"

)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt,
                  verbose=True, output_key='word_usage')

chain3 = SequentialChain(chains=[chain, chain2], input_variables=[
                         'word'], output_variables=['word_meaning', 'word_usage'], verbose=True)


if input_text:
    res = chain3({'word': input_text})
    st.write(res['word_meaning'])
    st.write(res['word_usage'])
