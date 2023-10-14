import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

import os

import random

import uuid

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_csv_agent
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.llms import OpenAI

from sqlalchemy.sql import text

st.set_page_config(page_title="Shopper ChatBot")

"""
# Shopper ChatBot
#### 💬 (Beta)

"""

# Sidebar contents
# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi I'm the Shopper Chatbot, how may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Convert the chat history into a text format
def chat_to_text(past, generated):
    chat_text = ""
    for i in range(len(past)):
        chat_text += f"User: {past[i]}\n"
        chat_text += f"Shopper Chatbot: {generated[i]}\n\n"
    return chat_text

st.sidebar.title('ABAiGuide 💬 (Beta)')
openai_api_key = ''
if len(os.environ['OPENAI_API_KEY']) > 0:
    openai_api_key = os.environ['OPENAI_API_KEY']
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_name = st.sidebar.radio("Model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)
st.sidebar.markdown("Note. GPT-4 is recommended for better performance.")
st.sidebar.markdown('''
## About
This is a Beta version/prototype of an AI powered shopping bot

AGI House
''')

# app_env = st.secrets["env_vars"]["app_env"]
# db_name = f"postgresql_{app_env}"
# print('db_name: ')
# print(db_name)

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = uuid.uuid1()
session_id = st.session_state['session_id']
print('session_id: ')
print(session_id)

if 'dialog_turn' not in st.session_state:
    st.session_state['dialog_turn'] = 0

# Provide a button to download the chat history
st.sidebar.download_button(
    label="Download chat history",
    data=chat_to_text(st.session_state['past'], st.session_state['generated']),
    file_name="chat_history.txt",
    mime="text/plain"
)

# Layout of input/response containers
response_container = st.container()
colored_header(label='', description='', color_name='blue-30')
input_container = st.container()

if 'input_buffer' not in st.session_state:
    st.session_state.input_buffer = ''

def submit_input():
    st.session_state.input_buffer = st.session_state.input
    st.session_state.input = ''

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("User: ", "", key="input", on_change=submit_input)
    return st.session_state.input_buffer

default_machine_prompt = """
You are an shopping chat bot. Respond with the best product recommendations.

"""

def generate_response(chat, machine_prompt, human_prompt, session_state):
    system_prompt = f"""
    {machine_prompt}

    These are the previous dialogs in the conversation:

    """
    for i, (past, generated) in enumerate(zip(session_state.past, session_state.generated)):
        previous_conv = f"""
            dialog_turn: {i}
            user: {past}
            shopper_chatbot: {generated}

        """
        system_prompt += previous_conv

    print("system_prompt: ")
    print(system_prompt)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    return chat(messages).content

if len(openai_api_key) > 0:

    st.divider()

    ## Set OpenAI API Key (get from https://platform.openai.com/account/api-keys)
    os.environ["OPENAI_API_KEY"] = openai_api_key

    ## Instantiate model
    llm = ChatOpenAI(model_name=model_name, temperature=0.5)

    ## Applying the user input box
    with input_container:
        user_input = get_text()

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            response = generate_response(llm, default_machine_prompt, user_input, st.session_state)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)
            # Log the response
            st.session_state['dialog_turn'] += 1
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
