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
import pinecone
import numpy as np


pinecone.init(api_key="API_KEY", environment="us-west1-gcp-free")
# Initialize the pinecone index
index = pinecone.Index("agihouse")

positive_image_ids = []
negative_image_ids = []
NAMESPACE = 'test2'
# TODO: start with random centroid
fetch_response = index.fetch(ids=["1"], namespace=NAMESPACE)
id = list(fetch_response.vectors.keys())[0]
centroid = np.array(fetch_response.vectors[id]['values'])
negative_centroid = []
print(centroid[0])





# TODO: Update after every turn
# def get_gallery_images():
#     return ["default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png",
#                   "default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png"]


class Product:
    def __init__(self, id, url, embedding):
        self.id = id
        self.url = url
        self.embedding = embedding

def update_gallery():
    id = list(fetch_response.vectors.keys())[0]
    query_response = index.query(
        namespace=NAMESPACE,
        top_k=13,
        include_values=True,
        include_metadata=True,
        vector=fetch_response.vectors[id]['values'],
    )
    st.session_state['gallery_images'] = []
    print("Updating gallery")
    # st.session_state['active_prod_vec'] = query_response['matches'][0]['vector']
    for i, match in enumerate(query_response['matches']):
        if i == 0:
            st.session_state['active_prod_vec'] = match
        st.session_state['gallery_images'].append(match['metadata']['link'])
        print(match['metadata']['link'])
    print('NANI')



def generate_frontend():
    main_col1, main_col2 = st.columns(2)

    # Use st.session_state to create a state for Streamlit that will contain a list of images
    if 'gallery_images' not in st.session_state or 'active_prod_vec' not in st.session_state or 'centroid' not in st.session_state or 'seen' not in st.session_state:
        st.session_state['seen'] = set()
        update_gallery()
        st.session_state['centroid'] = centroid


    # Left half - Tinder-like frontend
    with main_col1:
        st.subheader("Your Style")
        # images = get_gallery_images()
        used_list = st.session_state['gallery_images'][1:]
        for i in range(0, len(used_list), 3):
            row = st.columns(3)
            for j in range(3):
                with row[j]:
                    st.image(used_list[i + j], use_column_width=True)

    # Right half - Grid of images
    with main_col2:
        st.subheader("Swiper")
        card_image = st.image(st.session_state['gallery_images'][0], use_column_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("No"):
                # TODO: Push the current image to the negative_image_ids list
                update_gallery()
                # st.session_state['negative'] = # TODO

                # print("Previous button clicked")
        with col2:
            if st.button("Yes"):
                # TODO: Push the current image to the negative_image_ids list
                update_gallery()
                print(type(st.session_state['centroid']))
                # st.session_
                curr = np.array(st.session_state['centroid'])
                new = np.array(st.session_state['active_prod_vec']['values'])
                st.session_state['centroid'] = np.mean([curr, new], axis=0)
                # print("Next button clicked")


generate_frontend()
