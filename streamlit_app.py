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

positive_image_ids = []
negative_image_ids = []

# TODO: Update after every turn
# def get_gallery_images():
#     return ["default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png",
#                   "default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png", "default_image.png"]


def update_gallery():
    print("Updating gallery")
    st.session_state['gallery_images'] = [
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
        "default_image.png",
    ]


def generate_frontend():
    main_col1, main_col2 = st.columns(2)

    # Use st.session_state to create a state for Streamlit that will contain a list of images
    if 'gallery_images' not in st.session_state:
        update_gallery()
        

    # Left half - Tinder-like frontend
    with main_col1:
        st.subheader("Your Style")
        # images = get_gallery_images()
        for i in range(0, len(st.session_state['gallery_images']), 3):
            row = st.columns(3)
            for j in range(3):
                with row[j]:
                    st.image(st.session_state['gallery_images'][i + j], use_column_width=True)

    # Right half - Grid of images
    with main_col2:
        st.subheader("Swiper")
        card_image = st.image("default_image.png", use_column_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("No"):
                # TODO: Push the current image to the negative_image_ids list
                update_gallery()
                # print("Previous button clicked")
        with col2:
            if st.button("Yes"):
                # TODO: Push the current image to the negative_image_ids list
                update_gallery()
                # print("Next button clicked")


generate_frontend()
