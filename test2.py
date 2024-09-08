import json
import os
from datetime import datetime
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import speech_recognition as sr
import subprocess
import time
import gtts
import os
import platform
import pathlib
import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import cv2
import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
import os
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import getpass
import os
from langchain_google_genai import GoogleGenerativeAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import re
import os

import os
import glob






# Define the file path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

api_key= "AIzaSyB8EsK2ciq9xnu-8ebeusTgoEv-6yF6CmI"
directory = '/Users/vanshkumarsingh/Desktop/BEEHIVE/pythonProject/generated-pictures'


# Set up the LLM which will power our application.


st.toast("Using Gemini.")
model= ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
#model = Ollama(model='mistral:instruct')
chat_history = [] # Store the chat history


# Load chat history from file if exists
if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "r") as f:
        try:
            chat_history_data = json.load(f)
            for item in chat_history_data:
                if item['type'] == 'human':
                    chat_history.append(HumanMessage(content=item['content'] ))
                elif item['type'] == 'ai':
                    chat_history.append(AIMessage(content=item['content']))
        except json.JSONDecodeError:
            pass


# Define tools available.


@tool
def repl(input: str) -> str:
    """A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."""

    python_repl = PythonREPL()
    return python_repl.run(input)

@tool
def converse(input: str) -> str:
    """Provide a natural language response using the user input."""

    return model.invoke(input)

#tools = [repl, converse ,recognize_speech_from_microphone , ]
tools = [
    repl,
    converse
]


# Configure the system prompts
rendered_tools = render_text_description(tools)

system_prompt = f"""You just describe what is in the image ,Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys. The value associated with the 'arguments' key should be a dictionary of parameters."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
     MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]
)

# Define a function which returns the chosen tools as a runnable, based on user input.
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

# The main chain: an LLM with tools.
chain = prompt | model | JsonOutputParser() | tool_chain



def save_chat_history():
    chat_history_data = []
    for message in chat_history:
        if isinstance(message, HumanMessage) or isinstance(message, AIMessage):
            chat_history_data.append({"type": "human" if isinstance(message, HumanMessage) else "ai",
                                      "content": message.content,
                                      })

    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history_data, f, default=str)  # Use default=str to serialize datetime if needed

def clear_chat_history():
    global chat_history
    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)


def delete_uploaded_image(image_path):
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            st.toast(f"Image '{image_path}' deleted.")
        else:
            st.toast(f"File not found: {image_path}")
    except Exception as e:
        st.toast(f"Error deleting image: {str(e)}")
def delete_all_files():
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    for file in files:
        os.remove(file)
    return len(files)



# Set up message history.
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("From calculations to image generation, data analysis to task prioritization, I'm here to assist. Always on, always learning. How can I help you today?")

# Set the page title.
st.title("Ascendant Ai")

# Render the chat history.
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


# React to user input
uploaded_image = st.file_uploader("Upload an image (optional):")

if input := st.chat_input("What is up?"):

    if input == "/clear":
        clear_chat_history()
        st.chat_message("assistant").write("Chat history cleared.")
        delete_all_files()
        st.toast("Data Cleared")

    else:
        # Display user input and save to message history.
        st.chat_message("user").write(input)
        msgs.add_user_message(input)

        # If there is an uploaded image, use it
        if uploaded_image:
            # Process the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            if not os.path.exists("uploaded_images"):
                os.makedirs("uploaded_images")

                # Define the path where the image will be saved
            image_path = os.path.join("uploaded_images", uploaded_image.name)
            print(image_path)
            st.toast(image_path)

            # Save the image to the defined path
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text":  input,
                    },
                    {"type": "image_url", "image_url": image_path},
                ]
            )

            presponse = model.invoke([message])
            response = str(presponse.content)
            print(response)

            st.chat_message("assistant").write("Processing the uploaded image...")
            delete_uploaded_image(image_path)

        else:
            # Normal text-based input processing
            st.toast("Processing...")

            # Invoke chain to get response.

            presponse = chain.invoke({"input": input, "chat_history": chat_history})
            response = str(presponse.content)

            chat_history.append(HumanMessage(content=input))
            chat_history.append(AIMessage(content=str(response)))



        # Display AI assistant response and save to message history.
        st.chat_message("assistant").write(str(response))
        msgs.add_ai_message(response)

        save_chat_history()
        st.toast("Context Updated")








        # Ensure the model retains context
        # msgs.add_ai_mess
        #msgs.add_ai_message(model.invoke(input))