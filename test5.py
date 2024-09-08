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

def ping_google_dns():
    try:
        # Run the ping command
        output = subprocess.run(['ping', '8.8.8.8', '-c', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check the return code
        if output.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        # In case of any exceptions, return False
        print(f"An error occurred: {e}")
        return False

Online = ping_google_dns()
# Set up the LLM which will power our application.

if Online:
    st.toast("Using Gemini.")
    model=GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
elif not Online:
    st.toast("Using Mistral")
    model = Ollama(model='llama3:instruct')

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
    bar.progress(40)
    python_repl = PythonREPL()
    return python_repl.run(input)

@tool
def converse(input: str) -> str:
    """Provide a natural language response using the user input."""
    bar.progress(40)
    return model.invoke(input)


from langchain.tools import Tool
import csv

# Dictionary to store drug interactions
drug_interactions = {}


# Load CSV file and populate the dictionary
def load_interactions_from_csv(file_path):
    """
    Load drug interactions from a CSV file and store them in a dictionary.

    :param file_path: The path to the CSV file.
    """
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            drug1 = row['drug1_name'].strip()
            drug2 = row['drug2_name'].strip()
            interaction = row['interaction_type'].strip()

            # Store interaction both ways (drug1-drug2 and drug2-drug1)
            drug_interactions[(drug1, drug2)] = interaction
            drug_interactions[(drug2, drug1)] = interaction


# Load the interactions CSV data when initializing the tool (assumed to be available at a known path)
csv_file_path = "/Users/vanshkumarsingh/Downloads/data of multiple-type drug-drug interactions/DDI_data.csv"  # Replace with your actual CSV file path
load_interactions_from_csv(csv_file_path)

@tool
# LangChain tool function for finding drug interactions
def find_interaction(drug1_name: str, drug2_name: str) -> str:
    """
    Find the interaction between two drugs.
    """
    drug1_name = drug1_name.strip()
    drug2_name = drug2_name.strip()

    interaction = drug_interactions.get((drug1_name, drug2_name), None)

    if interaction:
        return f"The interaction between {drug1_name} and {drug2_name} is: {interaction}."
    else:
        return f"No known interaction between {drug1_name} and {drug2_name}."



# Now you can use the drug_interaction_tool in a LangChain chain or agent.


#tools = [repl, converse ,recognize_speech_from_microphone , ]
tools = [
    repl,
    converse,
    find_interaction
]


# Configure the system prompts
rendered_tools = render_text_description(tools)

system_prompt = f"""You answer questions with simple answers and no funny stuff , You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

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
if input := st.chat_input("What is up?"):

    if input == "/clear":
        clear_chat_history()
        #print("Chat history cleared.")
        st.chat_message("assistant").write("Chat history cleared.")
        delete_all_files()
        st.toast("Data Cleared")

    else:
        # Display user input and save to message history.
        st.chat_message("user").write(input)
        msgs.add_user_message(input)

        # Invoke chain to get response.
        bar = st.progress(0)
        response = chain.invoke({"input": input, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=input))
        chat_history.append(AIMessage(content=response))
        bar.progress(90)

        # Display AI assistant response and save to message history.
        st.chat_message("assistant").write(str(response))
        msgs.add_ai_message(response)

        save_chat_history()
        st.toast("Context Updated")
        bar.progress(100)

        # Ensure the model retains context
        #msgs.add_ai_message(model.invoke(input))
