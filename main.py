import json
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import AIMessage
import streamlit as st
import pywhatkit as kit
from langchain_core.messages import HumanMessage
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import csv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import re
import os
import glob






# Define the file path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"


directory = '/Users/vanshkumarsingh/Desktop/BEEHIVE/pythonProject/generated-pictures'


st.toast("Using Mistral")
model = Ollama(model='mistral:instruct')

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

# Create a dictionary to store drug interactions
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


# Function to check interaction between two drugs
@tool
def find_interaction(drug1_name, drug2_name):
    """
    Find the interaction between two drugs.

    drug1_name: Name of the first drug.
    drug2_name: Name of the second drug.
    Interaction type or 'No known interaction' if not found.
    """
    drug1_name = drug1_name.strip()
    drug2_name = drug2_name.strip()

    interaction = drug_interactions.get((drug1_name, drug2_name), None)

    if interaction:
        return f"The interaction between {drug1_name} and {drug2_name} is: {interaction}."
    else:
        return f"No known interaction between {drug1_name} and {drug2_name}."


# Load interactions from the CSV file
csv_file_path = "/Users/vanshkumarsingh/Desktop/hackx/pythonProject/data of multiple-type drug-drug interactions/DDI_data.csv"  # Replace with your actual CSV file path
load_interactions_from_csv(csv_file_path)

def get_task_decomposition(placeroute: str, query: str) -> str:
    """This is an RAG. It takes a URL or a PDF pathname with the query and gets the task decomposition for the given input."""

    # Ensure that you handle the input type properly
    bar.progress(30)
    def differentiate_input(input_string):
        # Regular expression for URL detection
        url_regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        # Check if the input is a URL
        if re.match(url_regex, input_string):
            return "URL"

        # Check if the input is a file path
        elif os.path.exists(input_string):
            if os.path.isfile(input_string):
                file_extension = os.path.splitext(input_string)[1].lower()
                if file_extension == '.pdf':
                    return "PDF File"
                else:
                    return "Unknown File Type"
            else:
                return "Not a File"
        else:
            return "Invalid Input"

    bar.progress(40)

    # Differentiate input
    input_type = differentiate_input(placeroute)

    # Process the input accordingly
    if input_type == "URL":
        loader = WebBaseLoader(web_paths=[placeroute])
        docs = loader.load()
    elif input_type == "PDF File":
        loader = PyPDFLoader(placeroute)
        docs = loader.load_and_split()
    else:
        return "No valid input provided"
    bar.progress(50)

    # Chunking and splitting documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    bar.progress(60)

    # Set up vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="mxbai-embed-large"))
    retriever = vectorstore.as_retriever()

    # Set up the RAG prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Formatting retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Set up the chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    bar.progress(70)

    # Invoke the chain and return the response
    return rag_chain.invoke(query)


import urllib.parse

@tool
def medicine(query):
    """Get the medicine reaction and safety and consumeablity for the specified query."""
    bar.progress(10)
    base_url = "https://www.google.com/search?q="
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"{base_url}{encoded_query}"
    bar.progress(20)
    return get_task_decomposition(search_url, query)



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

#tools = [repl, converse ,recognize_speech_from_microphone , ]
tools = [
    find_interaction,
    medicine,
    repl,
    converse
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

def send_message_via_whatsapp_desktop(phone_number, message):
    try:
        # Send the message via WhatsApp Desktop using pywhatkit
        kit.sendwhatmsg_instantly(phone_number, message, wait_time=10)
        return "Message sent successfully!"
    except Exception as e:
        return f"An error occurred: {e}"


# Set up message history.
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("I am an drug interaction checker , I'm here to assist in avoiding bad reactions and decisions. Always on, always learning. How can I help you today?")

# Set the page title.
st.title("Med Buddy")

# Render the chat history.
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
# Initialize the LLM with Google Generative AI

def get_image_caption(image_path : str) -> str:
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string representing the caption for the image.
    """
    bar.progress(30)
    image = Image.open(image_path).convert('RGB')

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"  # cuda

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    bar.progress(50)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    bar.progress(70)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption
def process_uploaded_image(uploaded_file):
    """Process the uploaded image, send to LLM, and delete afterward."""
    if uploaded_file is not None:
        # Save uploaded image temporarily
        image_path = os.path.join("tempDir", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the image in the app
        st.image(image_path, caption="Uploaded Image")
        bar.progress(10)

        # Print the image location
        st.write(f"Image saved at: {image_path}")

        # Get the caption for the image
        caption = get_image_caption(image_path)
        bar.progress(90)

        # Delete the image after processing
        if os.path.exists(image_path):
            os.remove(image_path)
            st.write(f"Image {image_path} deleted.")
        return str(caption)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

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
        if uploaded_file:
            res = process_uploaded_image(uploaded_file)
            response = chain.invoke({"input": input + "and"+ res + "medicine together safe or not", "chat_history": chat_history})
        else:
            response = chain.invoke({"input": input + "medicine together safe or not" , "chat_history": chat_history})
        chat_history.append(HumanMessage(content=input))
        chat_history.append(AIMessage(content=response))
        bar.progress(90)

        # Display AI assistant response and save to message history.
        st.chat_message("assistant").write(str(response))
        msgs.add_ai_message(response)


        #send_message_via_whatsapp_desktop("+9100000000", "User Input= " + input + " Response= " + str(response))   # Replace with your phone number

        save_chat_history()
        st.toast("Context Updated")
        bar.progress(100)