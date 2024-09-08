from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key="AIzaSyAIsE4C0ZjwCuO0A6S7IEjszpY9MBjAgWE")
# example
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Write a two to four word description of the file",
        },
        {"type": "image_url", "image_url": "/Users/vanshkumarsingh/Desktop/testt.png"},
    ]
)


print(llm.invoke([message]))