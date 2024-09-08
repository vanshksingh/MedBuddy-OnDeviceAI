import streamlit as st
import pywhatkit as kit

def send_message_via_whatsapp_desktop(phone_number, message):
    try:
        # Send the message via WhatsApp Desktop using pywhatkit
        kit.sendwhatmsg_instantly(phone_number, message, wait_time=10)
        return "Message sent successfully!"
    except Exception as e:
        return f"An error occurred: {e}"


print(send_message_via_whatsapp_desktop("+918570099643", "Hello, this is a test message!"))