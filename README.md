# MedBuddy-OnDeviceAI
AI-Powered Medical Assistant with On-Device Processing and Privacy-First Design

![Screenshot 2024-09-09 at 12 47 36 AM](https://github.com/user-attachments/assets/1122538f-b683-4c41-8113-7a84fad0078d)

**Med Buddy** is an AI-powered medical assistant that performs all processing on-device, ensuring data privacy and fast performance. It leverages advanced language models to assist users with drug interactions, generate natural language responses, and process images. The only exception to on-device processing is the RAG (Retrieval-Augmented Generation) task for drug interaction retrieval, which connects to a database.

## Key Features

- **On-Device Processing**: All tasks, including natural language processing, image captioning, and tool interaction, are done locally using powerful models like **Ollama** and **Mistral-Instruct**. No external APIs are used except for drug interaction retrieval.
- **Drug Interaction Checker**: Check for potential drug interactions using a local CSV database.
- **Task Decomposition**: Analyze URLs and PDFs for task-specific information using RAG.
- **Natural Language Conversations**: Get AI-generated responses in a conversational format based on user queries.
- **Python REPL**: Execute Python commands within the app for debugging or utility purposes.
- **Image Captioning**: Uses the **BLIP Transformer** to generate captions for uploaded images.
- **WhatsApp Messaging**: Send messages through WhatsApp Desktop using `pywhatkit`.
- **Speech Recognition**: Recognizes and processes user speech input on-device.

## Models Used

- **Ollama**: Powered by the **Mistral-Instruct** model for natural language understanding and generation.
- **BLIP Transformer**: Used for generating image captions from uploaded images.
- **Chroma**: Used for local document embedding and vector store management.
- **OpenAI Embeddings**: For embedding documents and creating vector stores in local retrieval tasks.

## No API Dependency

- All processes are handled on-device, ensuring privacy and faster processing times.
- The only external dependency is the drug interaction RAG retrieval for which a CSV-based drug interaction database is used.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/med-buddy.git
   cd med-buddy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (for drug interaction RAG retrieval):
   ```bash
   export GOOGLE_API_KEY="your-google-api-key"
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. (Optional) Add your CSV file containing drug interactions:
   - Place your CSV file (e.g., `DDI_data.csv`) in the `data` folder.
   - Update the file path in the code where necessary.

## Usage

- Open the web app in your browser using the provided local Streamlit URL.
- Interact with the chatbot for checking drug interactions or asking medical questions.
- Upload images to get AI-generated captions using the BLIP transformer.
- Clear the chat history and reset data if needed using the `/clear` command.
- Automatically send messages via WhatsApp using `pywhatkit`.

## File Structure

```
.
├── app.py                # Main application file
├── chat_history.json      # Stores chat history
├── requirements.txt       # List of dependencies
├── data
│   └── DDI_data.csv       # CSV file for drug interactions
└── README.md              # Project documentation
```

## Future Enhancements

- Expand the local drug database for more comprehensive interaction checks.
- Integrate voice commands for hands-free operation.
- Add additional medical tools and features for diagnosis and patient management.

## Contributors

- **Vansh Kumar Singh** ([@vanshksingh](https://github.com/vanshksingh)) – Developer

## License

This project is licensed under the MIT License.
