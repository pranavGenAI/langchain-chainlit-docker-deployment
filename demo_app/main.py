__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq


# Function to initialize conversation chain with GROQ language model
groq_api_key = "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"

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-8b-8192",
    temperature=0.2
)
print("Initialized GROQ....")

@cl.on_chat_start
async def on_chat_start():
    files = None  # Initialize variable to store uploaded files
    await cl.Avatar(
        name="BidBooster",
        url="https://static.vecteezy.com/system/resources/previews/025/213/708/original/ai-robot-chatbot-png.png",
    ).send()

    await cl.Avatar(
        name="You",
        url="https://cdn1.iconfinder.com/data/icons/user-pictures/100/male3-512.png",
    ).send()

    await cl.Message(content="BidBooster to help you with RFP queries!", author="BidBooster").send()
    
    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a RFP file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,  # Optionally limit the file size
            timeout=180,  # Set a timeout for user response,
        ).send()

    file = files[0]  # Get the first uploaded file
    print("printing file for debugging......",file)  # Print the file object for debugging

    # Inform the user that processing has started
    msg = cl.Message(content=f"Preparing BidBooster ... Please wait. It may take couple of minutes... ⏳⏳",)
    await msg.send()

    # Read the PDF file
    print("Reading PDF now....")

    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    print("Read PDF Success....")

    # Split the text into chunks
    print("Text split Start....")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    texts = text_splitter.split_text(pdf_text)
    print("Text split Success....")
    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    print("Embedding start....")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    print("Embedding end....")
    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing done. You can now ask questions!"
    await msg.update()
    # Store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    # Callbacks happen asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    await cl.Avatar(
        name="BidBooster",
        url="https://static.vecteezy.com/system/resources/previews/025/213/708/original/ai-robot-chatbot-png.png",
    ).send()

    await cl.Avatar(
        name="You",
        url="https://cdn1.iconfinder.com/data/icons/user-pictures/100/male3-512.png",
    ).send()

    # Call the chain with user's message content
    message.content = "Answer the question as detailed as possible from the provided context and in the most polite way, make sure to provide all the details in summarized format. Start with polite and nice statement. You can use greetings if you want. Don't provide the wrong answer. If you are giving answer and not using context to frame the answer then let the user know that the answer is not from the context.\n\n. And remember to format your answer in nicer way. End your response with disclaimer telling about the answer is from the context. Question from user is: " + message.content
    print("Prompt is called")
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    # Return results
    await cl.Message(content=answer, elements=text_elements, author="BidBooster").send()
