import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Please set it in a .env file.")

# FastAPI app instance
app = FastAPI(title="Conversational RAG API", version="1.1")

# Storage for chat history and session-PDF mapping
chat_history_store = {}
session_pdf_map = {}  # Maps session IDs to processed PDF filenames

# Vectorstore directory
VECTORSTORE_DIR = "./chroma_db"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

# Function to process PDF
def process_pdf(pdf_path: str):
    """Loads, splits, and embeds PDF content."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=VECTORSTORE_DIR)
    return vectorstore

# Load the vector store
def load_vectorstore():
    """Loads the persisted vector database."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

# Function to set up RAG pipeline
def setup_rag_pipeline(vectorstore):
    """Sets up the conversational RAG pipeline."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

    system_prompt = """
    Answer the following question based on the context provided. 
    Think step by step before providing the answer.
    <context>
    {context}
    </context>
    """

    # QA Chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    retriever = vectorstore.as_retriever()

    # Contextual question reformulation
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the question to be standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Wrap with session history management
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

# Global variable for RAG pipeline
rag_chain = None


# 📌 Endpoint: Upload and process PDF with Session ID
@app.post("/upload_pdf")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    """Uploads a PDF, processes it, and associates it with a Session ID."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    os.makedirs("./uploads", exist_ok=True)

    # Check if session already has a PDF
    if session_id in session_pdf_map:
        old_pdf_path = session_pdf_map[session_id]
        if os.path.exists(old_pdf_path):
            os.remove(old_pdf_path)  # Remove old file

    # Save new PDF
    new_pdf_filename = f"{session_id}_{uuid.uuid4().hex}.pdf"
    file_path = os.path.join("./uploads", new_pdf_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process PDF and create vector embeddings
    try:
        process_pdf(file_path)
        session_pdf_map[session_id] = file_path  # Associate PDF with session
    except Exception as e:
        os.remove(file_path)  # Clean up if processing fails
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    return JSONResponse(status_code=202, content={"message": "PDF processing successful.", "session_id": session_id})


# 📌 Endpoint: Handle user queries
@app.post("/query")
async def query_rag(session_id: str = Form(...), user_query: str = Form(...)):
    """Handles user queries using the RAG pipeline."""
    global rag_chain
    if not session_id or not user_query:
        raise HTTPException(status_code=400, detail="Missing session ID or query.")

    # Check if session exists
    if session_id not in session_pdf_map:
        return JSONResponse(status_code=400, content={"error": "No document uploaded for this session. Please upload a PDF using the /upload_pdf endpoint."})

    # Load vectorstore and initialize RAG pipeline if not already done
    if rag_chain is None:
        try:
            vectorstore = load_vectorstore()
            rag_chain = setup_rag_pipeline(vectorstore)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading RAG pipeline: {str(e)}")

    # Invoke RAG pipeline
    try:
        response = rag_chain.invoke(
            {"input": user_query},
            config={"configurable": {"session_id": session_id}},
        )
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# Run FastAPI if script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
