# Conversational RAG 

This project is a **Conversational Retrieval-Augmented Generation (RAG) API** built using **FastAPI** and **Google Generative AI**. It allows users to upload PDFs, process them into vector embeddings, and query them using a conversational interface.

## Features 🚀
- Upload and process **PDF documents**.
- Store and retrieve **chat history** per session.
- Utilize **Google Generative AI** for **query-based responses**.
- Implement **FastAPI** for a high-performance API.
- Store embeddings using **ChromaDB**.
- Automatically handle **session-based document mapping**.

---

## Installation 🛠️

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/RicardyC/Conversation-RAG-chatbot.git
cd conversational-rag-api
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add your **Google API Key**:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Usage 🚦

### **Run the API**
```bash
uvicorn main:app --reload
```
The API will be available at: **http://127.0.0.1:8000**

### **Endpoints**

#### 1️⃣ Upload a PDF
```http
POST /upload_pdf
```
**Request Body:**
- `session_id` (string) → Unique session identifier.
- `file` (PDF) → The document to process.

#### 2️⃣ Query the RAG Model
```http
POST /query
```
**Request Body:**
- `session_id` (string) → Matches the uploaded document.
- `user_query` (string) → User's input question.

---

## Project Structure 📂
```
/Conversation-RAG-chatbot
│── main.py                # FastAPI application
│── requirements.txt       # Dependencies
│── .env                   # Environment variables
│── /uploads               # Uploaded PDF storage
│── /chroma_db             # Vector database storage
└── README.md              # Documentation
```

---

## Future Improvements 🌱
- Add **authentication & authorization**.
- Enhance **multi-user session handling**.
- Implement **streaming responses** for real-time interaction.
- Support **multiple document types** (TXT, DOCX, etc.).

---

## License 📜
This project is licensed under the **MIT License**.
