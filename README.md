# 📚 DocsMate AI

> **AI-Powered Document Assistant built with LangChain, Hugging Face, Qwen, and Streamlit.**

DocsMate AI is an intelligent document assistant that allows users to upload a PDF document and interact with it through a conversational AI interface.

Instead of manually reading through a document, users can ask questions in natural language and receive answers based on the uploaded document's content.

The application is designed to work with different types of PDF documents, including **research papers, reports, CVs, articles, manuals, notes, academic documents, and business documents**.

---

## ✨ Features

* 📄 **PDF Document Upload**
* 🤖 **AI-powered document understanding**
* 💬 **Conversational question answering**
* 🧠 **Conversation History**
* 📖 **Page-aware document context**
* ⚡ **Streaming AI responses**
* 📊 **Document statistics**

  * Total pages
  * Total characters
  * Conversation messages
* 🔍 **Context-based answers**
* 🚫 **Reduces unsupported answers with strict document-grounding instructions**
* 🎨 **Clean Streamlit interface**
* 🔐 **Hugging Face API token through Streamlit Secrets**
* 📑 Supports documents up to **5 pages** in the current version

---

## 🧠 How It Works

The application follows a simple LangChain pipeline:

```text
                📄 PDF Document
                       │
                       ▼
                PyPDFLoader
                       │
                       ▼
                  Documents
                       │
                       ▼
              Document Context
                       │
                       │
User Question ─────────┤
                       │
Conversation History ──┤
                       ▼
              RunnableParallel
                       │
                       ▼
             ChatPromptTemplate
                       │
                       ▼
              Qwen2.5-7B-Instruct
                       │
                       ▼
               StrOutputParser
                       │
                       ▼
                 AI Response
                       │
                       ▼
              Conversation History
```

---

## 🔧 LangChain Concepts Used

This project was built to practically apply core LangChain concepts.

### 1. Chat Model

The application uses:

```python
HuggingFaceEndpoint
ChatHuggingFace
```

with:

```text
Qwen/Qwen2.5-7B-Instruct
```

---

### 2. Documents

PDF files are loaded using:

```python
PyPDFLoader
```

Each page is represented as a LangChain `Document` containing:

* `page_content`
* `metadata`

The application also uses page metadata to identify the relevant page number.

---

### 3. ChatPromptTemplate

The prompt contains three major components:

```text
System Instructions
        ↓
Conversation History
        ↓
Current User Question + Document Context
```

This allows the model to understand both the document and the ongoing conversation.

---

### 4. MessagesPlaceholder

Conversation history is inserted into the prompt using:

```python
MessagesPlaceholder(
    variable_name="history"
)
```

This allows previous `HumanMessage` and `AIMessage` objects to be passed to the model.

---

### 5. RunnableLambda

Custom Python functions are converted into LangChain runnables using:

```python
RunnableLambda
```

The project uses this for:

* Cleaning the user's question
* Preparing document context
* Providing conversation history

---

### 6. RunnableParallel

Multiple inputs are prepared simultaneously:

```python
RunnableParallel({
    "question": question_chain,
    "context": context_chain,
    "history": history_chain
})
```

The result is passed to the prompt as a dictionary.

---

### 7. RunnableSequence

The complete chain is created using the LangChain pipe operator:

```python
final_chain = parallel | prompt | model | parser
```

The flow is:

```text
Parallel
   ↓
Prompt
   ↓
Model
   ↓
Parser
```

---

### 8. StrOutputParser

The model output is converted into a clean string using:

```python
StrOutputParser()
```

---

### 9. Conversation History

The application stores:

```python
HumanMessage
AIMessage
```

inside:

```python
st.session_state.history
```

This allows the assistant to maintain context across multiple questions during the session.

---

## 🛠️ Tech Stack

| Technology           | Purpose                   |
| -------------------- | ------------------------- |
| Python               | Core programming language |
| Streamlit            | Web application interface |
| LangChain            | LLM application framework |
| Hugging Face         | Model/API provider        |
| Qwen 2.5 7B Instruct | AI language model         |
| PyPDF                | PDF processing            |
| python-dotenv        | Environment configuration |

---

## 📁 Project Structure

```text
DocsMate/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── secrets.toml
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/documate-ai.git
```

```bash
cd documate-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Hugging Face API Token

The application uses a Hugging Face API token.

Create:

```text
.streamlit/secrets.toml
```

and add:

```toml
HF_TOKEN = "your_huggingface_token"
```

The application reads the token using:

```python
HF_TOKEN = st.secrets["HF_TOKEN"]
```

**Never upload your real API token to GitHub.**

Add this to `.gitignore`:

```text
.streamlit/secrets.toml
.env
venv/
__pycache__/
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

---

## 💡 Example Use Cases

DocsMate can be used to interact with different types of documents.

### 📚 Research Papers

```text
What is the main contribution of this paper?

Explain the proposed methodology.

What are the key findings?
```

### 📄 CV / Resume

```text
What technical skills are mentioned?

What projects are listed?

Summarize the candidate's experience.
```

### 📊 Business Reports

```text
What is the total revenue?

What are the major findings?

Summarize the report.
```

### 📖 Manuals

```text
What are the installation steps?

What does this feature do?

What are the important requirements?
```

### 📝 Academic Notes

```text
Explain this concept.

Give me the important points.

Summarize this chapter.
```

---

## 🛡️ Answering Philosophy

DocsMate is instructed to prioritize the uploaded document as its primary source.

The assistant is instructed to:

* Answer using available document information
* Avoid inventing facts
* Avoid unsupported assumptions
* Focus on information relevant to the user's question
* Clearly indicate when requested information cannot be found
* Keep responses structured and easy to scan

This makes the application more useful for document-focused question answering.

---

## ⚡ Current Limitations

The current version has a few intentional limitations.

### 1. PDF only

The current interface accepts:

```text
PDF
```

files.

### 2. 5-page limit

The current version limits uploaded PDFs to:

```text
5 pages
```

### 3. No Vector Database Yet

The current implementation passes the loaded document context directly to the model.

It does **not yet use**:

* Text Splitters
* Embeddings
* Vector Databases
* Retrievers

Therefore, this version is a **document-aware conversational assistant**, rather than a complete RAG system.

---

## 🚀 Future Improvements

The next version can transform DocsMate into a complete RAG application.

### Version 2

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Splitter
 ↓
Document Chunks
 ↓
Embeddings
 ↓
Vector Database
 ↓
Retriever
 ↓
Relevant Context
 ↓
Qwen
 ↓
Answer
```

Planned improvements:

* 🔹 Text splitting
* 🔹 Hugging Face embeddings
* 🔹 FAISS / Chroma vector database
* 🔹 Semantic search
* 🔹 Retriever
* 🔹 True RAG pipeline
* 🔹 Support for larger documents
* 🔹 Multiple document uploads
* 🔹 Source/page citations
* 🔹 Document comparison
* 🔹 Improved conversation memory

---

## 🎯 Learning Goals

This project was built as a practical implementation of core **LangChain concepts**, including:

```text
Chat Models
      ↓
Messages
      ↓
Prompt Templates
      ↓
Conversation History
      ↓
Documents
      ↓
RunnableLambda
      ↓
RunnableParallel
      ↓
RunnableSequence
      ↓
Output Parsers
```

The goal is to understand how these components work together to build a real-world LLM application.

---

## 👨‍💻 Author

**Shafiq Ahmad**

AI Engineer | Machine Learning Engineer

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 🔥 Deep Learning
* 💬 NLP
* 🦜 LangChain
* 🤗 Hugging Face
* 🔎 RAG Systems

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is available for educational and personal development purposes.

```
```
