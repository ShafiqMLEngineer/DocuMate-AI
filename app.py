import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda
)


# ============================================================
# CONFIG
# ============================================================

# load_dotenv()

HF_TOKEN = st.secrets["HF_TOKEN"]


st.set_page_config(
    page_title="DocMate",
    page_icon="📚",
    huggingfacehub_api_token=HF_TOKEN,
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-top: 0px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">📚 DocsMate</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'AI-powered Documents Assistant'
    '</p>',
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "paper_name" not in st.session_state:
    st.session_state.paper_name = None


# ============================================================
# MODEL
# ============================================================

@st.cache_resource # ---- store it in cache do not load again and again
def load_model():

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        temperature=0.2,
        max_new_tokens=500
    )

    model = ChatHuggingFace(
        llm=llm
    )

    return model


model = load_model()

parser = StrOutputParser()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 DocsMate")

    st.write("Your AI Document Assistant")

    st.divider() #-------------------------

    uploaded_file = st.file_uploader("Upload Document",type=["pdf"]) # geting file from GUI

    st.divider() #-------------------------

    if st.button("🗑️ Clear Conversation",use_container_width=True): # full width button

        st.session_state.history = []

        st.rerun()


# ============================================================
# PDF LOADING
# ============================================================

if uploaded_file is not None:

    # Load only when a new PDF is uploaded

    if (st.session_state.paper_name != uploaded_file.name):

        with st.spinner("📖 Reading Document paper..."):

            # Create temporary PDF where we copy all data of uploaded_file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())

                temp_path = temp_file.name # path of file... temp_file is object

            #----------------PDF Document Loader Lanchain------------

            # Load PDF
            loader = PyPDFLoader(
                temp_path
            )

            documents = loader.load()


            # Save documents
            st.session_state.documents = documents

            st.session_state.paper_name = (uploaded_file.name) # file name store in session


            # New paper = new conversation
            st.session_state.history = []


            # Delete temporary file
            os.remove(temp_path)


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.documents: #--- if document is available in session_state

    documents = st.session_state.documents

    total_pages = len(documents)

    total_characters = sum(len(doc.page_content)for doc in documents) # it tell total text length 

    # total = 0
    # for doc in documents:
    #     total += len(doc.page_content)


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric("📄 Pages",total_pages)

    with col2:

        st.metric("📝 Characters",f"{total_characters:,}")

    with col3:

        st.metric("💬 Messages",len(st.session_state.history))

    st.success(
        f"Loaded: **{st.session_state.paper_name}**"
    )

else:

    st.info(
        "👈 Upload a Document from the sidebar to begin."
    )


# ============================================================
# DOCUMENT → TEXT
# ============================================================

def get_document_context(documents):

    context_parts = []

    for doc in documents:

        page_number = (doc.metadata.get("page",0) + 1)  # if page not found return 0 .... and pages start from 0 so we add 1 to show user

        content = doc.page_content.strip()

        context_parts.append(f"""--- Page {page_number} ---{content}""") # to give this style data to llm you create ny design

    return "\n".join(context_parts)


# ============================================================
# QUESTION CLEANER
# ============================================================

def clean_question(question):

    return question.strip()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",

            """
            You are DocumentsMate, an expert AI Document assistant.

            Your job is to answer questions using the
            provided document paper.

            Rules:

            1. Use the document context whenever possible.
            2. Do not invent information.
            3. If the answer cannot be found in the paper,
            clearly say so.
            4. Explain difficult concepts in beginner-friendly
            language when appropriate.
            5. Use examples when they help.
            6. If the user asks for a summary, provide a
            structured summary.
            7. If the user asks about a specific page,
            use the page information provided.
            """
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            """
            Document CONTEXT:{context}

            USER QUESTION:

            {question}
            """
        )

    ]

)


# ============================================================
# LANGCHAIN PIPELINE
# ============================================================

if st.session_state.documents:

    # Take current state OUTSIDE RunnableLambda
    documents = st.session_state.documents

    history = st.session_state.history


    # --------------------------------------------------------
    # PDF CONTEXT
    # --------------------------------------------------------

    question_cleaner = RunnableLambda(clean_question)

    context_chain = RunnableLambda(lambda _: get_document_context(documents))

    # def my_function(_):
    #   return get_document_context(documents)


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_chain = RunnableLambda(lambda _: history)


    # --------------------------------------------------------
    # PARALLEL
    # --------------------------------------------------------

    parallel = RunnableParallel(

        {

            "question":question_cleaner,

            "context":context_chain,

            "history":history_chain

        }

    )


    # --------------------------------------------------------
    # FINAL CHAIN
    # --------------------------------------------------------

    final_chain = (

        parallel | prompt | model | parser

    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.history:

    if isinstance(message,HumanMessage):

        with st.chat_message("user"):

            st.write(message.content)


    elif isinstance(message,AIMessage):

        with st.chat_message("assistant"):

            st.write(message.content)


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask anything about your Document..."
)


if user_question:

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not st.session_state.documents:

        st.warning(
            "Please upload a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(
            user_question
        )


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 DocMate is thinking..."
        ):

            answer = final_chain.invoke(
                user_question
            )

        st.write(answer)


    # --------------------------------------------------------
    # UPDATE HISTORY
    # --------------------------------------------------------

    st.session_state.history.append(

        HumanMessage(
            content=user_question
        )

    )

    st.session_state.history.append(

        AIMessage(
            content=answer
        )

    )


    # --------------------------------------------------------
    # RERUN
    # --------------------------------------------------------

    st.rerun()