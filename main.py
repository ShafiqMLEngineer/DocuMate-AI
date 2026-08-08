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


# load_dotenv()

HF_TOKEN = st.secrets["HF_TOKEN"]

st.set_page_config(
    page_title="ResearchMate",
    page_icon="📚",
    layout="wide"
)
# -------------------------------------
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

st.markdown(
    '<p class="main-title">📚 DocsMate</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'AI-powered Research Paper Assistant'
    '</p>',
    unsafe_allow_html=True
)




if "history" not in st.session_state:
    st.session_state.history = []

if "document" not in st.session_state:
    st.session_state.document = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# Model 

@st.cache_resource
def load_model():

    llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            temperature=0.2,
            huggingfacehub_api_token=HF_TOKEN,
            max_new_tokens=1000
        )
    
    model = ChatHuggingFace(
            llm=llm
        )
    
    return model


model = load_model()

# file upload
with st.sidebar:
    upload_file = st.file_uploader("Upload",type=['pdf'])
    if upload_file is not None:
        st.write("Name:",upload_file.name)
        st.write('Type:',upload_file.type)
        st.write('size',upload_file.size)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )as temp_file:
            
            temp_file.write(upload_file.getvalue()) # write mean empty file me write krdo

            filePath = temp_file.name # temp_file ki location

        # loader = PyPDFLoader(filePath)

        # docs = loader.load()

        # st.session_state.document = docs
        # st.session_state.doc_name = upload_file.name


        loader = PyPDFLoader(filePath)
        
        docs = loader.load()

        # Maximum 5 pages allowed
        if len(docs) > 5:
            st.error("❌ PDF must not contain more than 5 pages.")
            os.remove(filePath)
            st.stop()

        st.session_state.document = docs
        st.session_state.doc_name = upload_file.name


        

        os.remove(filePath)

        # for i, doc in enumerate(docs):
        #     st.subheader(f"page: {i+1}")
        #     st.write(doc.page_content)

        #     st.divider()

        #     st.write("Metadata:")
        #     st.write(doc.metadata)
        st.divider()

        if st.button("Clear History",use_container_width=True):
            st.session_state.history = []
            st.rerun()


parser = StrOutputParser()

# ---------------------------- Document info which available in session state

if st.session_state.document:

    document = st.session_state.document
    total_pages = len(document)

    total_characters = sum(len(doc.page_content)for doc in document) # it tell total text length 

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("pages:",total_pages)
    with col2:
        st.metric("Charactoers:",total_characters)
    with col3:
        st.metric("Messages:",len(st.session_state.history))

    st.success(f"Loaded: **{st.session_state.doc_name}**")

else:
    st.info("👈 Upload a research paper from the sidebar to begin.")


#  model prepration 

def get_document_context(document):
    context_parts = []

    for doc in document:
        p_number = (doc.metadata.get("page",0)+1)
        content = doc.page_content.strip()

        context_parts.append(f"---Page No:{p_number}----page Content:{content}")

    return "\n".join(context_parts)


    
def clean_question(question):
    return question.strip()



prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",

            """
            You are an expert AI Document Assistant.

            Your job is to answer the user's questions using the provided document context. The uploaded document can be **any type of document**, such as a research paper, CV, report, article, manual, notes, academic document, business document, or other PDF.

            ## Core Rules

            1. **Use the provided document context as the primary source.**

            * Answer based on the information available in the document.
            * Do not invent facts.
            * Do not assume information that is not present in the document.
            * If the answer cannot be found in the document, clearly say so.

            2. **Understand the user's question before answering.**

            * Identify the most important keywords and concepts in the question.
            * Focus on the document information that is directly relevant to those keywords.
            * Do not unnecessarily include unrelated information from the document.

            3. **Answer the question directly first.**

            * The main answer must appear separately and prominently.
            * Do not hide the main answer inside a long paragraph.
            * Put important values, facts, names, dates, numbers, results, conclusions, or other key information on separate lines when appropriate.

            ## Response Structure

            Choose the structure according to the user's question.

            For a simple factual question:

            *********************

            **[Most important answer]**

            ********************

            [Brief explanation if needed]

            For questions requiring multiple pieces of information:

            *******************

            * **Important Point:** ...
            * **Important Point:** ...
            * **Important Point:** ...

            For definitions:

            **🎯 Comparison**

            | Feature | Option 1 | Option 2 |
            | ------- | -------- | -------- |
            | ...     | ...      | ...      |

            For processes or steps:

            **🎯 Steps**

            1. **Step 1:** ...
            2. **Step 2:** ...
            3. **Step 3:** ...

            For summaries:

            **📋 Summary**

            **Main Idea:** ...

            **🔑 Key Points**

            * **Point:** ...
            * **Point:** ...
            * **Point:** ...

            **📊 Important Findings**

            * ...
            * ...

            **🏁 Conclusion**
            ...

            Only use sections that are relevant to the question. Do not force every response into the same structure.

            ## Important Information Formatting

            * **Bold the most important keywords and values.**
            * Put important information on a separate line when it improves readability.
            * Use bullet points when there are multiple related facts.
            * Use numbered lists for ordered information or procedures.
            * Use tables when a table makes comparison or structured information easier to understand.
            * Keep paragraphs short.
            * Avoid large blocks of text.
            * Use whitespace between sections.
            * Do not overuse bold formatting or emojis.

            ## Context Relevance

            When answering, prioritize information according to this order:

            1. **Directly answers the user's question**
            2. **Strongly related supporting information**
            3. **Useful context**
            4. Ignore unrelated document content

            Do not summarize the entire document unless the user specifically asks for a summary.

            ## Important Keyword Awareness

            Pay close attention to the important keywords in the user's question.

            For example, if the user asks:

            "What is the author's main contribution?"

            Focus on information related to:

            * **author**
            * **main contribution**
            * **contribution**

            Do not unnecessarily provide unrelated information such as background, references, or other sections.

            If the user asks:

            "What is the total cost?"

            Focus specifically on:

            * **total**
            * **cost**
            * relevant numbers or financial information

            If the user asks about a specific person, date, section, result, concept, or value, prioritize the document content related to that entity or concept.

            ## Accuracy

            If multiple pieces of information could answer the question, select the information most directly relevant to the user's wording.

            If the document contains conflicting information:

            * Mention the conflict clearly.
            * Do not choose one value without explaining the discrepancy.

            If the requested information is not present:

            **❗ Not Found**

            The requested information could not be found in the provided document.

            Do not fabricate an answer.

            ## Final Style

            Your response should feel like a **professional, intelligent document assistant**.

            The user should be able to quickly scan the response and immediately identify the most important information.

            Prefer:

            **Clear structure → Important information → Short explanation → Relevant supporting details**

            Avoid:

            **Long paragraphs → Unnecessary information → Repetition → Unsupported assumptions**


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




if st.session_state.document:
    document = st.session_state.document
    history = st.session_state.history



    def clean_question(question):
        return question.strip()

    def get_context(_):
        return get_document_context(document)

    def get_history(_):
        return history

    question_chain = RunnableLambda(clean_question)
    context_chain = RunnableLambda(get_context)
    history_chain = RunnableLambda(get_history)


# Yani parallel ke andar jo branches hain unka return type kuch bhi ho 
# sakta hai, lekin parallel un sab ko ek dictionary me wrap kar deta
#  hai.

    parallel = RunnableParallel({  # ye sab chezy yaha sy parallel process hokr prompt ko mily gi...
        "question":question_chain,
        
        "context":context_chain,

        "history":history_chain
    })


    final_chain = parallel | prompt | model | parser

# -------------Display messages--------------

for message in st.session_state.history:
    if isinstance(message,HumanMessage):
        with st.chat_message("User"):
            st.write(message.content)

    elif isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)

# ------------- input user question

user_question = st.chat_input(
    "Any Question About Document"
)

# -----------given input process

if user_question:

    if not st.session_state.document:
        st.warning("Please Upload Document")
        st.stop()


    #  asal ye chat boat k sath sath message show krta hy opr wala jab har message k bad streamlit 
    # opr sy nichy tak code chlta hy to wo session sy history msg disply krta hy

    with st.chat_message("User"):
        st.write(user_question)

    # with st.chat_message("AI"):
    #     with st.spinner("Processing"):

    #         model_reply = final_chain.invoke(user_question)

    #         st.write(model_reply)

    with st.chat_message("AI"):

        message_placeholder = st.empty()

        response = ""

        try:

            for chunk in final_chain.stream(user_question): # instead invoke we use stream taky rply ko attractive bna sky 
                response += chunk
                message_placeholder.markdown(response)


        except Exception as e:
            st.error("⚠️ Unable to connect to the AI model. Please try again.")
            st.stop()


        

        model_reply = response

# ----------- History update

    st.session_state.history.append(HumanMessage(content=user_question))

    st.session_state.history.append(AIMessage(content=model_reply))


    st.rerun()

