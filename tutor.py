import os
import base64
import fitz
import tempfile
import hashlib
import io
import json
import pandas as pd
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_text_splitters import CharacterTextSplitter

from pipeline.api_handler import ApiHandler

# Set page config
st.set_page_config(
    page_title="KnoWhiz Tutor",
    page_icon="frontend/images/logo_short.ico",  # Replace with the actual path to your .ico file
    layout="wide"
)

# Main content
with open("frontend/images/logo_short.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()
st.markdown(
    f"""
    <h1 style='text-align: center;'>
        <img src="data:image/png;base64,{encoded_image}" alt='icon' style='width:50px; height:50px; vertical-align: middle; margin-right: 10px;'>
        KnoWhiz Tutor
    </h1>
    """,
    unsafe_allow_html=True
)
st.subheader("Upload a document to get started.")

# Custom function to extract document objects from uploaded file
def extract_documents_from_file(file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file)
    temp_file.close()

    loader = PyMuPDFLoader(temp_file.name)

    # Load the document
    documents = loader.load()
    return documents

# Starts from Page 0
def find_pages_with_excerpts(doc, excerpts):
    pages_with_excerpts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                pages_with_excerpts.append(page_num + 1)  # Page numbers start from 1
                break  # No need to search further on this page
    return (
        pages_with_excerpts if pages_with_excerpts else [1]
    )  # Default to the first page if no excerpts are found

@st.cache_resource
def get_llm(llm_type, para):
    para = para
    api = ApiHandler(para)
    llm_basic = api.models['basic']['instance']
    llm_advance = api.models['advance']['instance']
    llm_creative = api.models['creative']['instance']
    if llm_type == 'basic':
        return llm_basic
    elif llm_type == 'advance':
        return llm_advance
    elif llm_type == 'creative':
        return llm_creative
    return llm_basic

@st.cache_resource
def get_embedding_models(embedding_model_type, para):
    para = para
    api = ApiHandler(para)
    embedding_model_default = api.embedding_models['default']['instance']
    if embedding_model_type == 'default':
        return embedding_model_default
    else:
        return embedding_model_default

def get_response(_documents, collection_name, embedding_folder):
    para = {
        'llm_source': 'openai',  # or 'anthropic'
        'temperature': 0,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env",
    }
    llm = get_llm('basic', para)
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    embeddings = get_embedding_models('default', para)

    # Define the default filenames used by FAISS when saving
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        # Load existing embeddings
        print("Loading existing embeddings...")
        db = FAISS.load_local(
            embedding_folder, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # Split the documents into chunks
        print("Creating new embeddings...")
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        texts = text_splitter.split_documents(_documents)

        # Create the vector store to use as the index
        db = FAISS.from_documents(texts, embeddings)
        # Save the embeddings to the specified folder
        db.save_local(embedding_folder)

    # Expose this index in a retriever interface
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.8}
    )

    # Create the RetrievalQA chain
    system_prompt = (
        """
        You are a patient and honest professor helping a student reading a paper.
        Use the given context to answer the question.
        If you don't know the answer, say you don't know.

        Context: ```{context}```

        For answer part, provide your detailed answer;
        For sources part, provide the
            "Direct sentences or paragraphs from the context that support 
            your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT 
            ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
        Organize final response in the following JSON format:

        ```json
        {{
            "answer": "Your concise answer and directly answer the question in easy to understand language here. In Markdown format.",
            "sources": [
                <source_1>,
                <source_2>,
                ...
                <source_n>,
            ]
        }}
        ```
        """
    )
    human_prompt = (
        """
        My question is: {input}
        Answer the question based on the context provided.
        Since I am a student with no related knowledge background, 
        please provide a concise answer and directly answer the question in easy to understand language.
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],  # input query
            "context": lambda x: format_docs(x["context"]),  # context
        }
        | prompt  # format query and context into prompt
        | llm  # generate response
        | error_parser  # parse response
    )
    # Pass input query to retriever
    retrieve_docs = (lambda x: x["input"]) | retriever
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    return chain

def get_highlight_info(doc, excerpts):
    annotations = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        for excerpt in excerpts:
            text_instances = page.search_for(excerpt)
            if text_instances:
                for inst in text_instances:
                    annotations.append(
                        {
                            "page": page_num + 1,  # Page numbers start from 1
                            "x": inst.x0,
                            "y": inst.y0,
                            "width": inst.x1 - inst.x0,
                            "height": inst.y1 - inst.y0,
                            "color": "red",
                        }
                    )
    return annotations

def previous_page():
    if st.session_state.current_page > 1:
        st.session_state.current_page -=1

def next_page():
    if st.session_state.current_page < st.session_state.total_pages:
        st.session_state.current_page += 1

def close_pdf():
    st.session_state.show_pdf = False

# Reset all states
def file_changed():
    for key in st.session_state.keys():
        del st.session_state[key]

#-----------------------------------------------------------------------------------------------#
# Streamlit file uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", on_change=file_changed)

if uploaded_file is not None:
    file = uploaded_file.read()
    # Compute a hashed ID based on the PDF content
    file_hash = hashlib.md5(file).hexdigest()
    course_id = file_hash
    embedding_folder = os.path.join('embedded_content', course_id)
    if not os.path.exists('embedded_content'):
        os.makedirs('embedded_content')
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)

    with st.spinner("Processing file..."):
        documents = extract_documents_from_file(file)
        st.session_state.doc = fitz.open(stream=io.BytesIO(file), filetype="pdf")
        st.session_state.total_pages = len(st.session_state.doc)

    if documents:
        qa_chain = get_response(documents, collection_name=course_id, embedding_folder=embedding_folder)
        # First run
        if "chat_history" not in st.session_state: 
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! How can I assist you today? "}
            ]
            st.session_state.show_pdf = True  # Always show PDF
            st.session_state.current_page = 1  # Start at the first page

        # Set up the columns with equal width
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### PDF Preview")

            # Navigation buttons
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("Previous Page"):
                    previous_page()
            with nav_col2:
                st.write(
                    f"Page {st.session_state.current_page} of {st.session_state.total_pages}"
                )
            with nav_col3:
                if st.button("Next Page"):
                    next_page()

            # Display the PDF viewer with the current page
            pdf_viewer(
                file,
                width=700,
                height=800,
                annotations=st.session_state.get('annotations', []),
                pages_to_render=[st.session_state.current_page],
            )

        with col2:
            # After every rerun, display chat history (assistant and client)
            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).write(msg["content"])

            # If there has been a user input, update chat_history, invoke model and get response
            if user_input := st.chat_input("Your message"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                st.chat_message("user").write(user_input)

                with st.spinner("Generating response..."):
                    try:
                        parsed_result = qa_chain.invoke({"input": user_input})
                        print("Result: ", parsed_result)
                        result = parsed_result['answer']
                        answer = result['answer']
                        sources = result['sources']

                        try:
                            # Check whether sources is a list of strings
                            if not all(isinstance(source, str) for source in sources):
                                raise ValueError("Sources must be a list of strings.")
                            sources = list(sources)
                        except:
                            sources = []

                        print("The content is from: ", sources)

                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                        st.chat_message("assistant").write(answer)

                        # Update the session state with new sources
                        st.session_state.sources = sources

                        # Generate annotations for highlighting
                        st.session_state.annotations = get_highlight_info(st.session_state.doc, st.session_state.sources)

                        # Find the pages with excerpts
                        pages_with_excerpts = find_pages_with_excerpts(st.session_state.doc, st.session_state.sources)

                        if pages_with_excerpts:
                            # Automatically navigate to the first page with an excerpt
                            st.session_state.current_page = pages_with_excerpts[0]
                        else:
                            st.session_state.current_page = 1  # Default to first page

                    except json.JSONDecodeError:
                        st.error(
                            "There was an error parsing the response. Please try again."
                        )
