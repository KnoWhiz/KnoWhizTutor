import os
import base64
import fitz
import tempfile
import hashlib
import io
import json
import pprint
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import streamlit_nested_layout
from streamlit_float import *

from pipeline.get_response import get_response
from pipeline.get_response import get_response_source
from pipeline.images_understanding import get_relevant_images, display_relevant_images, extract_images_with_context, save_images_temp


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
    <h2 style='text-align: center;'>
        <img src="data:image/png;base64,{encoded_image}" alt='icon' style='width:50px; height:50px; vertical-align: middle; margin-right: 10px;'>
        KnoWhiz Tutor
    </h2>
    """,
    unsafe_allow_html=True
)
st.subheader("Upload a document to get started.")


# Init float function for chat_input textbox
float_init(theme=True, include_unstable_primary=False)


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
                pages_with_excerpts.append(page_num)
                break  # No need to search further on this page
    return (
        pages_with_excerpts if pages_with_excerpts else [0]
    )  # Default to the first page if no excerpts are found


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
                            "page": page_num + 1,
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
        

def chat_content():
    st.session_state.chat_history.append(
        {"role": "user", "content": st.session_state.user_input}
    )


#-----------------------------------------------------------------------------------------------#
learner_avatar = "frontend/images/learner.svg"
tutor_avatar = "frontend/images/tutor.svg"


# Streamlit file uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", on_change=file_changed)


if __name__ == "__main__" and uploaded_file is not None:
    file_size = uploaded_file.size
    max_file_size = 10 * 1024 * 1024  # 10 MB

    if file_size > max_file_size:
        st.error("File size exceeds the 10 MB limit. Please upload a smaller file.")
    else:
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
            qa_chain = get_response(documents, embedding_folder=embedding_folder)
            qa_source_chain = get_response_source(documents, embedding_folder=embedding_folder)
            # First run
            if "chat_history" not in st.session_state: 
                st.session_state.chat_history = [
                    {"role": "assistant", "content": "Hello! How can I assist you today? "}
                ]
                st.session_state.show_chat_border = False
            else:
                st.session_state.show_chat_border = True

        outer_columns = st.columns([1,1])
    
        with outer_columns[1]:            
            with st.container(border=st.session_state.show_chat_border, height=800):
                with st.container():
                    st.chat_input(key='user_input', on_submit=chat_content) 
                    button_b_pos = "2.2rem"
                    button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
                    float_parent(css=button_css)
                # After every rerun, display chat history (assistant and client)
                for msg in st.session_state.chat_history:
                    avatar = learner_avatar if msg["role"] == "user" else tutor_avatar
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(msg["content"])
                # If there has been a user input, update chat_history, invoke model and get response
                if user_input := st.session_state.user_input:  
                    with st.spinner("Generating response..."):
                        try:
                            # Get the response from the model
                            import pprint

                            # Assuming this is inside your function where you get the response
                            parsed_result = qa_chain.invoke({"input": user_input})
                            print("qa_chain: ")
                            pprint.pprint(parsed_result)
                            answer = parsed_result['answer']

                            # Get sources
                            parsed_result = qa_source_chain.invoke({"input": user_input})
                            print("qa_source_chain: ")
                            pprint.pprint(parsed_result)
                            sources = parsed_result['answer']['sources']

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
                            # st.chat_message("assistant").write(answer)
                            with st.chat_message("assistant", avatar=tutor_avatar):
                                st.write(answer)

                            # Update the session state with new sources
                            st.session_state.sources = sources

                            # Set a flag to indicate chat interaction has occurred
                            st.session_state.chat_occurred = True

                        except json.JSONDecodeError:
                            st.error(
                                "There was an error parsing the response. Please try again."
                            )

                    # Highlight PDF excerpts
                    if file and st.session_state.get("chat_occurred", False):
                        doc = st.session_state.doc
                        
                        # Find the page numbers containing the excerpts
                        pages_with_excerpts = find_pages_with_excerpts(doc, sources)

                        if "current_page" not in st.session_state:
                            st.session_state.current_page = pages_with_excerpts[0]+1

                        if 'pages_with_exerpts' not in st.session_state:
                            st.session_state.pages_with_excerpts = pages_with_excerpts

                        # Get annotations with correct coordinates
                        st.session_state.annotations = get_highlight_info(doc, st.session_state.sources)
                        
                        # Find the first page with excerpts
                        if st.session_state.annotations:
                            st.session_state.current_page = min(annotation["page"] for annotation in st.session_state.annotations)
                        
        with outer_columns[0]:
            if "current_page" not in st.session_state:
                st.session_state.current_page = 1
            if "annotations" not in st.session_state:
                st.session_state.annotations = []
            # PDF display section
            # st.markdown("### PDF Preview")
            
            # Display the PDF viewer
            pdf_viewer(
                file,
                width=700,
                height=800,
                annotations=st.session_state.annotations,
                pages_to_render=[st.session_state.current_page],
                render_text=True,
            )
            # Navigation
            col1, col2, col3, col4 = st.columns([8, 4, 3, 3],vertical_alignment='center')
            with col1:
                st.button("←", on_click=previous_page)
            with col2:
                st.write(
                    f"Page {st.session_state.current_page} of {st.session_state.total_pages}"
                )
            # with col3:
            #     st.button("Next Page", on_click=next_page,use_container_width=True)
            with col4:
                # st.button("Close File", on_click=close_pdf,use_container_width=True)     
                st.button("→", on_click=next_page)