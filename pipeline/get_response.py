import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# GraphRAG imports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

from streamlit_float import *

from pipeline.config import load_config
from pipeline.utils import (
    tiktoken,
    truncate_chat_history,
    get_llm,
    get_embedding_models,
    translate_content
)
from pipeline.doc_processor import (
    generate_embedding,
)
from pipeline.sources_retrieval import (
    get_response_source,
)
from pipeline.utils import (
    detect_language
)


def tutor_agent(mode, _doc, _documents, user_input, chat_history, embedding_folder):
    """
    Taking the user input, documents, and chat history, generate a response and sources.
    If user_input is None, generates the initial welcome message.
    """
    # Handle initial welcome message when chat history is empty
    if not chat_history:
        try:
            # Try to load existing document summary
            documents_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
            with open(documents_summary_path, "r") as f:
                initial_message = f.read()
        except FileNotFoundError:
            initial_message = "Hello! How can I assist you today?"

        answer = initial_message
        # Translate the initial message to the selected language
        answer = translate_content(
            content=answer,
            target_lang=st.session_state.language
        )
        sources = []
            
        return answer, sources

    # Regular chat flow
    # Refine user input
    refined_user_input = get_query_helper(user_input, chat_history, embedding_folder)
    # Get response
    answer = get_response(mode, _doc, _documents, refined_user_input, chat_history, embedding_folder)
    # Get sources
    sources = get_response_source(_doc, _documents, refined_user_input, answer, chat_history, embedding_folder)

    answer = f"""Are you asking: **{user_input}**
    """ + "\n" + answer

    # Translate the answer to the selected language
    answer = translate_content(
        content=answer,
        target_lang=st.session_state.language
    )
    return answer, sources


def get_response(mode, _doc, _documents, user_input, chat_history, embedding_folder):
    # TEST
    print("Current language:", st.session_state.language)

    if mode == 'Professor':
        try:
            answer = get_GraphRAG_global_response(_doc,_documents, user_input, chat_history, embedding_folder)
            return answer
        except Exception as e:
            print("Error getting response from GraphRAG:", e)

    config = load_config()
    para = config['llm']
    llm = get_llm('advance', para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    embeddings = get_embedding_models('default', para)

    # Check if all necessary files exist to load the embeddings
    generate_embedding(_documents, embedding_folder)

    # Load existing embeddings
    print("Loading existing embeddings...")
    db = FAISS.load_local(
        embedding_folder, embeddings, allow_dangerous_deserialization=True
    )

    # Expose this index in a retriever interface
    # retriever = db.as_retriever(
    #     search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.8}
    # )
    config = load_config()
    retriever = db.as_retriever(search_kwargs={"k": config['retriever']['k']})

    # Create the RetrievalQA chain
    system_prompt = (
        """
        You are a patient and honest professor helping a student reading a paper.
        Use the given context to answer the question.
        If you don't know the answer, say you don't know.
        Context: ```{context}```
        If the concept can be better explained by formulas, use LaTeX syntax in markdown
        For inline formulas, use single dollar sign: $a/b = c/d$
        FOr block formulas, use double dollar sign:
        $$
        \frac{{a}}{{b}} = \frac{{c}}{{d}}
        $$
        """
    )
    human_prompt = (
        """
        Our previous conversation is: {chat_history}
        This time my query is: {input}
        Answer the question based on the context provided.
        Since I am a student with no related knowledge background, 
        provide a concise answer and directly answer the question in easy to understand language.
        Use markdown syntax for bold formatting to highlight important points or words.
        Use emojis when suitable to make the answer more engaging and interesting.
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
            "chat_history": lambda x: x["chat_history"],  # chat history
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
    parsed_result = chain.invoke({"input": user_input, "chat_history": truncate_chat_history(chat_history)})
    answer = parsed_result['answer']
    return answer


def get_GraphRAG_global_response(_doc, _documents, user_input, chat_history, embedding_folder):
    # Chat history and user input
    chat_history_text = truncate_chat_history(chat_history)
    user_input_text = str(user_input)

    # Search for the documents in the GraphRAG embedding
    try:
        load_dotenv(".env")
    except Exception as e:
        print("Error loading .env file:", e)
    api_key = os.getenv("GRAPHRAG_API_KEY")
    llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
    api_base = os.getenv("GRAPHRAG_API_BASE")
    api_version = os.getenv("GRAPHRAG_API_VERSION")

    # print("api_key", api_key)

    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        model=llm_model,
        api_type=OpenaiApiType.AzureOpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )
    token_encoder = tiktoken.encoding_for_model(llm_model)

    INPUT_DIR = os.path.join(embedding_folder, "GraphRAG/output")
    COMMUNITY_TABLE = "create_final_communities"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
    COMMUNITY_LEVEL = 2
    community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    communities = read_indexer_communities(community_df, entity_df, report_df)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    print(f"Total report count: {len(report_df)}")
    print(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )
    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    answer = search_engine.search(
        f"""
        You are a patient and honest professor helping a student reading a paper.
        The student asked the following question:
        ```{user_input_text}```
        Use the given context to answer the question.
        Previous conversation history:
        ```{chat_history_text}```
        """,
    )

    return answer.response


def get_query_helper(user_input, chat_history, embedding_folder):
    # If we have "documents_summary" in the embedding folder, we can use it to speed up the search
    documents_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
    if os.path.exists(documents_summary_path):
        with open(documents_summary_path, "r") as f:
            documents_summary = f.read()
    else:
        documents_summary = " "

    # Load languages from config
    config = load_config()
    llm = get_llm('basic', config['llm'])
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    system_prompt = (
        """
        You are a educational professor helping a student reading a document {context}.
        The goals are:
        1. to ask questions in a better way to make sure it’s optimized to query a Vector Database for RAG (Retrieval Augmented Generation).
        2. to identify the question is about local or global context of the document.

        Organize final response in the following JSON format:
        ```json
        {{
            "question": "<question rephrased in a better way to make sure it’s optimized to query a Vector Database for RAG (Retrieval Augmented Generation)>",
            "question_type": "<local/global>",
        }}
        ```
        """
    )
    human_prompt = (
        """
        Previous conversation history:
        ```{chat_history}```
        The student asked the following question:
        ```{input}```
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    chain = prompt | llm | error_parser
    parsed_result = chain.invoke({"input": user_input,
                                  "context": documents_summary,
                                  "chat_history": truncate_chat_history(chat_history)})
    question = parsed_result['question']
    question_type = parsed_result['question_type']
    language = detect_language(user_input)
    print("language detected:", language)

    st.session_state.language = language
    return question