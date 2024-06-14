import os
import faiss
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from secret_key import openai_key

from dotenv import load_dotenv

os.environ['OPENAI_API_KEY'] = openai_key

st.title("Article Research Tool ✍")

st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature = 0.9, max_tokens = 500)

embeddings = OpenAIEmbeddings()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading... Started...")
    data = loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter... Started...")

    docs = text_splitter.split_documents(data)

    #create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs,embeddings)

    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # Save the FAISS index
    faiss.write_index(vectorstore_openai.index, "vector_index.faiss")

    # Save the docstore and index_to_docstore_id using pickle
    metadata = {
        'docstore': vectorstore_openai.docstore,
        'index_to_docstore_id': vectorstore_openai.index_to_docstore_id,
        # Add any other attributes you need
    }

    with open("vector_store_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

query = main_placeholder.text_input("Question : ")
if query:
    if os.path.exists("vector_store_metadata.pkl"):

        vectorIndex = faiss.read_index("vector_index.faiss")

        with open("vector_store_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        vectorindex_openai = FAISS(
            index=vectorIndex,
            docstore=metadata['docstore'],
            index_to_docstore_id=metadata['index_to_docstore_id'],
            embedding_function=embeddings
        )

        retriever = vectorindex_openai.as_retriever()

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        result = chain({"question":query},return_only_outputs=True)

        #{"answer": "","Sources":[]}

        st.header("Answer: ")
        st.write(result["answer"])

        #Display Sources, if available
        sources = result.get("sources","")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)

