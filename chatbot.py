import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai import ChatMistralAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# from langchain_community.vectorstores.faiss import FAISS


api_key = "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"     

# Function to get embeddings from Hugging Face API
def get_huggingface_embeddings(api_key, text):
    url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": text
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()[0]["embedding"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

#Upload PDF Files

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("upload a pdf file and start asking questions" , type="pdf")


#Extract the text

if file is not None:
    pdf_reader = PdfReader(file)
    text =""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size = 10000,
        chunk_overlap=1500,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # generating embeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # chunk_embeddings = [get_huggingface_embeddings(api_key, chunk) for chunk in chunks]

    #creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks,embeddings)

    #get user questions
    

    # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    hf_endpoint = HuggingFaceEndpoint(
        repo_id="distilbert-base-uncased-distilled-squad",  # Replace with your actual repo ID
        task="text-generation",
        max_length=1000,
        temperature=0.7,
        top_p=0.95,
        huggingfacehub_api_token=api_key
    )
    qa_chain = load_qa_chain(hf_endpoint, chain_type="stuff")

    user_question = st.text_input("Type your questions hereeeeee")

    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        result = qa_chain({"input_documents": match, "question": user_question}, return_only_outputs=True)
        
        #output results

        # chain = load_qa_chain(llm,chain_type="stuff")
        # response = chain.run(input_documents = match, question = user_question)
        st.write(result['output_text'])
        
