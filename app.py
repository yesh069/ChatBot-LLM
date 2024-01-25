import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import time
import warnings


DB_FAISS_PATH = 'vectorstore/db_faiss'

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain 
def retrieval_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Loading the model
@st.cache_resource
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Streamlit App
def get_chain():
    return qa_bot()

def main():
    st.title("48 Laws of Power ChatBot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("Ask Your Question"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
                st.markdown(user_input)
        
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # Call your question-answering model here
                chain = get_chain()
                result = chain({'query': user_input})
                assistant_response = result["result"]
                
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response) 
                
                with st.sidebar:
                    sources = result["source_documents"]
                    if sources:
                        st.write("Sources:")
                        for source in sources:
                            st.write(f"- {source.metadata['source']}, Page {source.metadata['page']}")
            except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    
        st.session_state.messages.append({"role": "assistant", "content": full_response})
      
        
            
  
if __name__ == "__main__":
    main()
