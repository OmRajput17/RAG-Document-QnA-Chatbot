#RAG QnA Conversation with pdf with Conversation history
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

## Loading the Environment Variables
from dotenv import load_dotenv
load_dotenv()

def get_secret(key: str, default: str = None):
    # Prefer st.secrets if available, else fall back to os.environ
    return st.secrets.get(key, os.getenv(key, default))

# Load from Streamlit secrets
os.environ["LANGCHAIN_PROJECT"] = get_secret("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = get_secret("LANGCHAIN_API_KEY")
os.environ["HF_TOKEN"] = get_secret("HF_TOKEN")
groq_api_key = get_secret("GROQ_API_KEY")

### Embeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## set up Streamlit 
st.title("Chat with PDF")
st.write("Upload Pdf's and chat with their content")

## Check if groq api key is provided
if groq_api_key:
    llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

    ## chat interface

    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs = {"k":3})    

        ### Setting prompt

        ## Contrextual Prompt
        contextualize_q_system_prompt=( 
            "You are a query reformulation assistant. "
            "Your task is to take the latest user question and, if it depends on prior conversation, "
            "rewrite it into a complete, standalone query that can be understood without needing the chat history. "
            "Preserve all details, entities, and references from the conversation. "
            "If the question is already clear and standalone, return it unchanged. "
            "Do NOT answer the question. Only return the reformulated query."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        ### Creating History Aware retriever

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


        ### Prompt for how LLM will act like
        # Answer question
        system_prompt = (
            "You are an AI assistant that answers user questions using the retrieved document context as your primary source. "
            "Respond naturally, as if explaining directly to the user, without mentioning the phrases "
            "'based on the context' or 'according to the documents.'\n\n"

            "Rules:\n"
            "1. Use the provided context to answer the question directly, in natural language.\n"
            "2. If the context is incomplete, you may add general knowledge, but do not explicitly state this distinction.\n"
            "3. If the answer cannot be found at all, say: 'I could not find this information in the documents.'\n"
            "4. Keep answers concise, clear, and structured. Use bullet points or numbers when listing items.\n"
            "5. Do not copy the context verbatim — rephrase and explain clearly.\n\n"

            "Retrieved Context:\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        ## Creating RAG Chain
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


        # Define summarization prompt
        summary_prompt = PromptTemplate.from_template(
            "Summarize the following conversation briefly, keeping key facts and context:\n\n{conversation}"
        )

        ### Summarize Coonversation
        def summarize_conversation(llm, messages):
            """Convert old chat history into a summary string using the LLM."""
            chain = LLMChain(llm=llm, prompt=summary_prompt)
            text = "\n".join([f"{m.type}: {m.content}" for m in messages])
            summary = chain.run({"conversation": text})
            return summary.strip()
        

        ## Get Session History
        def get_session_history(session:str, llm)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            history =  st.session_state.store[session_id]

            # If history grows too long, summarize older messages
            if len(history.messages) > 10:  # > 5 Q&A pairs
                # Summarize everything except the last 4 messages (2 turns)
                old_messages = history.messages[:-4]
                summary_text = summarize_conversation(llm, old_messages)

                # Keep last 4 messages + add summary as system message
                history.messages = history.messages[-4:]
                history.add_message({
                    "type": "system",
                    "content": f"Summary of earlier conversation: {summary_text}"
                })

            return history
        

        ### Creating Conversational RAG Chain
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: get_session_history(session_id, llm),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User Input using chat_input
        user_input = st.chat_input("Your Question:")

        if user_input:
            session_history = get_session_history(session_id, llm=llm)

            # Invoke RAG chain
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            # Display all messages in chat style
            for msg in session_history.messages:
                role = msg.type  # 'human' or 'assistant'
                content = msg.content
                with st.chat_message(role):
                    st.markdown(content)
else:
    st.warning("Please enter the GRoq API Key")
