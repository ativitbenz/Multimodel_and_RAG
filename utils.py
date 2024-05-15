import os
import base64
import tempfile
import openai
import streamlit as st
from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

class VectorStore:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def vectorize_text(self, uploaded_files):
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                temp_dir = tempfile.TemporaryDirectory()
                file = uploaded_file
                temp_filepath = os.path.join(temp_dir.name, file.name)
                with open(temp_filepath, 'wb') as f:
                    f.write(file.getvalue())

                if uploaded_file.name.endswith('txt'):
                    file = [uploaded_file.read().decode()]
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    texts = text_splitter.create_documents(file, [{'source': uploaded_file.name}])
                    self.vectorstore.add_documents(texts)
                    st.info(f"{len(texts)} text documents loaded")

                elif uploaded_file.name.endswith('pdf'):
                    docs = []
                    loader = PyPDFLoader(temp_filepath)
                    docs.extend(loader.load())
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    pages = text_splitter.split_documents(docs)
                    self.vectorstore.add_documents(pages)
                    st.info(f"{len(pages)} PDF pages loaded")

                elif uploaded_file.name.endswith('csv'):
                    docs = []
                    loader = CSVLoader(temp_filepath, encoding="utf-8")
                    docs.extend(loader.load())
                    self.vectorstore.add_documents(docs)
                    st.info(f"{len(docs)} CSV documents loaded")

    def vectorize_url(self, urls):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                pages = text_splitter.split_documents(docs)
                self.vectorstore.add_documents(pages)
                st.info(f"{len(pages)} pages loaded from URL: {url}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

class PromptManager:
    @staticmethod
    def get_prompt(type, language, custom_prompt):
        template = ''
        if type == 'Extended results':
            template = f"""You're a helpful AI assistant tasked to answer the user's questions.
            You're friendly and you answer extensively with multiple sentences. You prefer to use bullet points to summarize.
            If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
            If you don't know the answer, just say 'I do not know the answer'.
            Use the following context to answer the question:
            {{context}}
            Use the following chat history to answer the question:
            {{chat_history}}
            Question:
            {{question}}
            Answer in {language}:"""

        elif type == 'Short results':
            template = f"""You're a helpful AI assistant tasked to answer the user's questions.
            You answer in an exceptionally brief way.
            If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
            If you don't know the answer, just say 'I do not know the answer'.
            Use the following context to answer the question:
            {{context}}
            Use the following chat history to answer the question:
            {{chat_history}}
            Question:
            {{question}}
            Answer in {language}:"""

        elif type == 'Custom':
            template = custom_prompt

        return ChatPromptTemplate.from_messages([("system", template)])

class ModelManager:
    @staticmethod
    def load_model():
        return ChatOpenAI(temperature=0.3, model='gpt-4o', streaming=True, verbose=True)

    @staticmethod
    def load_retriever(vectorstore, top_k_vectorstore):
        return vectorstore.as_retriever(search_kwargs={"k": top_k_vectorstore})

    @staticmethod
    def load_memory(chat_history, top_k_history):
        return ConversationBufferWindowMemory(
            chat_memory=chat_history,
            return_messages=True,
            k=top_k_history,
            memory_key="chat_history",
            input_key="question",
            output_key='answer',
        )

    @staticmethod
    def generate_queries(language, model):
        prompt = f"""You are a helpful assistant that generates multiple search queries based on a single input query in language {language}.
        Generate multiple search queries related to: {{original_query}}
        OUTPUT (4 queries):"""
        return ChatPromptTemplate.from_messages([("system", prompt)]) | model | StrOutputParser() | (lambda x: x.split("\n"))

    @staticmethod
    def reciprocal_rank_fusion(results: list[list], k=60):
        from langchain.load import dumps, loads
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0 
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
        return reranked_results

class ImageDescriptor:
    @staticmethod
    def describe_image(image_bin):
        image_base64 = base64.b64encode(image_bin).decode()
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe the image in detail"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=4096,
        )
        return response

@st.cache_resource()
def load_embedding():
    return OpenAIEmbeddings()

@st.cache_resource()
def load_vectorstore(_embedding, username):
    print(f'load vectorstore for user name : {username}')
    return AstraDB(
        embedding=_embedding,
        collection_name=st.secrets["COLLECTION_NAME"],
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=os.environ["ASTRA_ENDPOINT"],
    )

@st.cache_resource()
def load_chat_history(username):
    return AstraDBChatMessageHistory(
        session_id=f"{username}_{st.session_state.session_id}",
        api_endpoint=os.environ["ASTRA_ENDPOINT"],
        token=st.secrets["ASTRA_TOKEN"],
    )
