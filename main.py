import streamlit as st
import uuid
from pathlib import Path
from langdetect import detect
from langchain.schema import HumanMessage, AIMessage
from utils import (StreamHandler, VectorStore, PromptManager, ModelManager, ImageDescriptor, load_embedding,
                   load_vectorstore, load_chat_history)
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.tracers import ConsoleCallbackHandler
import os


st.set_page_config(page_title='Your Enterprise Sidekick', page_icon='🚀')

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I'm an AI assistant. How can I help you?")]

# Initialize
with st.sidebar:
    username = "firstlogic"
    chat_history = load_chat_history(username)
    embedding = load_embedding()
    vectorstore = load_vectorstore(embedding, username)

# Show a custom welcome text or the default text
try:
    st.markdown(Path(f"./customizations/welcome/{username}.md").read_text())
except FileNotFoundError:
    st.markdown(Path('./customizations/welcome/default.md').read_text())

with st.sidebar:
    try:
        st.image(f"./customizations/logo/{username}.svg", use_column_width="always")
    except FileNotFoundError:
        try:
            st.image(f"./customizations/logo/{username}.png", use_column_width="always")
        except FileNotFoundError:
            st.image('./customizations/logo/default.svg', use_column_width="always")

# Options panel
with st.sidebar:
    # Chat history settings
    disable_chat_history = st.checkbox('Disable chat history')
    top_k_history = st.slider('Number of chat history messages', 1, 50, 5, disabled=disable_chat_history)
    memory = ModelManager.load_memory(chat_history, top_k_history if not disable_chat_history else 0)
    delete_history = st.button('Delete chat history', disabled=disable_chat_history)
    if delete_history:
        with st.spinner('Deleting chat history...'):
            chat_history.clear()
            st.session_state.messages = [AIMessage(content="Hello! I'm an AI assistant. How can I help you?")]

    # Vector store settings
    disable_vector_store = st.toggle('Disable vector store')
    top_k_vectorstore = st.slider('Number of vector store documents', 1, 50, 5, disabled=disable_vector_store)
    strategy = st.selectbox('RAG strategy', ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion'), help='Retrieval strategy for vector store', disabled=disable_vector_store)

    custom_prompt_text = ''
    custom_prompt_index = 0
    try:
        custom_prompt_text = Path(f"./customizations/prompt/{username}.txt").read_text()
        custom_prompt_index = 2
    except FileNotFoundError:
        custom_prompt_text = Path(f"./customizations/prompt/default.txt").read_text()

    prompt_type = st.selectbox('System prompt', ('Short results', 'Extended results', 'Custom'), index=custom_prompt_index)
    custom_prompt = st.text_area('Custom prompt', custom_prompt_text, help='Enter a custom prompt for the AI assistant', disabled=(prompt_type != 'Custom'))

# Include the upload form for new data to be Vectorized
with st.sidebar:
   uploaded_files = st.file_uploader('Upload files', type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
   upload = st.button('Upload files')
   if upload and uploaded_files:
       vector_store = VectorStore(vectorstore)
       with st.spinner('Vectorizing the document...'):
           vector_store.vectorize_text(uploaded_files)

# Include the upload form for URLs to be Vectorized
with st.sidebar:
   urls = st.text_area('Enter URLs (comma-separated)', help='Enter URLs to load data from').split(',')
   upload = st.button('Upload from URLs')
   if upload and urls:
       vector_store = VectorStore(vectorstore)
       with st.spinner('Vectorizing the URLs...'):
           vector_store.vectorize_url(urls)

# Drop the vector data and start from scratch
if 'delete_option' in st.secrets and st.secrets.delete_option.get(username, 'False') == 'True':
   with st.sidebar:
       st.caption('Delete vector data')
       submitted = st.button('Delete vector data')
       if submitted:
           with st.spinner('Deleting vector data...'):
               vectorstore.clear()
               memory.clear()
               st.session_state.messages = [AIMessage(content="Hello! I'm an AI assistant. How can I help you?")]

# Draw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
   st.chat_message(message.type).markdown(message.content)

# Get a prompt from the user
question = st.chat_input('Enter your question')

with st.sidebar:
   st.divider()
   picture = st.camera_input('Take a picture')
   if picture:
       response = ImageDescriptor.describe_image(picture.getvalue())
       picture_desc = response.choices[0].message.content
       question = picture_desc

# Upload image button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width="always")
    uploaded_file_images = uploaded_file
    if uploaded_file_images:
        response = ImageDescriptor.describe_image(uploaded_file_images.getvalue())
        picture_desc = response.choices[0].message.content
        question = picture_desc

if question:
   st.session_state.messages.append(HumanMessage(content=question))

   # Detect the language of the question
   language = detect(question) # Get the language code
   st.write(f'Detected language: {language}')

   # Draw the prompt on the page
   with st.chat_message('human'):
       st.markdown(question)

   # Get model, retriever
   model = ModelManager.load_model()
   retriever = ModelManager.load_retriever(vectorstore, top_k_vectorstore)

   # RAG Strategy
   relevant_documents = []
   if not disable_vector_store:
       if strategy == 'Basic Retrieval':
           # Basic naive RAG
           relevant_documents = retriever.get_relevant_documents(query=question, k=top_k_vectorstore)
       if strategy == 'Maximal Marginal Relevance':
           relevant_documents = vectorstore.max_marginal_relevance_search(query=question, k=top_k_vectorstore)
       if strategy == 'Fusion':
           # Fusion: Generate new queries and retrieve most relevant documents based on that
           queries = ModelManager.generate_queries(language, model)
           fusion_queries = queries.invoke({"original_query": question})

           content = f"*Using fusion queries:*  \n"
           for fq in fusion_queries:
               content += f"📙 :orange[{fq}]  \n"
           with st.chat_message('assistant'):
               st.markdown(content)
           st.session_state.messages.append(AIMessage(content=content))

           chain = queries | retriever.map() | ModelManager.reciprocal_rank_fusion
           relevant_documents = chain.invoke({"original_query": question})

   # Get the results from Langchain
   with st.chat_message('assistant'):
       content = ''
       response_placeholder = st.empty()
       history = memory.load_memory_variables({})
       inputs = RunnableMap({
           'context': lambda x: x['context'],
           'chat_history': lambda x: x['chat_history'],
           'question': lambda x: x['question']
       })
       chain = inputs | PromptManager.get_prompt(prompt_type, language, custom_prompt) | model
       response = chain.invoke({'question': question, 'chat_history': history, 'context': relevant_documents}, config={'callbacks': [StreamHandler(response_placeholder), ConsoleCallbackHandler()]})
       content += response.content

       # Add the result to memory (without the sources)
       memory.save_context({'question': question}, {'answer': content})

       # Write the sources used
       if disable_vector_store:
           content += "\n\n*No context used.*"
       else:
           content += "\n\n*Sources used:*  \n"
           sources = []
           for doc in relevant_documents:
               if strategy == 'Fusion':
                   doc = doc[0]
               source = doc.metadata['source']
               if source not in sources:
                   content += f"📙 :orange[{os.path.basename(os.path.normpath(source))}]  \n"
                   sources.append(source)

       # Write the history used
       if disable_chat_history:
           content += "\n\n*No chat history used.*"
       else:
           content += f"\n\n*Chat history used: ({int(len(history['chat_history'])/2)} / {top_k_history})*"

       # Write the final answer without the cursor
       response_placeholder.markdown(content)
       st.session_state.messages.append(AIMessage(content=content))
