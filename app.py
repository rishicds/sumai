import os
from pathlib import Path
import streamlit as st
from io import StringIO
import time
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from modules.database import KnowledgeBase
from modules.main import Chatbase
from modules import EMBEDDINGS_BGE_BASE
from modules.pdftools import extract_and_clean_pdf_text as cleanPDF
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from pydub import AudioSegment
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from groq import Groq
from google.cloud import texttospeech

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

kb = KnowledgeBase()
home_md = Path("/home/ubuntu/workspace/readme.md").read_text()

def ask(query, persist_dir):
    Query = "Don't justify your answers. Don't give information not mentioned in the CONTEXT INFORMATION " + 'query="{}"'.format(query)
    Chat = Chatbase(EMBEDDINGS_BGE_BASE)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=EMBEDDINGS_BGE_BASE)
    with get_openai_callback() as cb:
        resp = {
        "response": Chat.chat(Query, vectordb),
        "prompt": cb.prompt_tokens,
        "completion": cb.completion_tokens,
        "cost": cb.total_cost
        }
        return resp

def embed_text(text_path, persist_path):
    cb = Chatbase()
    docs = cb.load_text(text_path)
    splitted_docs = cb.split_docs(docs)
    embed = cb.embed(persist_path, splitted_docs)
    return True

# Custom header and footer
st.markdown("""
    <style>
    @keyframes typewriter {
        from {
            width: 0;
        }
        to {
            width: 100%;
        }
    }

    .animated-header, .footer {
        overflow: hidden;
        white-space: nowrap;
        animation: typewriter 2s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 normal both;
        text-align: center;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        padding: 10px;
        font-size: 16px;
        color: #2c3e50;
    }
    </style>
    <div class="footer">Built with 💖 by Rishi © 2024</div>
    """, unsafe_allow_html=True)

sidebar = st.sidebar
sidebar.header('⚡IntelliDocChat', anchor=False, divider='rainbow')
sidebar.caption('Made with :heart: by Siddhartha')
selection = sidebar.selectbox("Select an Option", ['Home', 'Chat', 'Knowledge Base'])

if selection == 'Home':
    st.header("👋Hey, Welcome to IntelliDocChat", divider=None, anchor=False)
    st.subheader('AI Powered Document Chat Application', anchor=False, help=None, divider='rainbow')
    st.markdown(home_md)

elif selection == "Chat":
    base = []
    for i in kb.get_all_entries():
        base.append(i[0])
    kb_name = sidebar.selectbox("Select a Knowledge Base", base)
    try:
        persist_path = kb.get_entry_by_name(kb_name)[1]
        st.header("Welcome to IntelliDocChat👋", divider='rainbow', anchor=False)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(name=message["role"], avatar=message['avatar']):
                st.markdown(message["content"])
        if prompt := st.chat_input("What's Your Query Today?",):
            with st.chat_message("user", avatar="https://i.imgur.com/hjaMekQs.png"):
                st.markdown(prompt)
            with st.chat_message("ai", avatar="https://i.imgur.com/YbXPMFks.jpeg"):
                resp = ask(prompt, persist_path)
                st.markdown(resp['response'])
                st.metric(label="Prompt/Completion", value="{}/{} Tokens".format(resp['prompt'], resp['completion']), delta="Total Cost: ${}".format(resp['cost']))
            st.session_state.messages.append({"role": "User", "content": prompt, "avatar": "https://i.imgur.com/hjaMekQs.png"})
            st.session_state.messages.append({"role": "Assistant", "content": resp['response'], "avatar": "https://i.imgur.com/YbXPMFks.jpeg"})
    except:
        st.warning("No Knowledge Base Available, Try Creating One")

elif selection == "Knowledge Base":
    st.header("📚Knowledge Base", divider=None, help="🚀Upload Your File to Process Further, After a successful upload, the process of converting your text into embeddings will be initiated!", anchor=False)
    name = st.text_input("Name*", placeholder="Enter the Filename", help="Enter the name of this document, so that you can use it later")
    if name != "":
        if kb.entry_exists(name):
            st.error("Knowledge Base with the name {} already exists".format(name))
        else:
            File = st.file_uploader("📄Upload Your Document", type=['pdf',])
            if File is not None:
                save_folder = '/home/ubuntu/workspace/tmp'
                save_path = Path(save_folder, File.name)
                with open(save_path, mode='wb') as w:
                    w.write(File.getvalue())
                if save_path.exists():
                    with st.status("Processing Your Data...", expanded=True) as status:
                        st.write("Cleaning Your Data...")
                        text = cleanPDF(save_path)
                        st.write("Writing temporary files...")
                        text_path = "{}.txt".format(save_path)
                        with open(text_path, "w") as f:
                            f.write(text)
                        st.write("Generating Embeddings...")
                        persist_path = "/home/ubuntu/workspace/knowledge_base/{}".format(name)
                        embed_text(text_path, persist_path)
                        st.write("Removing temporary Files...")
                        os.remove(save_path)
                        os.remove(text_path)
                        st.write("Indexing Knowledge Base...")
                        kb.insert_entry(name, persist_path)
                        status.update(label="Knowledge base is now indexed and ready to use.", state="complete", expanded=False)

elif selection == "Study Assistant":
    st.markdown("""
        <style>
        @keyframes typewriter {
            from {
                width: 0;
            }
            to {
                width: 100%;
            }
        }

        .animated-header {
            font-size: 3vw;
            overflow: hidden;
            white-space: nowrap;
            animation: typewriter 2s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 normal both;
            text-align: center;
        }
        </style>
        <h3 class="animated-header">Upload Files Using the sidebar.</h3>
    """, unsafe_allow_html=True)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")
    prompt = ChatPromptTemplate.from_template(
        """Answer the questions based on the provided context and also from the internet if the context isn't present.
        Please provide the most accurate and long detailed responses based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "text_splitter" not in st.session_state:
        st.session_state.text_splitter = None
    if "final_documents" not in st.session_state:
        st.session_state.final_documents = None

    def process_uploaded_files(uploaded_files):
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        documents = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        return documents

    def get_youtube_transcripts(video_urls):
        transcript_texts = []
        for video_url in video_urls:
            video_id = video_url.split("v=")[-1]
            transcripts = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([t['text'] for t in transcripts])
            transcript_texts.append(transcript_text)
        return transcript_texts

    uploaded_files = sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    video_urls = sidebar.text_area("Enter YouTube Video URLs (one per line)", height=100)
    video_urls = video_urls.split("\n") if video_urls else []

    if sidebar.button("Process Files and Videos"):
        with st.spinner("Processing..."):
            documents = []
            if uploaded_files:
                documents.extend(process_uploaded_files(uploaded_files))
            if video_urls:
                transcript_texts = get_youtube_transcripts(video_urls)
                documents.extend([Document(page_content=transcript) for transcript in transcript_texts])
            st.session_state.docs = documents

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"))

            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    if st.session_state.vectors:
        def generate_answer(query):
            doc_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(doc_chain, st.session_state.vectors.as_retriever())
            result = chain({"input": query})
            return result["output_text"]

        query = st.text_area("Enter your question", height=100)

        if st.button("Submit"):
            with st.spinner("Generating answer..."):
                answer = generate_answer(query)
                st.text_area("Answer", value=answer, height=300)
