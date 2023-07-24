import os
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
import streamlit as st
from openai.error import OpenAIError
from langchain.vectorstores import FAISS
from pypdf import PdfReader
from io import BytesIO
import re
from hashlib import md5
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
import docx


persist_directory = 'persistent_data'
csv_path = 'c1.csv'

def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key
    
def sidebar():
    with st.sidebar:
        if api_key_input:
            set_openai_api_key(api_key_input)
            
        st.markdown(
            "# How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) on the right🔑\n" 
            "2. Upload a pdf, docx or txt version resume📄\n"
            "3. Enter your job preference if you want💬\n"
        )
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "JobGPT allows you to match the best job with your resume "
            "and preference. "
        )
        st.markdown(
            "JobGPT is an innovative tool designed to seamlessly connect your unique qualifications, as reflected in your resume, with the ideal job that matches your preferences. It currently utilizes a self-sourced dataset, painstakingly scraped from Capital One, ensuring an update frequency of every two weeks or even more rapidly whenever possible. While this product is still under development, we actively encourage and welcome suggestions for improvement, as we strive to enhance your job-matching experience."
            "Feel free to improve this product https://github.com/yuchuheng626/JobGPT"
        )
        st.markdown("Made by [Chuheng Yu](https://www.linkedin.com/in/chuheng-yu99/)")
        st.markdown("---")

@st.cache_data()
def parse_pdf(file: BytesIO) -> list[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output

def remove_null_bytes(file_path, encoding='utf-8'):
    with open(file_path, 'rb') as f:
        content = f.read()
    content = content.replace(b'\x00', b'') 
    with open(file_path, 'wb') as f:
        f.write(content)

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def clear_submit():
    st.session_state["submit"] = False

st.set_page_config(page_title="JobGPT", page_icon="👨‍💻", layout="wide")
st.header("JobGPT👨‍💻")

api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.", 
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )

if api_key_input:
    set_openai_api_key(api_key_input)

uploaded_file = st.file_uploader(
    "Upload a pdf or a docx version of your resume",
    type=["pdf", "docx"],
    on_change=clear_submit,
)

resume = st.text_input(label="Or paste your resume here", placeholder="Paste your resume here")

sidebar()

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.name.endswith(".docx"):
            text = getText(uploaded_file)
        else:
            raise ValueError("File type not supported!")
    except OpenAIError as e:
            st.error(e._message)


preference = st.text_area("Optional: Enter your job preference(title, location, hybrid, etc.)", on_change=clear_submit)

button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please configure your OpenAI API key!")
        st.stop()
    if not uploaded_file and not resume:
        st.error("Please upload a resume!")
        st.stop()

    st.session_state["submit"] = True
    
    
    csv_loader = CSVLoader(csv_path, encoding='utf-8')
    documents = csv_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY")))
    faissIndex.save_local("faissIndex")

    retriever = FAISS.load_local("faissIndex", OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY"))).as_retriever(search_type="similarity", search_kwargs={"k":5})
    
    system_template="""You are a job applying assistance, an automated service to help applicats to apply to listed jobs. \
        You are given a resume and a job preference that the user input. \
        Then you search the job listings based on the job preference(title, location, or hybrid, etc.) and resume from the user input.\
        The priority of searching of preference should be resume over preference.\
        If there are no job preference, just use the resume. \
        The priority of searching of preference should be title over other attributes.\
        Then find the most matched 5 different jobs from the dataset. \
        For the matched jobs, you will provide the job title, Company name, location, and a link to the job listing follow to the format below. \
        If any of the (job title, location, link, company name) is missing, you will output it as "Missing". \
        Do not make up information that is not in the dataset. \
        If there are no matches, you will output "No matches found".
        
        {context}
        
        Example output format:
        Based on your resume and job preference, here are the top 1 jobs that you can apply to:
        
        1. **Job Title:** job title **Company Name:** name **Location:** location **Link:** link
        
        Please use the link to aplly if it is avaliable. Good luck!
      """
        
    prompt = PromptTemplate(
        input_variables = ['context'],
        template = system_template
    )
    
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=st.session_state.get("OPENAI_API_KEY"), temperature=0)
    memory = ConversationSummaryMemory(llm=ChatOpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY")), max_token_limit=700)
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", verbose=True, retriever=retriever, memory=memory, chain_type_kwargs={
        "verbose": True,
        "prompt": prompt
    })
    

    try:
        if resume:
            query = "Job preference: " + preference + " The resume: "+ resume
        else:
            query = "Job preference: " + preference + " The resume: "+ text
        
        llm_response = qa.run(query)

        if llm_response: 
            st.markdown("""---""")
            st.markdown("## Job Found:")
            st.write(llm_response)
            
            st.markdown("""---""")
            with st.expander('Message History:'):
                st.info(memory.buffer)

    except OpenAIError as e:
        st.error(e._message)
