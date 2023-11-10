import os
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
import streamlit as st
# from openai.error import OpenAIError
from langchain.vectorstores import FAISS
from pypdf import PdfReader
from hashlib import md5
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
import docx
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain, ConversationChain
from streamlit_chat import message


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
            "JobGPT is an innovative tool designed to seamlessly connect your unique qualifications, as reflected in your resume, with the ideal job that matches your preferences. It currently utilizes a self-sourced dataset, painstakingly scraped from Capital One and Fidelity, ensuring an update frequency of every two weeks or even more rapidly whenever possible. While this product is still under development, we actively encourage and welcome suggestions for improvement, as we strive to enhance your job-matching experience."
            "Feel free to improve this product https://github.com/yuchuheng626/JobGPT"
        )
        st.markdown("Made by [Chuheng Yu](https://www.linkedin.com/in/chuheng-yu99/)")
        st.markdown("---")

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

def generate_response(prompt, answer):
    des_template = """ You are a conversational job assistant. \
        You get multiple job listings from the previous step. \
        And you will answer questions from then user about the job listings. \
        Based on the description of the job, you will answer the questions. \
        Do not make up any answers. \
        If you do not know the answer, you will output "I do not know". \
        
        {history} 
        
        Human: {input}
        Ai:
    """
    
    des_template += answer
    
    des_prompt = PromptTemplate(input_variables=["history", "input"], template=des_template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3, openai_api_key=st.session_state.get("OPENAI_API_KEY"))
    chat_chain = ConversationChain(llm=llm, prompt=des_prompt, verbose=True, memory=ConversationSummaryMemory(llm=ChatOpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY")), max_token_limit=700))
    return chat_chain.run(prompt)
    
def get_text():
        input_text = st.text_input("You: ", "Ask me anything about the jobs...", key="input")
        return input_text

def chat_des(answer):
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    user_input = get_text()

    if user_input:
        output = generate_response(user_input, answer)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            
    

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

sidebar()

uploaded_file = st.file_uploader(
    "Upload a pdf or a docx version of your resume",
    type=["pdf", "docx"],
    on_change=clear_submit,
)

resume = st.text_input(label="Or paste your resume here", placeholder="Paste your resume here")


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


preference = st.text_area("Optional: Enter your job preference(title, location, hybrid, etc.)", placeholder="Full time data analysis job at New York", on_change=clear_submit)

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
        For the matched jobs, you will return the exact row content from the data source. \
        If any of the job title, description, location, link, or company name is missing, you will output it as "Missing". \
        Do not make up information that is not in the dataset. \
        If there are no matches, you will output "No matches found".
        
        {context}
        
        Example output format:
        - Job Title: job title Company Name: name Location: location Description: description Link: link
      """
        
    prompt = PromptTemplate(
        input_variables = ['context'],
        template = system_template
    )
    
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=st.session_state.get("OPENAI_API_KEY"), temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", verbose=True, retriever=retriever, chain_type_kwargs={
        "verbose": True,
        "prompt": prompt
    })
    
    response_template = """Reshape the input to the format below:
        {context}
        Example output format:
        Based on your resume and job preference, here are the top 1 jobs that you can apply to:
        
        1. **Job Title:** job title **Company Name:** name **Location:** location **Link:** link
        
        Please use the link to aplly if it is avaliable. Good luck!"""
    prompt_ans = PromptTemplate(input_variables=["context"], template=response_template)
    
    llm_response = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=st.session_state.get("OPENAI_API_KEY"), temperature=0)
    answer_chain = LLMChain(llm=llm_response, prompt=prompt_ans)
    memory = ConversationSummaryMemory(llm=ChatOpenAI(openai_api_key=st.session_state.get("OPENAI_API_KEY")), max_token_limit=700)
    
    overall_chain = SimpleSequentialChain(
        memory=memory,
        chains=[qa_chain, answer_chain],
        verbose=True)
    
    try:
        if resume:
            query = "Job preference: " + preference + " The resume: "+ resume
        else:
            query = "Job preference: " + preference + " The resume: "+ text
        
        llm_response = overall_chain.run(query)

        if llm_response: 
            st.session_state.past.append(query)
            st.session_state.generated.append(llm_response)
            
            st.markdown("""---""")
            st.markdown("## Job Found:")
            st.write(llm_response)
            
            st.markdown("""---""")
            st.markdown("## Chat For Detail:")
            answer = qa_chain.run(query)
            
            st.session_state["past"] = []
            st.session_state["generated"] = []
            
            chat_des(answer)
            
            st.markdown("""---""")
            with st.expander('Message History:'):
                st.info(memory.buffer)
                
            # Allow to download as well
            download_str = []
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("Conversation", expanded=True):
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    st.info(st.session_state["past"][i],icon="🧐")
                    st.success(st.session_state["generated"][i], icon="🤖")
                    download_str.append(st.session_state["past"][i])
                    download_str.append(st.session_state["generated"][i])
                
                # Can throw error - requires fix
                download_str = '\n'.join(download_str)
                if download_str:
                    st.download_button('Download',download_str)

    except OpenAIError as e:
        st.error(e._message)
