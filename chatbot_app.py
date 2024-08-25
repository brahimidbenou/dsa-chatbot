import utils
import streamlit as st

embeddings = utils.get_embedding()
llm = utils.get_llm()
db = utils.load_db("./dsa_db", embeddings)
prompt = utils.get_prompt()
retriever = utils.get_retriever(db)
rag_chain = utils.get_rag(llm, prompt, retriever)

utils.custom_ui()
st.title('DSA Chatbot')

def get_answer(question):
    for chunk in rag_chain.stream({'input': question}):
        if 'answer' in chunk:
            yield chunk['answer']

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg['content'])

if query := st.chat_input():
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message('user'):
        st.write(query)

    with st.chat_message('assistant'):
        with st.spinner('Thinking ...'):
            answer = st.write_stream(get_answer(query))

    st.session_state.messages.append({'role': 'assistant', 'content': answer})
