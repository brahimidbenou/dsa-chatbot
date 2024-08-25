import utils

# database_creation.create_db()
embeddings = utils.get_embedding()
llm = utils.get_llm()
db = utils.load_db("./dsa_db", embeddings)

prompt = utils.get_prompt()

retriever = utils.get_retriever(db)

rag_chain = utils.get_rag(llm, prompt, retriever)

for chunks in rag_chain.stream({"input": "What is the complexity of searching in a linked list?"}):
    print(chunks)
