import utils

def create_db(name="./dsa_db", file="./Linked_List.pdf"):
    embeddings = utils.get_embedding()

    loader = utils.get_loader(file)
    doc = loader.load()

    text_splitter = utils.get_splitter()
    docs = text_splitter.split_documents(doc)
    print(len(docs))

    db = utils.init_db(name, docs, embeddings)
    print(db)