import sys

from psychicapi import Psychic, ConnectorId
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Create a document loader for Notion. We can also load from other connectors e.g. ConnectorId.gdrive
psychic = Psychic(secret_key="PSYCHIC_SECRET_KEY") # Replace this with your secret API key

raw_docs = psychic.get_documents(ConnectorId.notion, "account_id") #replace this with the account_id you set when creating this connection

if raw_docs is None:
    answer = "No documents were found"
else:
	documents = [
		Document(page_content=doc["content"], metadata={"title": doc["title"], "source": doc["uri"]},)
		for doc in raw_docs
	]

	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
	texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY") # Replace this with your OpenAI API Key
    vdb = Chroma.from_documents(texts, embeddings)
    query = sys.argv[1]
    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=vdb.as_retriever())
    answer = chain({"question": query}, return_only_outputs=True)
print(answer)