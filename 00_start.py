import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
OPENAI_API_KEY = "os.env('OPENAI_API_KEY')"

# Check if the OPENAI_API_KEY is found
#if OPENAI_API_KEY is None:
#    raise ValueError("OPENAI_API_KEY not found in the environment variables. Please check your .env file or system environment variables.")


load_dotenv()
embeddings = OpenAIEmbeddings()
text = "The Ross-Blakley Law Library is the law library associated with the Sandra Day O'Connor College of Law at Arizona State University."
doc_embeddings = embeddings.embed_documents([text])


print(OPENAI_API_KEY)
print(doc_embeddings)
