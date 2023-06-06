import json
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from discord.ext import commands
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def get_response_from_query(client:commands.Bot, query:str, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """
    docs = client.video_db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    system_message_prompt = SystemMessagePromptTemplate.from_template(client.template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=client.openaiChat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content, meta_info=client.video_meta)
    return response, docs

def set_video_as_vector(link:str, embeddings:OpenAIEmbeddings):
    loader = YoutubeLoader.from_youtube_url(link, add_video_info=True)
    transcript = loader.load()
    video_meta = loader._get_video_info()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    
    return db, video_meta