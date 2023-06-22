from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from slack_bolt import App
from discord.ext import commands
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain_functions.custom_chain import custom_loading_qa_chain as custom_qa

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

def get_qa(client:commands.Bot):
    qa = ConversationalRetrievalChain.from_llm(
        llm=client.openaiChat,
        chain_type="stuff",
        retriever=client.video_db.as_retriever(),
        verbose=True,
        return_source_documents=True
        )
    client.chat_history = []
    return qa

def get_qa_from_query(client:commands.Bot, query):
    answer = client.qa({"question": query, "chat_history": client.chat_history})
    client.chat_history.append((query, answer['answer']))
    return answer['answer']

def set_video_as_vector(link:str, embeddings:OpenAIEmbeddings):
    loader = YoutubeLoader.from_youtube_url(link, add_video_info=True)
    transcript = loader.load()
    video_meta = loader._get_video_info()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    
    return db, video_meta

def get_response_qa_from_query_bolt(query:str, app:App, chain_type:str):
    doc_chain = custom_qa.load_qa_with_sources_chain(app.openaiChat, chain_type=chain_type, verbose=True)
    question_generator = LLMChain(llm=app.openaiQuestion, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)

    qa = ConversationalRetrievalChain(
        retriever=app.document_db.as_retriever(search_kwargs={"k": 30}),
        question_generator=question_generator,
        combine_docs_chain = doc_chain,
        max_tokens_limit=10000,
        return_generated_question=True
    )

    answer = qa({"question": query, "chat_history": app.chat_history})
    app.chat_history.append((query, answer['answer']))
    return answer['answer'], answer['generated_question']