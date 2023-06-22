# flake8: noqa
from langchain.prompts import PromptTemplate

custom_yt_template = """ You are a specialist in youtube videos.
Given the following extracted parts of the complete video transcript, the video meta data and a question, create a final answer. 
If you don't know the answer, just say that you don't know.
Format the answer using MarkDown.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
CUSTOM_YT_PROMPT = PromptTemplate(template=custom_yt_template, input_variables=["summaries", "question"])

CUSTOM_YT_EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)