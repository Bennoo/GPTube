from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI


def get_waiting_time_generator():
    prompt_template = "Make a variation of the followinf sentence: \
    {sentence} \
    The sentence should keep the same meaning but should be different. Keep it simple. \
    "
    prompt = PromptTemplate(input_variables=["sentence"], template=prompt_template)
    chatopenai = ChatOpenAI(
                model_name="gpt-3.5-turbo", temperature=0.9)
    llmchain_chat = LLMChain(llm=chatopenai, prompt=prompt)
    return llmchain_chat


