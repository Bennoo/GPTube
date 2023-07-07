import os
from slack_bolt import App
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import inflect

def get_slack_bolt_app(openai_model_chat:str, openai_model_question:str, model_temp:float) -> App:
    app = App(token=os.environ.get("SLACK_BOT_TOKEN"), raise_error_for_unhandled_request=True)

    app.document_db = None
    app.meta_data = None
    app.openaiQuestion = ChatOpenAI(model_name=openai_model_question, temperature=model_temp)
    app.openaiChat = ChatOpenAI(model_name=openai_model_chat, temperature=model_temp)
    app.chat_history = []
    return app

def get_slack_bolt_app_azure(model_chat_id:str, model_question_id:str, model_temp:float) -> App:
    app = App(token=os.environ.get("SLACK_BOT_TOKEN"), raise_error_for_unhandled_request=True)

    app.document_db = None
    app.meta_data = None
    app.openaiQuestion = AzureChatOpenAI(
            deployment_name=model_question_id,
            openai_api_version="2023-03-15-preview",
            temperature=model_temp
        )
    app.openaiChat = AzureChatOpenAI(
            deployment_name='gpt35t16k',
            openai_api_version="2023-03-15-preview",
            temperature=model_temp,
            openai_api_base=os.environ.get("OPENAI_API_BASE_2"),
            openai_api_key=os.environ.get("OPENAI_API_KEY_2")
        )
    app.chat_history = []
    return app

def say_standard_block_answer_message(say, answer, exchanges=0, channel_id=None):
    text_exchange = f'{exchanges} {inflect.engine().plural("exchange", exchanges)}'
    text = answer
    blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{answer}"}
            },
            {
			    "type": "divider",
			    "block_id": "divider1"
		    },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_You can clear what the bot reminds from the conversation ({text_exchange})_"
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Clear history"
                    },
                    "style": "danger",
                    "action_id": "button-clear"
                }
            }
        ]
    say(blocks=blocks, text=text, channel_id=channel_id)