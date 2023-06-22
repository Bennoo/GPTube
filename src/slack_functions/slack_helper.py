import os
from slack_bolt import App
from langchain.chat_models import ChatOpenAI

def get_slack_bolt_app(openai_model_chat:str, openai_model_question:str, model_temp:float) -> App:
    app = App(token=os.environ.get("SLACK_BOT_TOKEN"), raise_error_for_unhandled_request=True)

    app.document_db = None
    app.meta_data = None
    app.openaiQuestion = ChatOpenAI(model_name=openai_model_question, temperature=model_temp)
    app.openaiChat = ChatOpenAI(model_name=openai_model_chat, temperature=model_temp)
    app.chat_history = []
    return app

def say_standard_block_answer_message(say, answer):
    text = answer
    blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"_{answer}_"}
            },
            {
			"type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_Clean Bot's memory_"
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Setup"
                    },
                    "action_id": "clean_button"
                }
            }
        ]
    say(blocks=blocks, text=text)

# def ack_and_clean_conv_history(ack, say, app, text_generator):
#     app.chat_history = []
#     waiting_text = text_generator.run('I forgot all you already said before')
#     ack()
#     say(
#         blocks=[
#             {
#                 "type": "section",
#                 "text": {"type": "mrkdwn", "text": f"_..{waiting_text}.._"}
#             }
#         ],
#         text=f"{waiting_text}")