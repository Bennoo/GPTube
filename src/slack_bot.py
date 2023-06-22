import os

from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.error import BoltUnhandledRequestError

from slack_functions import slack_helper
from langchain_functions import langchain_helper
from langchain_functions.custom_chain import waiting_time

from langchain.embeddings.openai import OpenAIEmbeddings

app = slack_helper.get_slack_bolt_app('gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 0.1)
embeddings = OpenAIEmbeddings()

@app.event('app_mention')
def on_mention(body, say):
    waiting_time_generator = waiting_time.get_waiting_time_generator()
    # Check if a video is already set
    if app.document_db is None:
        warning_text = waiting_time_generator.run('no video set, you should give me a valid youtube video link.')
        slack_helper.say_standard_block_answer_message(say, answer=warning_text)
    else:
        waiting_text = waiting_time_generator.run('Give me some time, I am thinking and I check the video content')
        response = say(
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"_ ... {waiting_text}_"}
                }
            ],
            text=f"{waiting_text}")
        text_question = body['event']['text'][15:]
        answer, generated_question = langchain_helper.get_response_qa_from_query_bolt(text_question, app, "stuff")
        
        text = generated_question
        blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"_ You asked for:_ *{generated_question}*"}
                }
            ]
        
        app.client.chat_update(channel=response['channel'], ts=response['ts'], text=text, blocks=blocks)
        
        slack_helper.say_standard_block_answer_message(say, answer=answer)

@app.command("/set_video")
def repeat_text(ack, say, command):
    # Acknowledge command request
    ack()
    url = command['text']
    say(text="Watching the whole video...")
    db, meta_data = langchain_helper.set_video_as_vector(url, embeddings)
    app.document_db = db
    app.meta_data = meta_data
    say(text="Video is set!")

@app.action('clean_button')
def on_clear(ack, say):
    app.chat_history = []
    app.document_db = None
    app.meta_data = None
    ack()
    clear_text = "All is cleared"
    say(
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"_..{clear_text}.._"}
            }
        ],
        text=f"{clear_text}")

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
