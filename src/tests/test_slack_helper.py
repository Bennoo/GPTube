from slack_functions.slack_helper import get_slack_bolt_app, get_slack_bolt_app_azure, say_standard_block_answer_message

def test_get_slack_bolt_app():
    app = get_slack_bolt_app("openai_model_chat", "openai_model_question", 0.5)
    assert app is not None
    assert app.openaiQuestion is not None
    assert app.openaiChat is not None
    assert app.chat_history == []

def test_get_slack_bolt_app_azure():
    app = get_slack_bolt_app_azure("model_chat_id", "model_question_id", 0.5)
    assert app is not None
    assert app.openaiQuestion is not None
    assert app.openaiChat is not None
    assert app.chat_history == []

def test_say_standard_block_answer_message():
    say = lambda blocks, text, channel_id: None
    answer = "This is a test answer."
    exchanges = 1
    channel_id = "test_channel_id"
    say_standard_block_answer_message(say, answer, exchanges, channel_id)
    # No assertion, just checking if there's no error