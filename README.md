# GPTube
Discussing with a Youtube video!

## Quick start

1. Clone the repository
2. Install Docker
3. Get yourself an OpenAI API key
4. Create a bot on discord and get the token
5. Build the app with docker

    `docker buildx build -t gptube .`
6. Run the container

    `docker run -e OPENAI_API_KEY={your openai API key} -e DISCORD_TOKEN={Your discord bot token} gptube`
7. Ask the bot to check a video

    By using the command `!set_video {youtube_url}` in the discord server where the bot is installed


8. Ask the bot anything about the video

## 🤔 What is this?

With the emergence of instructional Language Models, there is now the possibility to have personal assistant to have summaries and Question/Answers over material. 
This includes YouTube videos.

## Remarks
This has been build with OPENAI API and Langchain