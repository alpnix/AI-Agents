import os
from typing import List, Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.praw import RedditTools
from fastapi import FastAPI
import uvicorn


# use Agno's built-in RedditTools or build a custom one for more control
reddit_tools = RedditTools(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[reddit_tools],
    description="You are a prolific Reddit researcher and assistant. You can find subreddits, read posts, and summarize discussions.",
    instructions=[
        "Use natural language to search for subreddits that match the user's query.",
        "When searching, look for subreddits that are active and relevant.",
        "Provide summaries of top posts if requested.",
        "Always be helpful and concise.",
    ],
    show_tool_calls=True,
    markdown=True,
)

app = FastAPI()

@app.get("/search")
async def search_reddit(query: str):
    """
    Search subreddits using natural language and return the agent's findings.
    """
    response = agent.run(query)
    return {"response": response.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
