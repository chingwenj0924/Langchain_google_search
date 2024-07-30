from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# set up google search tool using Langchain Tool
google_search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
google_tool = Tool(
    name="google-search",
    description="Search Google for recent results.",
    func=google_search.run
)

google_search.run("What's Obama's first name?")


# load llm model from huggingface
llm=HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
chat_model=ChatHuggingFace(llm=llm)

# https://smith.langchain.com/hub/hwchase17/structured-chat-agent
# structured-chat-agent basically is a set of instruction that we will give to the LLM
# in which we specified a sequence of actions to take in
# so if the model doesn't know the answer, it will go to the search engine
# and if the answer obtain from the search engine is not enought
# it will go back to the search engine again and keep looking for the answer until it get it
 
prompt = hub.pull("hwchase17/structured-chat-agent")

# create our langchain agent
agent=create_structured_chat_agent(chat_model, [google_tool], prompt)

# create an agentExecutor
agent_executor=AgentExecutor(agent=agent, tools=[google_tool], verbose=True, handle_parsing_errors=True, max_iterations=5)


agent_executor.invoke({"input":"Who won the best picture on the academy awards in 2024?"})

# agent_executor.invoke({"input": "Who won the best actor on the academy awards in 2024?"})

# agent_executor.invoke({"input":"What happened to the bridge in Baltimore?"})

# agent_executor.invoke({"input": "Who won the best actress on the academy awards in 2024?"})
