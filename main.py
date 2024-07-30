from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain import hub
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# set up google search tool using Langchain Tool
google_search = GoogleSearchAPIWrapper(google_api_key="AIzaSyCfVwVB7SQjl4OIrzYg98rTMbpIMMKeUTs", google_cse_id=GOOGLE_CSE_ID)
tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="useful for when you need to ask with search"
    )
]

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


agent = initialize_agent(tools = tools, 
                         llm = llm, 
                         agent=AgentType.SELF_ASK_WITH_SEARCH, 
                         verbose=True)

agent.run("What is the hometown of the 2001 US PGA champion?")
