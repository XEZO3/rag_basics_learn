
from dotenv import load_dotenv

# Load environment variables before initializing tracing
load_dotenv(override=True)

import os
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Verify configuration without exposing secrets
print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
print("Endpoint:", os.getenv("LANGSMITH_ENDPOINT"))
print("LangSmith API key loaded:", bool(os.getenv("LANGSMITH_API_KEY")))


@traceable(name="demo_tracing", project_name="my-agent")
def demo_tracing():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        "Summarize: {topic}"
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "topic": "mansaf food jordan"
    })

    print("Result:", result)
    return result


if __name__ == "__main__":
    demo_tracing()
