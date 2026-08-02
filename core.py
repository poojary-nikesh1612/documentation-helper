from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

model = init_chat_model(model="gemini-3.1-flash-lite", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2", output_dimensionality=1536
)
vector_store = PineconeVectorStore(
    embedding=embeddings, index_name="documentation-helper"
)


@tool(response_format="content_and_artifact")
def retrive_knowledge(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    documents = vector_store.as_retriever().invoke(query, k=4)

    result = "\n\n".join(
        f"Source:{doc.metadata.get('source','unknown')}\n\nContent:{doc.page_content}"
        for doc in documents
    )

    return result, documents


def run_llm(query: str) -> dict[str, Any]:
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrive_knowledge], system_prompt=system_prompt)

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content[-1]["text"]

    context_docs = []

    for message in response["messages"]:
        if (
            isinstance(message, ToolMessage)
            and hasattr(message, "artifact")
            and isinstance(message.artifact, list)
        ):
            context_docs.extend(
                [doc.metadata.get("source", "Unknown") for doc in message.artifact]
            )

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    res = run_llm(query="What is a agent?")
    print(f"{res['answer']}\n\n{res['context']}")
