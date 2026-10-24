
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from rag import search_documents, ingest_documents


# ============================================================
# 1. TOOL DEFINITIONS
# These descriptions tell Gemini which tools it can request.
# Gemini chooses a tool; Python executes our custom functions.
# ============================================================

search_tool = {
    "name": "search_knowledge_base",
    "description": (
        "Search local documents for information about Python, "
        "cloud computing, data engineering, or topics covered "
        "in the knowledge base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The topic or question to search for.",
            }
        },
        "required": ["query"],
    },
}

calculator_tool = {
    "name": "calculator",
    "description": "Evaluate a mathematical expression.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression, e.g. 2 + 2.",
            }
        },
        "required": ["expression"],
    },
}


# ============================================================
# 2. CUSTOM TOOL EXECUTION
# These functions actually run on your computer.
# ============================================================

def execute_search(query):
    """Search ChromaDB and return matching text and source names."""

    results = search_documents(query)

    if not results:
        return {
            "context": "No relevant documents were found.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for result in results:
        source = result.get("source", "Unknown source")
        content = result.get("content", "")

        context_parts.append(
            f"[Source: {source}]\n{content}"
        )
        sources.append(source)

    # Return the retrieved text so Gemini can use it in its answer.
    return {
        "context": "\n\n".join(context_parts),
        "sources": sorted(set(sources)),
    }


def execute_calculator(expression):
    """Evaluate basic arithmetic after checking the input."""

    allowed_chars = set("0123456789+-*/.() ")

    if not expression or not all(
        char in allowed_chars for char in expression
    ):
        return {
            "error": "Only numbers and basic arithmetic operators are allowed."
        }

    # NOTE: This retains the tutorial's eval approach.
    # A character check is not a complete security boundary.
    # Replace eval with a restricted AST-based evaluator before
    # exposing this tool to untrusted users.
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": str(result)}
    except Exception:
        return {"error": "Could not evaluate that expression."}


# ============================================================
# 3. GOOGLE SEARCH GROUNDING
# Gemini handles the web search; we don't need a custom
# execute_web_search() function for this built-in tool.
# ============================================================

def display_web_sources(response):
    """Display Google Search queries and grounded web sources."""

    if not response.candidates:
        return

    metadata = getattr(
        response.candidates[0],
        "grounding_metadata",
        None,
    )

    if not metadata:
        return

    queries = getattr(metadata, "web_search_queries", None)

    if queries:
        print("\n[Web search queries]")
        for query in queries:
            print(f"  - {query}")

    chunks = getattr(metadata, "grounding_chunks", None)

    if chunks:
        print("\n[Web sources]")
        for chunk in chunks:
            web_page = getattr(chunk, "web", None)

            if web_page:
                title = getattr(web_page, "title", None) or "Web source"
                uri = getattr(web_page, "uri", None)

                if uri:
                    print(f"  - {title}: {uri}")
                else:
                    print(f"  - {title}")


# ============================================================
# 4. MAIN AGENT
# Connect the Gemini API, RAG pipeline, tools, and chat history.
# ============================================================

def run_agent():
    """Run the Agentic RAG Assistant."""

    # Load the API key from .env or the existing environment.
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in your .env file.")
        return

    client = genai.Client(api_key=api_key)

    # Prepare the local document knowledge base.
    print("Preparing the knowledge base...")
    ingest_documents()

    # IMPORTANT: Configure the custom functions and built-in
    # Google Search together, before creating the config.
    all_tools = types.Tool(
        function_declarations=[
            search_tool,
            calculator_tool,
        ],
        google_search=types.GoogleSearch(),
    )

    # One configuration and one system instruction for the agent.
    config = types.GenerateContentConfig(
        tools=[all_tools],
        system_instruction=(
            "You are a helpful research assistant. "
            "Use search_knowledge_base for questions about local documents. "
            "Use calculator for mathematical calculations. "
            "Use Google Search for current events and recent information. "
            "Answer general questions directly when appropriate. "
            "When using local documents, identify their source files. "
            "When using Google Search, use the available web evidence "
            "and cite sources where supported."
        ),
    )

    # Stores user messages, assistant replies, and tool exchanges.
    # This history lasts only while the program is running.
    contents = []

    print(
        "\nAgentic RAG Assistant ready! "
        "Type 'quit' to exit.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Add the new user message to the existing conversation.
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_input)],
                )
            )

            # Send the complete conversation history to Gemini.
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=contents,
                config=config,
            )

            # Handle custom Python function calls.
            # Google Search itself is handled by Gemini.
            while response.function_calls:

                # Save Gemini's request to call the custom tool.
                contents.append(response.candidates[0].content)

                function_response_parts = []

                for function_call in response.function_calls:
                    print(f"\n[Using tool: {function_call.name}]")

                    try:
                        if function_call.name == "search_knowledge_base":
                            result = execute_search(
                                function_call.args["query"]
                            )

                            sources = result.get("sources", [])

                            if sources:
                                print(
                                    "[Local sources: "
                                    + ", ".join(sources)
                                    + "]"
                                )

                        elif function_call.name == "calculator":
                            result = execute_calculator(
                                function_call.args["expression"]
                            )

                        else:
                            result = {
                                "error": (
                                    f"Unknown tool: {function_call.name}"
                                )
                            }

                    except Exception as tool_error:
                        result = {"error": str(tool_error)}

                    # Return the custom tool result to Gemini.
                    # No id= argument: this matches the documented
                    # function-response pattern in the current SDK.
                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=function_call.name,
                            response=result,
                        )
                    )

                # Preserve the tool results in conversation history.
                contents.append(
                    types.Content(
                        role="tool",
                        parts=function_response_parts,
                    )
                )

                # Let Gemini use the tool results to continue answering.
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config,
                )

            # Check whether Gemini returned a usable response.
            if not response.candidates:
                print("\nAssistant: No response was returned.\n")
                continue

            # Display Google Search grounding information, if present.
            display_web_sources(response)

            # Save Gemini's final answer for future follow-up questions.
            contents.append(response.candidates[0].content)

            print(f"\nAssistant: {response.text}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as error:
            print(f"\nError: {error}\n")


# Run the agent when this file is executed directly.
if __name__ == "__main__":
    run_agent()
