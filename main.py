from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import os
from dotenv import load_dotenv

load_dotenv(override=True)
assert(os.getenv("OPENAI_API_KEY")is not None)

model = ChatOpenAI()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

generate_prompt = SystemMessage(
    """You are an essay assistant tasked with writing excellent 3-paragraph 
        essays."""
    "Generate the best essay possible for the user's request."
    """If the user provides critique, respond with a revised version of your 
        previous attempts."""
)

def generate(state: State) -> State:
    answer = model.invoke([generate_prompt] + state["messages"])
    return {"messages": [answer]}

reflection_prompt = SystemMessage(
    """You are a teacher grading an essay submission. Generate critique and 
        recommendations for the user's submission."""
    """Provide detailed recommendations, including requests for length, depth, 
        style, etc."""
)

def reflect(state: State) -> State:
    # Invert the messages to get the LLM to reflect on its own output
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    # First message is the original user request. 
    # We hold it the same for all nodes
    translated = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](content=msg.content) 
            for msg in state["messages"][1:]
    ]
    answer = model.invoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=answer.content)]}

def should_continue(state: State):
    if len(state["messages"]) > 6:
        # End after 3 iterations, each with 2 messages
        return END
    else:
        return "reflect"

builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

starting_state = {
    'messages': [
        HumanMessage(content="""Your essay on the topicality of "The Little Prince" 
            and its message in modern life is well-written and insightful. You 
            have effectively highlighted the enduring relevance of the book's 
            themes and its importance in today’s society. However, there are a 
            few areas where you could enhance your essay:

            1. **Depth**: While you touch upon the themes of cherishing simple joys, 
            nurturing connections, and understanding human relationships, 
            consider delving deeper into each of these themes. Provide specific 
            examples from the book to support your points and explore how these 
            themes manifest in contemporary life.

            2. **Analysis**: Consider analyzing how the book’s messages can be 
            applied to current societal issues or personal experiences. For instance, 
            you could discuss how the Little Prince's perspective on materialism relates 
            to consumer culture or explore how his approach to relationships can 
            inform interpersonal dynamics in the digital age.

            3. **Length**: Expand on your ideas by adding more examples, discussing 
            counterarguments, or exploring the cultural impact of "The Little Prince" 
            in different parts of the world. This will enrich the depth of your analysis 
            and provide a more comprehensive understanding of the book’s relevance.

            4. **Style**: Your essay is clear and well-structured. To enhance the 
            engagement of your readers, consider incorporating quotes from the book 
            to illustrate key points or including anecdotes to personalize your analysis.

            5. **Conclusion**: Conclude your essay by summarizing the enduring 
            significance of "The Little Prince" and how its messages can inspire 
            positive change in modern society. Reflect on the broader implications 
            of the book’s themes and leave the reader with a lasting impression.

            By expanding on your analysis, incorporating more examples, and deepening 
            your exploration of the book’s messages, you can create a more comprehensive 
            and compelling essay on the topicality of "The Little Prince" in modern life. 
            Well done on your thoughtful analysis, and keep up the good work!""")
    ]
}

final_output = graph.invoke(starting_state)


print(final_output)