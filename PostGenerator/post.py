import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage

# ---------------------------------------------------------
# 1. MODEL SETUP
# ---------------------------------------------------------
generator_llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
evaluator_llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
optimizer_llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

# ---------------------------------------------------------
# 2. STATE SETUP
# ---------------------------------------------------------
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iterations: int
    max_iterations: int


# ---------------------------------------------------------
# 3. NODE DEFINITIONS
# ---------------------------------------------------------
def generate_tweet(state: TweetState) -> TweetState:

    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet about: "{state['topic']}".

Rules:
- No Q&A format.
- Under 280 characters.
- Use sarcasm, irony, meme-logic, or observational humor.
- Simple English.
- Version {state['iterations'] + 1}.
""")
    ]

    response = generator_llm.invoke(messages).content

    return {
        **state,
        "tweet": response,
        "iterations": state["iterations"] + 1
    }


class TweetSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str


structured_evaluator = evaluator_llm.with_structured_output(TweetSchema)


def evaluate_tweet(state: TweetState) -> TweetState:

    messages = [
        SystemMessage(content="You are a ruthless Twitter critic. No mercy."),
        HumanMessage(content=f"""
Evaluate the following tweet:

"{state['tweet']}"

Criteria:
1. Fresh originality
2. Humor impact
3. Punchiness
4. Virality potential
5. Proper tweet format (no Q&A, no setup-punchline, <280 chars)

Auto-reject if:
- It's Q&A format
- Too long
- Setup-punchline joke
- Ends with generic or flat lines

Respond ONLY as:
evaluation: "approved" or "needs_improvement"
feedback: <one paragraph>
""")
    ]

    result = structured_evaluator.invoke(messages)

    return {
        **state,
        "evaluation": result.evaluation,
        "feedback": result.feedback
    }


def optimize_tweet(state: TweetState) -> TweetState:

    messages = [
        SystemMessage(content="You improve tweets for humor, punchiness, and virality."),
        HumanMessage(content=f"""
Improve this tweet:

Original: "{state['tweet']}"

Feedback:
"{state['feedback']}"

Rewrite it as a sharper, funnier, viral-worthy tweet.
Rules:
- Under 280 chars
- No Q&A
- No setup-punchline
- Keep same topic
- Output ONLY the improved tweet
""")
    ]

    new_tweet = optimizer_llm.invoke(messages).content

    return {
        **state,
        "tweet": new_tweet,
        "iterations": state["iterations"] + 1
    }


# ---------------------------------------------------------
# 4. ROUTER
# ---------------------------------------------------------
def route_evaluation(state: TweetState):
    if state["evaluation"] == "approved" or state["iterations"] >= state["max_iterations"]:
        return "approved"
    return "needs_improvement"


# ---------------------------------------------------------
# 5. BUILD LANGGRAPH
# ---------------------------------------------------------
graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {"approved": END, "needs_improvement": "optimize"}
)

graph.add_edge("optimize", "evaluate")

workflow = graph.compile()


# ---------------------------------------------------------
# 6. STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Tweet Generator AI", page_icon="🐦", layout="centered")

st.title("🐦 Tweet Improvement AI (Generator → Evaluator → Optimizer)")
st.write("Automatically generates, evaluates, and improves tweets until they are approved!")

st.divider()

topic = st.text_input("Enter a topic for the tweet:")
max_iter = st.number_input("Maximum improvement iterations:", value=5, min_value=1)

if st.button("Generate Tweet"):
    if topic.strip() == "":
        st.error("Please enter a topic.")
    else:
        state = {
            "topic": topic,
            "tweet": "",
            "evaluation": "needs_improvement",
            "feedback": "",
            "iterations": 0,
            "max_iterations": max_iter
        }

        st.write("### 🔄 Running Tweet Improvement Loop...")
        st.divider()

        step_container = st.container()

        # Run workflow step-by-step (stream-like simulation)
        with st.spinner("Optimizing Tweet..."):
            result = workflow.invoke(state)

        st.success("🎉 Final Tweet Approved!")
        st.write("## ✅ Final Tweet")
        st.info(result["tweet"])

        st.write("### 🧠 Final Evaluation")
        st.success(result["evaluation"])

        st.write("### 💬 Final Feedback")
        st.write(result["feedback"])

        st.write("### 🔢 Iterations Used")
        st.metric(label="Iterations", value=result["iterations"])
