import streamlit as st
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# ----------------------------
# 1. MODEL SETUP
# ----------------------------
model = ChatOllama(model="llama3.2:3b", temperature=0.7)

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"]

structured_model = model.with_structured_output(SentimentSchema)


class DiagnosisSchema(BaseModel):
    issue_type: Literal["product", "service", "pricing", "delivery"]
    tone: Literal["positive", "neutral", "negative", "calm", "angry"]
    urgency: Literal["low", "medium", "high"]

structured_model2 = model.with_structured_output(DiagnosisSchema)


# ----------------------------
# 2. STATE STRUCTURE
# ----------------------------
class ReviewState(TypedDict):
    review: str
    sentiment: str
    diagnosis: dict
    response: str


# ----------------------------
# 3. NODES
# ----------------------------
def find_sentiment(state: ReviewState) -> ReviewState:
    prompt = f"""
    Determine if the sentiment of this review is positive or negative:

    "{state['review']}"
    """
    result = structured_model.invoke(prompt).sentiment
    return {**state, "sentiment": result}


def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    return "run_diagnosis"


def positive_response(state: ReviewState) -> ReviewState:
    prompt = f"""
    Write a warm and friendly thank-you message in response to this review:

    "{state['review']}"

    Also encourage the user to leave feedback on our website.
    """
    reply = model.invoke(prompt).content
    return {**state, "response": reply}


def run_diagnosis(state: ReviewState) -> ReviewState:
    prompt = f"""
    Diagnose this negative review:

    "{state['review']}"

    Identify issue_type, tone, and urgency.
    """
    diag = structured_model2.invoke(prompt)
    return {**state, "diagnosis": diag.model_dump()}


def negative_response(state: ReviewState) -> ReviewState:
    d = state["diagnosis"]
    prompt = f"""
    You are a professional customer support assistant.

    The user had a "{d['issue_type']}" issue.
    Tone: "{d['tone']}"
    Urgency: "{d['urgency']}"

    Write an empathetic, helpful response with a clear resolution.
    """
    reply = model.invoke(prompt).content
    return {**state, "response": reply}


# ----------------------------
# 4. BUILD LANGGRAPH
# ----------------------------
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)

graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()


# ----------------------------
# 5. STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Review Response Assistant", page_icon="💬", layout="centered")

st.title("💬 AI-Powered Review Response Assistant")
st.markdown("Automatically analyze customer reviews and craft perfect responses using LLMs.")

st.write("### ✍️ Enter a customer review:")
review_input = st.text_area("Customer Review", height=150)

if st.button("Generate Response"):
    if not review_input.strip():
        st.error("Please enter a review!")
    else:
        with st.spinner("Analyzing review using LangGraph workflow..."):
            initial_state = {
                "review": review_input,
                "sentiment": "",
                "diagnosis": {},
                "response": ""
            }

            result = workflow.invoke(initial_state)

        st.success("Response Ready!")

        st.write("---")
        st.write("### 🧠 Sentiment Detected:")
        st.info(result["sentiment"].capitalize())

        if result["sentiment"] == "negative":
            st.write("### 🔍 Diagnosis")
            st.json(result["diagnosis"])

        st.write("### 💡 Suggested Response:")
        st.success(result["response"])
