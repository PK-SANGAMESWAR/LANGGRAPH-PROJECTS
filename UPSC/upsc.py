import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator


# ---------------------------
# 1. LLM SETUP
# ---------------------------
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)

class EvaluationSchema(BaseModel):
    feedback: str
    score: int = Field(ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)


# ---------------------------
# 2. STATE
# ---------------------------
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


# ---------------------------
# 3. NODE FUNCTIONS
# ---------------------------
def evaluate_language(state: UPSCState):
    out = structured_model.invoke(f"""
    Evaluate LANGUAGE quality of this UPSC essay.
    Give feedback + score.

    Essay:
    {state["essay"]}
    """)
    return {
        "language_feedback": out.feedback,
        "individual_scores": [out.score]
    }


def evaluate_analysis(state: UPSCState):
    out = structured_model.invoke(f"""
    Evaluate DEPTH OF ANALYSIS of this UPSC essay.
    Give feedback + score.

    Essay:
    {state["essay"]}
    """)
    return {
        "analysis_feedback": out.feedback,
        "individual_scores": [out.score]
    }


def evaluate_clarity(state: UPSCState):
    out = structured_model.invoke(f"""
    Evaluate CLARITY OF THOUGHT of this UPSC essay.
    Give feedback + score.

    Essay:
    {state["essay"]}
    """)
    return {
        "clarity_feedback": out.feedback,
        "individual_scores": [out.score]
    }


def evaluate_overall(state: UPSCState):
    summary = model.invoke(f"""
    Combine these into final UPSC evaluation:

    LANGUAGE:
    {state["language_feedback"]}

    ANALYSIS:
    {state["analysis_feedback"]}

    CLARITY:
    {state["clarity_feedback"]}
    """).content

    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {
        "overall_feedback": summary,
        "avg_score": avg_score
    }


# ---------------------------
# 4. BUILD LANGGRAPH PARALLEL CHAIN
# ---------------------------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("evaluate_overall", evaluate_overall)

# Parallel execution
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

# Fan-in to final node
graph.add_edge("evaluate_language", "evaluate_overall")
graph.add_edge("evaluate_analysis", "evaluate_overall")
graph.add_edge("evaluate_clarity", "evaluate_overall")

graph.add_edge("evaluate_overall", END)

workflow = graph.compile()


# ---------------------------
# 5. STREAMLIT FRONTEND
# ---------------------------
st.title("📝 UPSC Essay Evaluator (Parallel LLM Chain)")
st.caption("Powered by LangGraph + Llama 3.2 3B (Local)")

st.write("### Enter your essay below:")

essay_input = st.text_area("Essay", height=300)

if st.button("Evaluate Essay"):
    if not essay_input.strip():
        st.error("Please paste or type an essay first.")
    else:
        with st.spinner("Evaluating with parallel LLM chains..."):
            initial_state = {
                "essay": essay_input,
                "language_feedback": "",
                "analysis_feedback": "",
                "clarity_feedback": "",
                "overall_feedback": "",
                "individual_scores": [],
                "avg_score": 0.0
            }

            result = workflow.invoke(initial_state)

        # Display results
        st.success("Evaluation Complete!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Language Score", result["individual_scores"][0])
        with col2:
            st.metric("Analysis Score", result["individual_scores"][1])
        with col3:
            st.metric("Clarity Score", result["individual_scores"][2])

        st.write("### 🗣️ Language Feedback")
        st.info(result["language_feedback"])

        st.write("### 📘 Analysis Feedback")
        st.info(result["analysis_feedback"])

        st.write("### 🧠 Clarity of Thought Feedback")
        st.info(result["clarity_feedback"])

        st.write("### ⭐ Overall UPSC Evaluation")
        st.success(result["overall_feedback"])

        st.write("### 🎯 Final Average Score")
        st.metric("Avg Score", f"{result['avg_score']:.2f}")
