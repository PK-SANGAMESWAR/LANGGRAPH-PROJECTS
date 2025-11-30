import streamlit as st
import math
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# ----------------------------
# 1. STATE
# ----------------------------

class QuadState(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str


# ----------------------------
# 2. NODE FUNCTIONS
# ----------------------------

def show_equation(state: QuadState) -> QuadState:
    eq = f"{state['a']}x² + {state['b']}x + {state['c']} = 0"
    return {**state, "equation": eq}


def calculate_discriminant(state: QuadState) -> QuadState:
    d = state["b"]**2 - (4 * state["a"] * state["c"])
    return {**state, "discriminant": d}


def real_roots(state: QuadState) -> QuadState:
    root1 = (-state["b"] + math.sqrt(state["discriminant"])) / (2 * state["a"])
    root2 = (-state["b"] - math.sqrt(state["discriminant"])) / (2 * state["a"])
    result = f"Real Roots: {root1:.4f} and {root2:.4f}"
    return {**state, "result": result}


def repeated_roots(state: QuadState) -> QuadState:
    root = -state["b"] / (2 * state["a"])
    result = f"Repeated Root: {root:.4f}"
    return {**state, "result": result}


def no_real_roots(state: QuadState) -> QuadState:
    result = "No real roots (Discriminant < 0)"
    return {**state, "result": result}


def check_condition(state: QuadState) -> Literal["real_roots", "repeated_roots", "no_real_roots"]:
    if state["discriminant"] > 0:
        return "real_roots"
    elif state["discriminant"] == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"


# ----------------------------
# 3. LANGGRAPH WORKFLOW
# ----------------------------

graph = StateGraph(QuadState)

graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("real_roots", real_roots)
graph.add_node("repeated_roots", repeated_roots)
graph.add_node("no_real_roots", no_real_roots)

graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_discriminant")

graph.add_conditional_edges(
    "calculate_discriminant",
    check_condition,
)

graph.add_edge("real_roots", END)
graph.add_edge("repeated_roots", END)
graph.add_edge("no_real_roots", END)

workflow = graph.compile()


# ----------------------------
# 4. STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="Quadratic Solver", page_icon="🧮", layout="centered")

st.title("🧮 Quadratic Equation Solver")
st.write("Solve equations of the form: **ax² + bx + c = 0**")
st.markdown("---")

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    a = st.number_input("Enter a", value=1)
with col2:
    b = st.number_input("Enter b", value=0)
with col3:
    c = st.number_input("Enter c", value=0)

if st.button("Solve Equation"):
    if a == 0:
        st.error("Coefficient 'a' cannot be zero for a quadratic equation.")
    else:
        initial_state = {
            "a": a,
            "b": b,
            "c": c,
            "equation": "",
            "discriminant": 0.0,
            "result": ""
        }

        with st.spinner("Calculating..."):
            final_state = workflow.invoke(initial_state)

        st.success("Solution Found!")
        st.markdown("### 📘 Equation")
        st.info(final_state["equation"])

        st.markdown("### 🔍 Discriminant")
        st.info(final_state["discriminant"])

        st.markdown("### 🧮 Result")
        st.success(final_state["result"])
