# LangGraph Projects

A collection of AI-powered applications and workflows built using [LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain](https://www.langchain.com/), and [Streamlit](https://streamlit.io/).

## Projects

### 1. PostGenerator
**Directory:** `PostGenerator/`
**Description:** An AI agent that generates, evaluates, and optimizes tweets. It uses a cycle of generation, critique, and refinement to produce high-quality social media content.
**Key Features:**
- **Generator**: Creates initial tweet drafts.
- **Evaluator**: Critiques tweets based on humor, virality, and constraints.
- **Optimizer**: Refines tweets based on feedback.
- **Loop**: Iterates until the tweet meets quality standards or max iterations are reached.

### 2. LLMBasedReviewHandling
**Directory:** `LLMBasedReviewHandling/`
**Description:** An intelligent customer support assistant that analyzes reviews and drafts appropriate responses.
**Key Features:**
- **Sentiment Analysis**: Detects if a review is positive or negative.
- **Diagnosis**: Identifies specific issues (product, service, etc.) in negative reviews.
- **Response Generation**: Crafts personalized responses based on sentiment and diagnosis.

### 3. UPSC Essay Evaluator
**Directory:** `UPSC/`
**Description:** A tool for evaluating essays based on UPSC standards using parallel LLM chains.
**Key Features:**
- **Parallel Evaluation**: Simultaneously evaluates Language, Depth of Analysis, and Clarity.
- **Scoring**: Provides individual scores and an overall average.
- **Feedback**: detailed feedback for each criterion.

### 4. Quadratic Equation Solver
**Directory:** `quadraticeqn/`
**Description:** A workflow to solve quadratic equations ($ax^2 + bx + c = 0$).
**Key Features:**
- **Discriminant Calculation**: Determines the nature of roots.
- **Root Finding**: Calculates real or repeated roots.
- **Condition Handling**: Handles cases with no real roots.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd LangGraph
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ollama Setup:**
    Ensure you have [Ollama](https://ollama.com/) installed and the `llama3.2:3b` model pulled:
    ```bash
    ollama pull llama3.2:3b
    ```

## Usage

Navigate to a project directory and run the Streamlit app:

**Example (PostGenerator):**
```bash
cd PostGenerator
streamlit run post.py
```

**Example (Review Handler):**
```bash
cd LLMBasedReviewHandling
streamlit run llm.py
```
