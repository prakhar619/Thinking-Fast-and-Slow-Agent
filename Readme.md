# **Thinking, Fast and Slow Agent**

"Thinking, Fast and Slow Agent" is a **LangGraph-based agent** that autonomously generates, critiques, and improves essays through a **self-reflection loop**. Inspired by Daniel Kahneman’s *Thinking, Fast and Slow*, the project simulates the process of quick thinking (generation) followed by slow, deliberate reflection (critique and improvement). The agent uses a **reflection prompting technique** to critique and refine its own output iteratively, improving the quality of the generated essay with each cycle.

## **Features**

- **Self-Improvement Loop**: Generates essays, critiques them, and improves them based on feedback in an iterative loop.
- **Reflection Prompting**: Uses the reflection technique to provide feedback on generated essays, allowing the agent to revise and enhance its outputs.
- **LangGraph Architecture**: A dynamic graph structure manages state transitions and multi-step reasoning, ensuring efficient, scalable feedback and improvement.

## **Technologies Used**

- **LangChain**: For chaining model calls and managing conversation history.
- **OpenAI GPT (e.g., GPT-4)**: For generating and critiquing essays.
- **LangGraph**: For managing state transitions and building the feedback-driven workflow.

## **How It Works**

1. **Essay Generation**: The agent generates a 3-paragraph essay based on a given prompt.
2. **Critique and Reflection**: The agent critiques the generated essay using a reflection prompt, simulating feedback from a teacher or reviewer.
3. **Improvement**: Based on the critique, the agent revises the essay to enhance quality.
4. **Iteration**: The agent iterates on this process for a set number of times (3 iterations) to improve the essay through self-reflection.

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/thinking-fast-and-slow-agent.git
   cd thinking-fast-and-slow-agent

pip install -r requirements.txt
