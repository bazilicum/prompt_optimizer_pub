# Prompt Optimizer

A Python-based framework for optimizing prompts using genetic algorithms and local language models. This project provides a flexible and extensible system for creating and optimizing prompts for various tasks.

## Features

- Genetic algorithm-based prompt optimization
- Support for local language models via LocalAI
- Extensible task system for different optimization scenarios
- Configurable optimization parameters
- Detailed logging and progress tracking
- Docker support for easy deployment

## Project Structure

```
prompt_optimizer/
├── app/
│   ├── __init__.py
│   ├── task.py                      # Core task definition and optimization logic
│   ├── memory.py                    # Memory management for top prompts
│   ├── llm_client.py                # LLM API client implementation
│   └── prompt_gen_task_definitions/ # Task definitions
│       ├── __init__.py
│       └── summarize_text.py
├── tests/                   # Test suite
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/prompt-optimizer.git
cd prompt-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Docker compose file example
```dockerfile
services:
  prompt_optimizer:
    build:
      context: ../
      dockerfile: Dockerfile
    volumes:
      - <your code folder>:/usr/local/bin/cde
    env_file:
      - .env
    command: sleep infinity
```

## LLM Client Factory

The project uses a factory pattern to create LLM clients. The `get_llm_client` function in `llm_client.py` serves as a factory that can create different types of LLM clients based on the provider:

```python
from app.llm_client import get_llm_client

# Get a LocalAI client
llm = get_llm_client(provider="localai", host="localhost", port=8080, model="mistral-7b-instruct")

# In the future, you could add more providers:
# llm = get_llm_client(provider="openai", api_key="your-key")
# llm = get_llm_client(provider="anthropic", api_key="your-key")
```

The factory pattern allows for:
- Easy switching between different LLM providers
- Consistent interface across different providers
- Extensibility for adding new providers
- Centralized configuration management

## Creating a New Task Definition

A task definition consists of two main components:

### 1. Evaluation Function

The evaluation function is responsible for scoring the quality of the LLM's output. It should:

- Take the LLM client, input text, and output as parameters
- Return a score between 0 and 100
- Handle potential errors gracefully

Example from `summarize_text.py`:
```python
def evaluate_output_with_llm(
        llm_client,
        input_text: str,
        summary: str,
        max_tokens: int = 50,
        strict: bool = True,
    ) -> int:
    """
    Evaluates the quality of a text summary based on multiple criteria.
    Returns a score between 0 and 100.
    """
    if not summary:
        return 0

    instruction = (
        "You are a strict evaluator of text summaries. Given an original text and its summary, "
        "evaluate the quality across these dimensions:\n\n"
        # ... rest of the evaluation prompt
    )
```

Key points about the evaluation function:
- Evaluation doesn't have to use an LLM—it can be any function that returns a score as feedback.
- It should be strict and critical in its evaluation
- It should handle edge cases (empty output, parsing errors)
- It should provide clear criteria for scoring
- It should return a normalized score between 0 and 100

### 2. Task Definition

The task definition specifies:
- The task name
- The base prompt template
- Training texts
- The evaluation function

Example from `summarize_text.py`:
```python
task_definition = PromptOptimizationTaskDefinition(
    task_name="Text Summarization",
    base_prompt_template=(
        "You are designing a system prompt for another LLM. "
        "The goal of that LLM is to create a concise summary of an input text. "
        "The summary should:\n"
        "1. Capture the main points and key information\n"
        "2. Be significantly shorter than the original text\n"
        "3. Maintain factual accuracy\n"
        "4. Be well-structured and easy to read\n\n"
        "Write ONLY the system prompt that instructs the LLM to do this. "
        "Do NOT include any examples or content. The prompt should be reusable across different texts."
    ),
    training_texts=[
        # Example texts that represent the type of content to be processed
    ],
    evaluation_function=evaluate_output_with_llm,
)
```

Key points about the task definition:
- The base prompt template should be clear and specific
- Training texts should be representative of the task
- The task name should be descriptive
- The evaluation function should be properly referenced

### Best Practices for Creating Tasks

1. **Choose Appropriate Training Texts**
   - Include diverse examples that cover different aspects of the task
   - Ensure texts are representative of real-world use cases
   - Include edge cases and challenging scenarios

2. **Design Clear Evaluation Criteria**
   - Define specific dimensions for evaluation
   - Provide clear scoring guidelines
   - Include examples of good and bad outputs

3. **Write Effective Base Prompts**
   - Be specific about the task requirements
   - Include clear instructions about format and style
   - Avoid including examples in the base prompt
   - Focus on reusability

4. **Handle Edge Cases**
   - Implement proper error handling
   - Provide fallback mechanisms for parsing
   - Include logging for debugging

5. **Test Thoroughly**
   - Test with various input types
   - Verify evaluation consistency
   - Check error handling

## Example Tasks

The project includes an example task that demonstrates these principles:
`summarize_text.py`: Creates concise summaries of input text

This example can serve as template for creating your own tasks.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 