from app.task import PromptOptimizationTaskDefinition    
import json
import re

def evaluate_summary_output(
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

    # Calculate length score (0-100)
    # The shorter the summary relative to the original, the higher the score
    # But we want to avoid extremely short summaries, so we cap the score
    original_length = len(input_text.split())
    summary_length = len(summary.split())
    
    if original_length == 0:
        length_score = 0
    else:
        length_ratio = (summary_length / original_length) * 100
        length_score = max(0, 100 - round(length_ratio))

    instruction = (
        "You are a strict evaluator of text summaries. Given an original text and its summary, "
        "evaluate the quality across these dimensions:\n\n"
        "1. Accuracy (0-100): How well does the summary capture the key points and facts from the original text?\n"
        "2. Conciseness (0-100): Is the summary appropriately brief while maintaining important information?\n"
        "3. Coherence (0-100): Is the summary well-structured and easy to understand?\n\n"
        "Then compute a final score from 0 to 100 as an average of these three.\n\n"
        "Be very critical:\n"
        "- Score below 50 if the summary is inaccurate, too verbose, or poorly structured.\n"
        "- Penalize summaries that include information not present in the original text.\n"
        "- Only give 90+ if the summary excels in all three dimensions.\n\n"
        f"Original Text:\n{input_text}\n\n"
        f"Summary:\n{summary}\n\n"
        "Output only JSON format, with no commentary or explanation. JSON format:\n"
        "{\"accuracy\": <0-100>, \"coverage\": <0-100>, \"specificity\": <0-100>, \"score\": <0-100>}"
    )

    response = llm_client.chat_generate(prompt=instruction, message_list=[], max_tokens=max_tokens)
    
    try:
        parsed = json.loads(response.strip())
        llm_score = float(parsed.get("score", 0))
        llm_score = round(max(0, min(100, llm_score)))
    except Exception as e:
        print(f"Failed to parse score from JSON. Attempting regex fallback. Error: {e}")
        match = re.search(r'"?score"?\s*[:=]\s*(\d+(\.\d+)?)', response)
        if match:
            llm_score = round(float(match.group(1)))
            llm_score = max(0, min(100, llm_score))
        else:
            print("Failed to extract score using regex.")
            llm_score = 0

    # Combine LLM score and length score with weights
    # LLM score is weighted more heavily (70%) than length score (30%)
    final_score = (llm_score * 0.7) + (length_score * 0.3)
    
    return round(final_score)

task_definition = PromptOptimizationTaskDefinition(
    task_name="Text Summarization",
    base_prompt_template=(
        "You are designing a system prompt for another LLM. "
        "The goal of that LLM is to create a concise summary of an input text. "
        "The summary should:\n"
        "1. Capture the main points and key information\n"
        "2. Be significantly shorter than the original text (ideally 20-50% of original length)\n"
        "3. Maintain factual accuracy\n"
        "4. Be well-structured and easy to read\n\n"
        "Write ONLY the system prompt that instructs the LLM to do this. "
        "Do NOT include any examples or content. The prompt should be reusable across different texts."
    ),
    training_texts=[
        """
        The Industrial Revolution was a period of major industrialization and innovation during the late 18th and early 19th centuries. The Industrial Revolution began in Great Britain and quickly spread throughout the world. This period saw the development of new manufacturing processes, including the use of steam power, the growth of factories, and the mass production of goods. The Industrial Revolution also led to significant social and economic changes, including urbanization, the rise of the middle class, and the development of new forms of transportation and communication.
        """,
        """
        Machine learning is a branch of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. These systems improve their performance as they are exposed to more data over time. There are three main types of machine learning: supervised learning, where the model learns from labeled data; unsupervised learning, where the model finds patterns in unlabeled data; and reinforcement learning, where the model learns through trial and error based on rewards and penalties.
        """,
        """
        Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas. This produces heat-trapping gases that cause global warming. The effects of climate change include rising sea levels, more frequent and intense extreme weather events, and changes in precipitation patterns. Addressing climate change requires global cooperation and the transition to renewable energy sources.
        """
    ],
    evaluation_function=evaluate_summary_output,
) 