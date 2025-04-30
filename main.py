from config import LLM_PROVIDER,LLM_HOST,LLM_PORT,LLM_MODEL,PERMUTATIONS,TARGET_PROMPT_MAX_TOKENS,VERBOSE_MODE
from app.llm_client import get_llm_client
from app.task import PromptOptimizationTask
from app.prompt_gen_task_definitions.summarize_text import task_definition

def main():
    # Initialize LLM client
    llm = get_llm_client(provider=LLM_PROVIDER,host=LLM_HOST,port=LLM_PORT,model=LLM_MODEL)

    # Initialize prompt optimizer with the task definition config class
    prompt_optimizer = PromptOptimizationTask(llm,task_definition,target_prompt_max_tokens=TARGET_PROMPT_MAX_TOKENS)
    prompt_optimizer.run_optimization_loop(permutations=PERMUTATIONS, verbose=VERBOSE_MODE)
    # Plot the final results
    prompt_optimizer.plot_optimization_results()

if __name__ == "__main__":
    main() 