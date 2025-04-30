from app.memory import TopNMemory
from config import OPT_MAX_CAPACITY

class PromptOptimizationTaskDefinition:
    def __init__(self, task_name: str, base_prompt_template: str, training_texts: list, evaluation_function, output_parser_function=None, llm_client_params=None, extra_params=None):
        self.task_name = task_name
        self.base_prompt_template = base_prompt_template
        self.training_texts = training_texts
        self.evaluation_function = evaluation_function
        self.output_parser_function = output_parser_function

class PromptOptimizationTask:
    """
    Represents a configurable LLM prompt optimization task.

    Attributes:
        task_name (str): Name of the task (e.g., "Search Phrase Extraction").
        base_prompt_template (str): The main prompt template to be optimized.
        training_texts (list): List of input texts used for training/permutations.
        evaluation_function (callable): Function to evaluate LLM outputs.
        output_parser_function (callable, optional): Function to parse LLM output into a usable format.
        llm_client_params (dict, optional): Parameters for initializing the LLM client.
        extra_params (dict, optional): Any additional parameters for custom use.
    """

    def __init__(self, llm_client, task_definition: PromptOptimizationTaskDefinition,target_prompt_max_tokens: int):
        self.task_definition = task_definition
        self.llm_client = llm_client
        self.target_prompt_max_tokens = target_prompt_max_tokens
        self.memory = TopNMemory(capacity=OPT_MAX_CAPACITY)

    def generate_prompt(self):
        """
        Generate a prompt for the LLM using the base prompt template.
        """
        return self.llm_client.generate(self.task_definition.base_prompt_template, max_tokens=self.target_prompt_max_tokens)

    def generate_prompt_with_feedback(self, top_prompts: list):
        """
        Generate a new prompt using feedback from top prompts.
        This method is generic and adapts to the task's name and base prompt.
        """
        formatted = "\n".join(
            f"{i+1}. \"{item['prompt']}\" (Score: {item['score']})"
            for i, item in enumerate(top_prompts)
        )
        instruction = (
            f"You are optimizing prompts for the task: {self.task_definition.task_name}.\n\n"
            f"Task description:\n{self.task_definition.base_prompt_template}\n\n"
            f"Below are examples of prompts that performed well:\n{formatted}\n\n"
            "Reflect briefly on what makes these prompts effective. Then, generate a new prompt that builds on their strengths and introduces new ideas that might improve the quality further.\n\n"
            "Only output the new prompt. Do not repeat the examples. Prompt max number of tokens is 80."
        )
        return self.llm_client.generate(instruction, max_tokens=self.target_prompt_max_tokens)

    def evaluate_output(self, input_text, output):
        """
        Evaluate the LLM's output using the provided evaluation function.
        """
        return self.task_definition.evaluation_function(self.llm_client, input_text, output)
    
    def default_output_parser(output):
        """
        Splits the LLM output into a list of non-empty, stripped lines.
        """
        return [line.strip() for line in output.split("\n") if line.strip()]  

    def parse_output(self, raw_output):
        """
        Parse the raw LLM output into a usable format.
        If a parser function is provided, use it; otherwise, return the raw output.
        """
        if self.task_definition.output_parser_function:
            return self.task_definition.output_parser_function(raw_output)
        return self.default_output_parser(raw_output) 
    
    def run_base_prompt (self, system_prompt: str, input_text: str) -> list:
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        result = self.llm_client.chat_generate(prompt=input_text, message_list=messages)
        return result
    

    def run_optimization_loop(self, permutations=10, verbose=False):
        """
        Run the main optimization loop for this task.
        Args:
            permutations: Number of different prompt permutations to try.
            verbose: If True, print progress and results.
        """

        for permutation in range(permutations):
            top = self.memory.get_top()
            if len(top) >= 3:
                prompt = self.generate_prompt_with_feedback(top[:3])
            else:
                prompt = self.generate_prompt()
            if verbose:
                print ('\n\n\n'+'='*80)
                print (f'Permutation No. {permutation+1}')
                print(f'Generated Prompt:\n{prompt}')
                print ('='*80)
            else:
                print (f'\nPermutation {permutation+1}')
                print('.'*50)

            total_score = 0
            all_output = []

            for idx, text in enumerate(self.task_definition.training_texts):
                if verbose:
                    print ('\n'+'-'*65+f'\nDataset no. #{idx + 1}\n'+'-'*65)
                    print(f"Input:\n{text.strip()}")
                    print('.'*50)
                else:
                    print (f'Processing dataset no. #{idx + 1}')
                base_prompt_output = self.run_base_prompt(prompt, text)
                if verbose:
                    print(f"Generated Output:\n{base_prompt_output}")
                    print('.'*50)
                score = self.evaluate_output(text, base_prompt_output)
                if verbose:
                    print(f"\nPermutation Score: {score}\n")
                total_score += score
                all_output.append(base_prompt_output)

            avg_score = total_score // len(self.task_definition.training_texts)
            self.memory.add(prompt, all_output, avg_score, "\n\n".join(self.task_definition.training_texts))

    def plot_optimization_results(self):
        """Plot the optimization results in a structured format.
        """
        print("\n\n\n" + "="*80)
        print("OPTIMIZATION RESULTS".center(80))
        print("="*80 + "\n")
        
        top_prompts = self.memory.get_top()
        if not top_prompts:
            print("No results to display")
            return
            
        for i, entry in enumerate(top_prompts, 1):
            print(f"\nPrompt permutation #{i} - Performance score: {entry['score']}/100")
            print("\nPrompt:")
            print(entry['prompt'])
            print("."*80+"\n")



