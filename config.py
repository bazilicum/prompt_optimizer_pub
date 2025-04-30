import os

# LLM settings
LLM_PROVIDER = "LocalAI"
LLM_MODEL = "mistral-7b-instruct-v0.3"
LLM_HOST = os.getenv("LLM_HOST", "localhost")
LLM_PORT = int(os.getenv("LLM_PORT", "8080"))
LLM_TIMEOUT = 60
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.15

# Optimizer settings
OPT_MAX_CAPACITY = 3  #The number of the final top prompts to save in memory
PERMUTATIONS = 3 #The number of prompt permutations to try
TARGET_PROMPT_MAX_TOKENS = 200
VERBOSE_MODE = False