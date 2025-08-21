import re
import os
import datetime

def log_token_usage(operation: str, input_tokens: int, output_tokens: int = 0) -> None:
    """
    Logs the token usage to the total_tokens.txt file.

    Parameters:
    - operation: str; The operation being performed (e.g., "response_generation").
    - input_tokens: int; The number of input tokens used.
    - output_tokens: int; The number of output tokens generated (default is 0).
    """

    if operation not in ["response_generation", "embeddings_generation"]:
        raise ValueError("Operation must be either 'response_generation' or 'embeddings_generation'.")

    # Get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Normalize the path to an absolute one
    log_dir = os.path.abspath(os.path.join(base_dir, "..", "tokens_count"))
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "total_tokens.txt")
    with open(log_file, "a") as file:
        timestamp = datetime.datetime.now().isoformat()
        file.write(f"{timestamp} - {operation}: input={input_tokens}, output={output_tokens}\n")

def clean_string(s: str) -> str:
    cleaned = re.sub(r'\s+', ' ', s).strip()
    return cleaned