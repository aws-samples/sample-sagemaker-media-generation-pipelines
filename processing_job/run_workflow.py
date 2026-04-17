import argparse
import json
from urllib import request

ENDPOINT = "http://127.0.0.1:8188/prompt"

def send_workflow(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request(ENDPOINT, data=data)
    request.urlopen(req)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ComfyUI workflow')
    parser.add_argument('--workflow', type=str, default='prompt_template.json', help='Path to workflow JSON')
    parser.add_argument('--prompt-file', type=str, default='prompts.txt',
                        help='Path to JSON file containing prompt template')
    parser.add_argument('--seed', type=int, default=12345, help='Seed value')

    return parser.parse_args()

def load_workflow_from_file(file_path):
    """Load workflow JSON from file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Workflow file '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"Error loading workflow: {e}")
        return None

def read_prompts_from_file(file_path):
    """Read all prompts from the prompt file."""
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Error: Prompt file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

def get_prompt_by_seed(prompts, seed):
    """Select a prompt based on the last digit of the seed."""
    if not prompts:
        return None

    last_digit = seed % 10
    row_index = 9 if last_digit == 0 else last_digit - 1

    if 0 <= row_index < len(prompts):
        selected_prompt = prompts[row_index]
        print(f"Using prompt from line {row_index + 1}: {selected_prompt}")
        return selected_prompt
    else:
        print(f"Warning: Line {row_index + 1} doesn't exist in the prompt file. File has {len(prompts)} lines.")
        return None

def update_workflow_with_prompt(workflow, prompt_node_id, prompt_text):
    """Update the workflow with the selected prompt."""
    if workflow and prompt_text:
        workflow[prompt_node_id]["inputs"]["text"] = prompt_text
    return workflow

def update_workflow_with_seed(workflow, seed_node_id, seed_value):
    """Update the workflow with the seed value."""
    if workflow:
        workflow[seed_node_id]["inputs"]["seed"] = seed_value
    return workflow

def main():
    args = parse_arguments()

    workflow = load_workflow_from_file(args.workflow)
    if not workflow:
        return

    prompts = read_prompts_from_file(args.prompt_file)
    prompt_text = get_prompt_by_seed(prompts, args.seed)

    workflow = update_workflow_with_prompt(workflow, "45", prompt_text)
    workflow = update_workflow_with_seed(workflow, "44", args.seed)
    print(workflow)
    if workflow:
        send_workflow(workflow)
        print(f"Workflow queued with seed {args.seed}")
    else:
        print("Failed to queue workflow due to previous errors.")

if __name__ == "__main__":
    main()
