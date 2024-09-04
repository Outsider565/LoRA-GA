from human_eval.data import write_jsonl, read_problems
from fire import Fire
from tqdm import trange, tqdm
from utils import initialize_text_to_text_model, model_inference
import re
from vllm import LLM, SamplingParams
import os

ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""


def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', "", text, flags=re.DOTALL)
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r"^( *)", line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except:
        print("No def line found")
        print(text)
        def_line_space = 0
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)


def generate_one_completion(model, sampling_params, prompt, template=True):
    if template:
        prompt_in = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt)
    pred_text = (
        model.generate(prompt_in, sampling_params=sampling_params, use_tqdm=False)[0]
        .outputs[0]
        .text
    )
    post_pred = post_process(pred_text)
    return post_pred


def main(model_name):
    problems = read_problems()
    model = LLM(model_name, dtype="bfloat16")
    sampling_params = SamplingParams(top_p=0.95, temperature=0.8, max_tokens=1024)
    num_samples_per_task = 5
    samples = [
        dict(
            task_id=task_id,
            completion=generate_one_completion(
                model, sampling_params, problems[task_id]["prompt"]
            ),
        )
        for task_id in tqdm(problems, desc="Tasks")
        for _ in range(num_samples_per_task)
    ]
    target_name = f"{model_name.replace('/', '_')}_humaneval_samples.jsonl"
    target_name = os.path.join("./code_eval", target_name)
    write_jsonl(target_name, samples,append=False)


if __name__ == "__main__":
    Fire(main)
