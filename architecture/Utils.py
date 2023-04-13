import os
import re
import time
import random
import torch

from transformers import GPT2TokenizerFast
from .CodeVerbLM import CodeVerbDLM
from .AccelerateMap import codeverb_map


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Time elapsed {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def cast(model, fp16=True):
    if fp16:
        model.half()
    return model


def load_model(ckpt, fp16=True):
    if fp16:
        return CodeVerbDLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, device_map=codeverb_map)
    else:
        return CodeVerbDLM.from_pretrained(ckpt)


def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


def Inference(
    device,
    model,
    tokenizer,
    context,
    pad_token_id,
    num_return_sequences=1,
    temp=0.2,
    top_p=0.95,
    max_length_sample=128,
    max_length=2048
):

    input_ids = tokenizer(
        context,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).input_ids

    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length

    with torch.no_grad():
        input_ids = input_ids.to(device)
        tokens = model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temp,
            max_length=input_ids_len + max_length_sample,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

    return text


def final_processing(completion):

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def test_final_processing():
    assert final_processing('\nif len_a > len_b:\n    result = a\nelse:\n    result = b\n\n\n\n#') == '\nif len_a > len_b:\n    result = a\nelse:\n    result = b'

def remove_extraspace(inp):
    lines = inp.split('\n')
    stripped_lines = lines
    whitespace_count = len(lines[0]) - len(lines[0].lstrip())
    # Iterate over each line in the string
    for i in range(1, len(lines)):
        stripped_lines[i] = lines[i][whitespace_count:]
    # Join the stripped lines back into a single string with newline characters
    return '\n'.join(stripped_lines)


# def main():
#     prompt = "# write a function to print factorial of a number"

#     set_env()
#     set_seed(42, deterministic=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ckpt = "../model"

#     with print_time('Loading Parameters: '):
#         model = load_model(ckpt=ckpt, fp16=True).to(device)

#     with print_time('Fetching Tokenizer: '):
#         tokenizer = create_custom_gpt2_tokenizer()
#         tokenizer.padding_side = 'left'
#         tokenizer.pad_token = 50256

#     with print_time('Getting Prediction:'):
#         completion = Inference(device=device, model=model, tokenizer=tokenizer, context=prompt, pad_token_id=50256, num_return_sequences=1, temp=0.2, top_p=0.95, max_length_sample=512)[0]
#         output = preprocessOutput(completion)

#         print('=' * 100)
#         print(completion)
#         print('=' * 100)
#         print(output)
#         print('=' * 100)
#         print(remove_extraspace(output))
#         print('=' * 100)



# if __name__ == '__main__':
#     test_truncate()
#     main()