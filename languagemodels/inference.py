import requests
import os
from transformers import pipeline, T5Tokenizer
import re
from llama_cpp import Llama
import urllib.request

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class InferenceException(Exception):
    pass


modelcache = {}


def list_tokens(prompt):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    )

    ids = [int(id) for id in input_ids[0]]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True)

    return list(zip(tokens, ids))


def generate_ts(engine, prompt, max_tokens=200):
    """Generates a single text response for a prompt from a textsynth server

    The server and API key are provided as environment variables:

    ts_server is the server such as http://localhost:8080
    ts_key is the API key
    """
    apikey = os.environ.get("ts_key") or ""
    server = os.environ.get("ts_server") or "https://api.textsynth.com"

    response = requests.post(
        f"{server}/v1/engines/{engine}/completions",
        headers={"Authorization": f"Bearer {apikey}"},
        json={"prompt": prompt, "max_tokens": max_tokens},
    )
    resp = response.json()
    if "text" in resp:
        return resp["text"]
    else:
        raise InferenceException(f"TextSynth error: {resp}")


def generate_oa(engine, prompt, max_tokens=200, temperature=0):
    """Generates a single text response for a prompt using OpenAI

    The server and API key are provided as environment variables:

    oa_key is the API key
    """
    apikey = os.environ.get("oa_key")

    response = requests.post(
        "https://api.openai.com/v1/completions",
        headers={
            "Authorization": f"Bearer {apikey}",
            "Content-Type": "application/json",
        },
        json={
            "model": engine,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    resp = response.json()

    try:
        return resp["choices"][0]["text"]
    except KeyError:
        raise InferenceException(f"OpenAI error: {resp}")


def generate_instruct(prompt, max_tokens=200, temperature=0.1, repetition_penalty=1.2):
    """Generates one completion for a prompt using an instruction-tuned model

    This may use a local model, or it may make an API call to an external
    model if API keys are available.
    """
    if os.environ.get("ts_key") or os.environ.get("ts_server"):
        return generate_ts("flan_t5_xxl_q4", prompt, max_tokens)

    if os.environ.get("oa_key"):
        return generate_oa("text-babbage-001", prompt, max_tokens)

    url = "https://huggingface.co/Sovenok-Hacker/nanoalpaca-3b/resolve/main/"
    model = "nano-alpaca-3b-q4_0-ggml.bin"

    if model not in modelcache:
        cache_dir = os.path.expanduser(os.path.join(
            os.getenv("XDG_CACHE_HOME", "~/.cache"), "langaugemodels"
        ))

        os.makedirs(cache_dir, exist_ok=True)

        modelfile = os.path.join(cache_dir, model)

        if not os.path.isfile(modelfile):
            download_url(url + model, modelfile)

        modelcache[model] = Llama(model_path=modelfile)

    prompt = 'Below is an instruction that describes a task. '\
             'Write a response that appropriately completes the request.\n\n'\
             f'### Instruction:\n{prompt}'\
             '\n\n### Response:\n'

    return modelcache[model].create_completion(
        prompt,
        repeat_penalty=repetition_penalty,
        top_p=0.9,
        stop=['\n'],
        max_tokens=max_tokens,
        temperature=temperature,
    )["choices"][0]["text"]


def get_pipeline(task, model):
    """Gets a pipeline instance

    This is thin wrapper around the pipeline constructor to provide caching
    across calls.
    """

    if model not in modelcache:
        modelcache[model] = pipeline(
            task, model=model, model_kwargs={"low_cpu_mem_usage": True}
        )

    return modelcache[model]


def convert_chat(prompt):
    """Converts a chat prompt using special tokens to a plain-text prompt

    This is useful for prompting generic models that have not been fine-tuned
    for chat using specialized tokens.

    >>> convert_chat("<|system|>A helpful assistant<|endoftext|>" \\
    ...              "<|prompter|>What time is it?<|endoftext|>" \\
    ...              "<|assistant|>")
    'A helpful assistant\\n\\n### Input:\\nWhat time is it?\\n\\n### Response:\\n'

    >>> convert_chat("<|prompter|>Who are you?<|endoftext|>" \\
    ...              "<|assistant|>")
    '### Input:\\nWho are you?\\n\\n### Response:\\n'

    >>> convert_chat("<|prompter|>What is 1+1?<|endoftext|>\\n\\n" \\
    ...              "<|assistant|>")
    '### Input:\\nWhat is 1+1?\\n\\n### Response:\\n'

    >>> convert_chat("<|user|>Who are you?<|endoftext|>" \\
    ...              "<|assistant|>")
    Traceback (most recent call last):
        ....
    inference.InferenceException: Invalid special token in chat prompt: <|user|>
    """

    prompt = re.sub(r"\s*<\|system\|>\s*", "", prompt)
    prompt = re.sub(r"\s*<\|prompter\|>\s*", "### Input:\n", prompt)
    prompt = re.sub(r"\s*<\|assistant\|>\s*", "### Response:\n", prompt)
    prompt = re.sub(r"\s*<\|endoftext\|>\s*", "\n\n", prompt)

    special_token_match = re.search(r"<\|.*?\|>", prompt)
    if special_token_match:
        token_text = special_token_match.group(0)
        raise InferenceException(f"Invalid special token in chat prompt: {token_text}")

    return prompt
