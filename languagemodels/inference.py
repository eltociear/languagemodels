import requests
import os
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
    """Returns a list of token (name, id) tuples

    TODO: Implement this using ggml backend

    >>> list_tokens("Hello world")
    [('Hello', 27903), (' world', 924)]
    """

    model = get_model("SlyEcho/open_llama_3b_ggml", "open-llama-3b-q5_1.bin")

    ids = model.tokenize(prompt.encode(), add_bos=False)

    return [(model.detokenize([i]).decode(), i) for i in ids]


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


def get_model(url, model):
    if model not in modelcache:
        cache_dir = os.path.expanduser(
            os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "langaugemodels")
        )

        os.makedirs(cache_dir, exist_ok=True)

        modelfile = os.path.join(cache_dir, model)

        if not os.path.isfile(modelfile):
            download_url(
                "https://huggingface.co/" + url + "/resolve/main/" + model, modelfile
            )

        modelcache[model] = Llama(model_path=modelfile, embedding=True)

    return modelcache[model]


def generate_completion(
    prompt, max_tokens=200, temperature=0.1, repeat_penalty=1.2
):
    if os.environ.get("ts_key") or os.environ.get("ts_server"):
        return generate_ts("flan_t5_xxl_q4", prompt, max_tokens)

    if os.environ.get("oa_key"):
        return generate_oa("text-babbage-001", prompt, max_tokens)

    model = get_model("SlyEcho/open_llama_3b_ggml", "open-llama-3b-q5_1.bin")

    return model.create_completion(
        prompt,
        repeat_penalty=repeat_penalty,
        top_p=0.1,
        stop=["\n"],
        max_tokens=max_tokens,
        temperature=temperature,
    )["choices"][0]["text"]


def generate_instruct(prompt, max_tokens=200, temperature=0.1, repeat_penalty=1.2):
    """Generates one completion for a prompt using an instruction-tuned model

    This may use a local model, or it may make an API call to an external
    model if API keys are available.
    """

    instruction = prompt.split(":")[0].strip()
    context = ":".join(prompt.split(":")[1:]).strip()

    prompt = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    prompt += f"### Instruction:\n{instruction}"

    if context:
        prompt += f"\n\n### Input:\n{context}"

    prompt += "\n\n### Response:\n"

    return generate_completion(
        prompt,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def convert_chat(prompt):
    """Converts a chat prompt using special tokens to a plain-text prompt

    This is useful for prompting generic models that have not been fine-tuned
    for chat using specialized tokens.

    >>> convert_chat("<|system|>A helpful assistant<|endoftext|>" \\
    ...              "<|prompter|>What time is it?<|endoftext|>" \\
    ...              "<|assistant|>")
    'A helpful assistant\\n\\nUser:What time is it?\\n\\nAssistant:'

    >>> convert_chat("<|prompter|>Who are you?<|endoftext|>" \\
    ...              "<|assistant|>")
    'User:Who are you?\\n\\nAssistant:'

    >>> convert_chat("<|prompter|>What is 1+1?<|endoftext|>\\n\\n" \\
    ...              "<|assistant|>")
    'User:What is 1+1?\\n\\nAssistant:'

    >>> convert_chat("<|system|>A friend<|endoftext|>" \\
    ...              "<|prompter|>Hi<|endoftext|>" \\
    ...              "<|assistant|>Yo<|endoftext|>" \\
    ...              "<|prompter|>We good?<|endoftext|>" \\
    ...              "<|assistant|>")
    'A friend\\n\\nUser:Hi\\n\\nAssistant:Yo\\n\\nUser:We good?\\n\\nAssistant:'
    >>> convert_chat("\\n<|system|>Be nice<|endoftext|>" \\
    ...              "<|prompter|>brb\\n<|endoftext|>" \\
    ...              "<|assistant|>k<|endoftext|>" \\
    ...              "<|prompter|>back<|endoftext|>" \\
    ...              "<|assistant|>")
    'Be nice\\n\\nUser:brb\\n\\nAssistant:k\\n\\nUser:back\\n\\nAssistant:'

    >>> convert_chat("<|user|>Who are you?<|endoftext|>" \\
    ...              "<|assistant|>")
    Traceback (most recent call last):
        ....
    inference.InferenceException: Invalid special token in chat prompt: <|user|>
    """

    prompt = re.sub(r"\s*<\|system\|>\s*", "", prompt)
    prompt = re.sub(r"\s*<\|prompter\|>\s*", "User:", prompt)
    prompt = re.sub(r"\s*<\|assistant\|>\s*", "Assistant:", prompt)
    prompt = re.sub(r"\s*<\|endoftext\|>\s*", "\n\n", prompt)

    special_token_match = re.search(r"<\|.*?\|>", prompt)
    if special_token_match:
        token_text = special_token_match.group(0)
        raise InferenceException(f"Invalid special token in chat prompt: {token_text}")

    return prompt
