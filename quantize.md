# Quantizing Model Weights for Faster Inference

Naive CPU inference via the `transformers` package used FP32 weights. This requires 4x memory in bytes for each model parameter. This puts models larger than ~2 billion parameters out of reach for common consumer systems.

Quantization allows for faster inference and lower memory consumption by decreasing the number of bits used to store parameters. Common choices are FP16, BF16, int8, and int4.

## Notes


[Example export for GPT2](https://gist.github.com/Norod/3495e86e7224031e1dd071af382d0c86)

[INCITE Models release](https://www.together.xyz/blog/redpajama-models-v1)

[LLM Worksheet](https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4/edit#gid=0)

[togethercomputer/RedPajama-INCITE-Instruct-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1)

[Optimum ONNX Runtime Docs](https://huggingface.co/docs/optimum/v1.8.6/onnxruntime/overview)

## TODO

Pythia-based models do not appear to be [supported by ONNX yet](https://github.com/huggingface/optimum/issues/555)