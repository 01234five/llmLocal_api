from huggingface_hub import hf_hub_download
import torch
from langchain.llms import HuggingFacePipeline, LlamaCpp
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,    
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from langchain.vectorstores import Chroma
def LoadModel(device_type, model_id, model_basename=None):
    print(f"Loading Model: {model_id}, on: {device_type}")
    _model=None
    _tokenizer = None
    if model_basename is not None:
        if ".ggml" in model_basename:
            print("Using Llamacpp")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            print("Using AutoGPTQForCausalLM")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")
            _tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            _model = AutoModelForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (device_type.lower() == "cuda"):  
        #For huggingface models that ends with -HF or which have a .bin file in their HF repo.
        print("Using AutoModelForCausalLM")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        _model.tie_weights()
    else:
        print("Using LlamaTokenizer")
        _tokenizer = LlamaTokenizer.from_pretrained(model_id)
        _model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def DatabaseReturn(directory,embeddings,settings):
    return Chroma(
        persist_directory=directory,
        embedding_function=embeddings,
        client_settings=settings,
    )

