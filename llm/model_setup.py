"""
LLM model setup and initialization for the multi-agent dropout prediction system
"""
import torch
import warnings
from transformers import (set_seed, AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, pipeline)
from peft import PeftModel
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFacePipeline


def setup_ollama_models(llama_model_id):
    """Setup Ollama LLM models for agent and arbiter"""
    try:
        OLLAMA_SHORT = ChatOllama(model=llama_model_id, temperature=0.0, max_tokens=256)
        OLLAMA_LONG  = ChatOllama(model=llama_model_id, temperature=0.0, max_tokens=1024)
        return OLLAMA_SHORT, OLLAMA_LONG, True
    except Exception as e:
        print(f"Ollama models not available: {e}")
        return None, None, False


def setup_gpt2_model(gpt2_dir):
    """Setup GPT-2 QLoRA model for SMETimes agent"""
    try:
        tok = AutoTokenizer.from_pretrained(gpt2_dir, local_files_only=True)
        tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
        gpt2_base = AutoModelForCausalLM.from_pretrained(
            "gpt2-xl", local_files_only=True,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=bnb)
        gpt2_lora = PeftModel.from_pretrained(
            gpt2_base, gpt2_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True, device_map="auto" if torch.cuda.is_available() else None)

        PIPE_SLM = pipeline("text-generation", model=gpt2_lora, tokenizer=tok,
                            max_new_tokens=60, device_map="auto" if torch.cuda.is_available() else None, return_full_text=False)
        GPT2_SLM = HuggingFacePipeline(pipeline=PIPE_SLM)
        return GPT2_SLM
    except Exception as e:
        print(f"GPT-2 model not found: {e}")
        return None


def setup_local_llama_model(llama_local_path):
    """Setup local Llama model (alternative to Ollama)"""
    try:
        bnb_llama = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tok_llama = AutoTokenizer.from_pretrained(llama_local_path, local_files_only=True)
        model_llama = AutoModelForCausalLM.from_pretrained(
            llama_local_path,
            quantization_config=bnb_llama,
            device_map="auto",
            local_files_only=True
        )
        pipe_llama_short = pipeline(
            "text-generation",
            model=model_llama,
            tokenizer=tok_llama,
            max_new_tokens=128,
            return_full_text=False
        )
        pipe_llama_long = pipeline(
            "text-generation",
            model=model_llama,
            tokenizer=tok_llama,
            max_new_tokens=512,
            return_full_text=False
        )
        from langchain_huggingface import HuggingFacePipeline
        SLM_LLM_SHORT = HuggingFacePipeline(pipeline=pipe_llama_short)
        SLM_LLM_LONG = HuggingFacePipeline(pipeline=pipe_llama_long)
        return SLM_LLM_SHORT, SLM_LLM_LONG
    except Exception as e:
        print(f"Local Llama model setup failed: {e}")
        return None, None