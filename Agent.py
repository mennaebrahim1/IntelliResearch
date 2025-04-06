import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

def initialize_assistant_pipeline():
    """
    Initialize the pipeline with a pre-trained model and tokenizer for an AI assistant.

    Returns:
        pipeline: The text-generation pipeline for AI assistant.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    MODEL_NAME = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
    )

    assistant_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return assistant_pipeline

def ask_ai_assistant(question, max_new_tokens=4096, temperature=0.1, top_p=0.9, top_k=20, repetition_penalty=1.1):
    """
    Interact with the AI assistant by providing a structure prompt and asking a question.

    Args:
        prompt (str): The initial prompt defining the AI assistant's structure and role.
        question (str): The specific question for the AI assistant.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        top_k (int): Number of highest probability vocabulary tokens to keep for top-k filtering.
        repetition_penalty (float): Penalty for repetition.

    Returns:
        str: The generated response from the AI assistant.
    """
    prompt = """
    You are an advanced AI assistant. Your purpose is to provide accurate, detailed, and well-referenced answers to questions.
    Very Important Note:Your responses must include:
    - Accurate and concise answers to the user's question.
    - References with full details such as:
    - Source name (e.g., book, article, or website).
    - Publication year (if available).
    - Direct links (URLs or DOIs) to the referenced material to enable verification.
    If you are unsure of a specific reference, indicate this clearly in your response.
    """
    assistant_pipeline = initialize_assistant_pipeline()
    input_text = f"{prompt.strip()}\n\n{question.strip()}"

    response = assistant_pipeline(
        input_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )

    return response[0]['generated_text']
