from typing import Any

from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from transformers import pipeline
from llama_index.core.llms.callbacks import llm_completion_callback

class Llama(CustomLLM):
    num_output: int = 512
    model_name: str = "Llama"
    model: Any = None
    tokenizer: Any = None

    def __init__(self, model, tokenizer):
        super(Llama, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=generate_text(query=prompt, base_model= self.model, llama_tokenizer= self.tokenizer))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = ""
        for token in generate_text(query=prompt, base_model= self.model, llama_tokenizer= self.tokenizer):
            response += token
            yield CompletionResponse(text=response, delta=token)

def generate_text(query, base_model, llama_tokenizer):
    text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_length = 100000)
    output = text_gen(f"<s>[INST] {query} [/INST]")
    return format_text(output[0]['generated_text'])

import re

def format_text(input_text):

    formatted_text = re.sub(r'\*[^*]+\*', '', input_text)
    
    inst_end_index = formatted_text.find('[/INST]')
    if inst_end_index != -1:
        formatted_text = formatted_text[inst_end_index + len('[/INST]'):].strip()

    return formatted_text