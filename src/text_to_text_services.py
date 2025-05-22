import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
import os

class QwenService:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def text_to_text(self, system_prompt, prompt):
        inputs = f"{system_prompt}\n{prompt}"
        input_ids = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
        outputs = self.model.generate(input_ids, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class GPTService:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def text_to_text(self, system_prompt, prompt):
        inputs = f"{system_prompt}\n{prompt}"
        input_ids = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
        outputs = self.model.generate(input_ids, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class LlamaService:
    def __init__(self, model_name):
        self.cls_mapping = {
                "Llama-7b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-7b-chat-hf", "meta-llama"),
                "Llama-13b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-13b-chat-hf", "meta-llama"),
                "Mistral-7b": (MistralForCausalLM, AutoTokenizer, True, "Mistral-7B-Instruct-v0.2"),
                "vicuna-7b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-7b-v1.5"),
                "vicuna-13b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-13b-v1.5"),
                "gemma-7b": (AutoModelForCausalLM, AutoTokenizer, True, "gemma-7b-it")
            }
        self.model_name = model_name
        model_cls, tokenizer_cls, self.is_decoder, hf_name, prefix = self.cls_mapping[model_name]
        self.model = model_cls.from_pretrained(os.path.join(prefix, hf_name)).cuda()
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(prefix, hf_name))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generate_kwargs = dict(
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            do_sample=False,  # greedy decoding
            top_p=None,
            temperature =None,
            
        )
        if self.is_decoder:
            self.tokenizer.padding_side = "left"

    def text_to_text(self, system_prompt, prompt):
        inputs = f"[INST]{system_prompt}/n{prompt}[/INST]Answer: "
        input_ids = self.tokenizer(inputs, 
                                   return_tensors="pt", 
                                   padding=True, truncation=True)
        outputs = self.model.generate(input_ids=input_ids.input_ids.to(self.model.device), 
                                      attention_mask=input_ids.attention_mask.to(self.model.device), 
                                      **self.generate_kwargs)
        decoded_outptus = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        if isinstance(decoded_outptus, list):
            return [o.split("Answer:")[-1].strip() for o in decoded_outptus]
        else:
            return decoded_outptus.split("Answer:")[-1].strip()
