import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from qwen.qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

class QWENModel:
    def __init__(self, model_name="Qwen/Qwen-1.8B-Chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            pad_token='<|extra_0|>',
                            eos_token='<|endoftext|>',
                            padding_side='left',
                            trust_remote_code=True
                        )
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        pad_token_id=self.tokenizer.pad_token_id,
                        device_map="auto",
                        trust_remote_code=True
                    ).eval()
        
        self.model.generation_config = GenerationConfig.from_pretrained('Qwen/Qwen-7B-Chat', pad_token_id=self.tokenizer.pad_token_id)
    
    def text_to_text(self, prompts, system_prompt): 
        """
            prompts: List['str']
            system_prompt: str
        """
        batch_raw_text = []
        for q in prompts:
            raw_text, _ = make_context(
                self.tokenizer,
                q,  
                system=system_prompt,
                max_window_size=self.model.generation_config.max_window_size,
                chat_format=self.model.generation_config.chat_format,
            )
            batch_raw_text.append(raw_text)
        batch_input_ids = self.tokenizer(batch_raw_text, padding='longest')
        batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(self.model.device)
        batch_out_ids = self.model.generate(
                            batch_input_ids,
                            return_dict_in_generate=False,
                            generation_config=self.model.generation_config
                        )
        padding_lens = [batch_input_ids[i].eq(self.tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]
        batch_response = [
            decode_tokens(
                batch_out_ids[i][padding_lens[i]:],
                self.tokenizer,
                raw_text_len=len(batch_raw_text[i]),
                context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
                chat_format="chatml",
                verbose=False,
                errors='replace'
            ) for i in range(len(prompts))
        ]
        
        return batch_response
        