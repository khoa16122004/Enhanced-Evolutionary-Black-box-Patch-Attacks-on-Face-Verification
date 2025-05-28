from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
import torch
import warnings
warnings.filterwarnings("ignore")

class LLava:
    def __init__(self, pretrained, model_name, tempurature=0):
        
        # llava-next-interleave-7b
        # llava-onevision-qwen2-7b-ov
        self.pretrained = f"lmms-lab/{pretrained}"
        self.model_name = model_name
        self.device = "cuda"
        self.device_map = "auto"
        self.llava_model_args = {
            "multimodal": True,
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        self.llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.pretrained, None, model_name, device_map=self.device_map, **self.llava_model_args)
        self.tempurature = tempurature
        self.model.eval()
    
    def reload(self):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.pretrained, None, self.model_name, device_map=self.device_map, **self.llava_model_args)
        self.model.eval()
        
    
    
    def inference(self, qs, img_files, num_return_sequences=1, do_sample=True, temperature=0, reload=True):
        # reload_llm
        if reload == True:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.reload()
        
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_tensors = process_images(img_files, self.image_processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
        image_sizes = [image.size for image in img_files]
        
        with torch.inference_mode():
            cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=4096,
            num_return_sequences=num_return_sequences,
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        outputs = text_outputs
        return outputs
    
    def get_image_embedding(self, img_files):
        image_tensors = process_images(img_files, self.image_processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
        with torch.inference_mode():
            image_embeddings = self.model.encode_images(image_tensors)  # (B, D)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
        return image_embeddings

    def get_text_embedding(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.inference_mode():
            embeddings = self.model.model.embed_tokens(input_ids)  # (1, seq_len, dim)
            pooled = embeddings.mean(dim=1)  # mean pooling
            pooled = F.normalize(pooled, dim=-1)
        return pooled  # (1, dim)

    def get_similarity(self, image_embedding, text_embedding):
        sim = torch.matmul(image_embedding, text_embedding.T)  # (B, 1)
        return sim.squeeze()  # trả về tensor 1 chiều (B,)