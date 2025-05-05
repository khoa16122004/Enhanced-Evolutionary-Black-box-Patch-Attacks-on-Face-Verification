from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
import torch
import warnings

warnings.filterwarnings("ignore")

class LLava:
    def __init__(self, pretrained, model_name, temperature=0):
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
        self.temperature = temperature
        self.model.eval()
        self.model.to(self.device)  # Ensure model is on the correct device
    
    def reload(self):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.pretrained, None, self.model_name, device_map=self.device_map, **self.llava_model_args)
        self.model.eval()
        self.model.to(self.device)  # Ensure model is on the correct device

    def inference(self, qs, img_files, temperature=0, reload=True):
        # reload_llm
        if reload:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.reload()
        
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        # Process images and move them to the device
        image_tensors = process_images(img_files, self.image_processor, self.model.config)
        image_tensors = [image.to(dtype=torch.float16, device=self.device) for image in image_tensors]
        
        # Ensure the images are not meta tensors
        for img in image_tensors:
            if img.is_meta:
                raise ValueError("Encountered a meta tensor. The image tensor is not properly initialized.")
        
        image_sizes = [image.size for image in img_files]
        
        with torch.inference_mode():
            cont = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=4096,
            )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        outputs = text_outputs
        return outputs
