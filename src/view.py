from get_architech import init_lvlm_model
import torch
from PIL import Image
import os
from dataset import get_dataset




def view(args):
    dataset = get_dataset(args.dataset)
    questions = [
            "Do the eyes of the two individuals have similar size and shape?",
            "Is there a noticeable difference in the nose length and width between the two individuals?",
            "Are the mouths of the two individuals similar in terms of lip thickness and symmetry?",
            "Do the facial structures, such as the jaw and chin, appear similar?",
            "Do the individuals have similar eyebrow shapes, density, or gaps between brows?"
    ]
    input_dir = f"question_pretrained={args.lvlm_pretrained}_modelname={args.lvlm_model_name}_dataset={args.dataset}_num_samples={args.num_samples}"
    
    




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory containing the images")
    args = parser.parse_args()

    main_with_detailed_questions(args)
