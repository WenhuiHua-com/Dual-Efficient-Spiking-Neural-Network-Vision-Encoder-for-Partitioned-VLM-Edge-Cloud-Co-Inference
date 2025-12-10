import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_itm import blip_itm
from models.blip import blip_decoder
import json
from timm.models import create_model
from model import *
from torch.cuda.amp import autocast
from spikingjelly.clock_driven import functional

with open('/media/data/huawenhui/BLIP-main/BLIP-main/flick/annotation/flickr30k_test.json', 'r') as file:
    data = json.load(file)  # Load the JSON file

def one_hot_to_indices(one_hot):
    """
    Restore indices from one-hot encoding.

    Args:
        one_hot (Tensor): One-hot encoded tensor, shape is (batch_size, num_classes).

    Returns:
        list[Tensor]: List of indices for each sample.
    """
    # Find indices of non-zero positions
    indices = [torch.nonzero(sample, as_tuple=False).squeeze(-1) for sample in one_hot]
    return indices


class CodeBook:
    def __init__(self, size, dim, device='cpu'):
        self.size = size
        self.dim = dim
        self.device = device
        self.book = torch.empty(0, dim, device=self.device)  # Initialize empty code-book
        self.weights = torch.empty(0, device=self.device)  # Store corresponding weights
        self.counter = 0  # Number of entries currently in the code-book

    def add(self, tokens, importance_scores):
        tokens = tokens.to(self.device)
        importance_scores = importance_scores.to(self.device)
        token_num, token_dim = tokens.shape

        if (self.counter + token_num) <= self.size:
            self.book = torch.cat([self.book, tokens], dim=0)
            self.weights = torch.cat([self.weights, importance_scores], dim=0)
            self.counter += token_num
        else:
            sim_matrix = torch.mm(tokens, self.book.T)
            for i, token in enumerate(tokens):
                sim_scores, indices = torch.topk(sim_matrix[i], k=1, largest=True)
                index = indices[0]

                # Weighted aggregation
                wa = self.weights[index]
                wb = importance_scores[i]
                self.book[index] = (wa * self.book[index] + wb * token) / (wa + wb)
                self.weights[index] = wa + wb  # Update merged weights

    def search(self, indices):  # The input now is not tokens, but indices
        indices = indices.to(self.device) 
        #ind = indices.argmax(dim=2)
        result_vectors = self.book[indices.squeeze()]  # Retrieve replacement vectors from the codebook
        result_vectors = result_vectors[1::]
        return result_vectors  # Return the replacement vectors after removing all-zero vectors

    def search_visual(self, tokens):
        tokens = tokens.to(self.device)
        sim_matrix = torch.mm(tokens, self.book.T)  # Compute similarity between input tokens and codebook
        # Step 2: Get the most similar codebook entry for each token
        sim_scores, indices = torch.topk(sim_matrix, k=1, largest=True)  # Get similarity scores and indices
        return indices.squeeze()  # Return the replacement vectors processed for zeroing

    def save(self, path):
        torch.save({'book': self.book, 'weights': self.weights}, path)  # Save codebook and weights

    def load(self, path):
        data = torch.load(path)
        self.book = data['book'].to(self.device)  # Load codebook
        self.weights = data['weights'].to(self.device)  # Load weights
        self.counter = self.book.size(0)


def test_codebook(ori_image, model, codebook):
    image_embeds_index = model(ori_image)
    token = codebook.search(image_embeds_index)
    return token


def load_demo_image(image_size, device, picture_name, image_root):
    raw_image = Image.open(os.path.join(image_root, picture_name)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


image_name_list = []
for item in data:
    image_name_list.append(item['image'])  # Add the image filename to the list
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_image_root = '/media/data/huawenhui/BLIP-main/BLIP-main/flick/flickr30k-images'


image_size = 224
model_url = '/media/data/huawenhui/BLIP-main/BLIP-main/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

codebook_size = 5000
embedding_dim = 768
codebook = CodeBook(size=codebook_size, dim=embedding_dim, device=device)
codebook.load('flicker_purning_05w_codebook.pth')
print('Loading codebook sets')
#print(f'Loading booksize = {codebook.book.shape}', )
spike_model = create_model(
        'sdt',
        pretrained=False,
        depths=6, sr_ratios=1
    )
print("Creating model")
#print(spike_model)
checkpoint_path = 'model_best.pth.tar'

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
spike_model.load_state_dict(checkpoint['state_dict'], strict=False)
spike_model.eval()
spike_model.to(device)
print('loading spikeformer successfully')

captions = []  # To store all captions for this sparsity level
count = 0
cos = []
for picture_name in image_name_list:
    with torch.no_grad():
        ori_image = load_demo_image(image_size=image_size, device=device, picture_name=picture_name,
                                        image_root=val_image_root)
        retrieved_tokens = test_codebook(ori_image, spike_model, codebook)
        caption = model.feature_generate(retrieved_tokens.unsqueeze(0))
        functional.reset_net(spike_model)
        captions.append(caption[0])  # Save caption

        count += 1

        if count % 100 == 0:
            print(picture_name, count, caption)

    del ori_image, retrieved_tokens
    torch.cuda.empty_cache()

# Calculate BPP based on sparsity percentage
BPP = ((1 - sparsity_percentages) * 197 * 19) / (224 * 224)

# Save the captions to a text file with BPP in the filename
filename = f"BPP_{BPP:.4f}_Ours_Caption_spike.txt"
with open(filename, "w") as f:
    for caption in captions:

        f.write(f"{caption}\n")
