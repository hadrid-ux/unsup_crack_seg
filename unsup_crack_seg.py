import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from mat_extract import descriptor_mat
from torch_geometric.data import Data
from extractor import ViTExtractor
from gnn_pool import GNNpool
import torch.optim as optim
from tqdm import tqdm
import util
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit
import urllib.request
import warnings
import math

def segment_image(uploaded_image):
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    K = 2
    epoch = 10
    res = (224, 224)
    stride = 4
    facet = 'key'
    layer = 11
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_bin = False
    cc = False

    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)
    uploaded_image = uploaded_image.convert('RGB')
    prep = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image_tensor = prep(uploaded_image)[None, ...]
    image_np = np.array(uploaded_image)
    extractor = ViTExtractor('dino_vits8', stride, model_dir=pretrained_weights, device=device)
    feats_dim = 384
    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()
    W, F, D = descriptor_mat(image_tensor, extractor, layer, facet, bin=log_bin, device=device)
    node_feats, edge_index, edge_weight = util.load_data(W, F)
    data = Data(node_feats, edge_index, edge_weight).to(device)
    model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
    opt = optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(epoch):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()
    S = S.detach().cpu()
    S = torch.argmax(S, dim=-1)
    mask0, S = util.graph_to_mask(S, cc, stride, image_tensor, image_np)
    mask0_image = Image.fromarray(mask0 * 255).convert('L')

    white_pixels = np.sum(mask0 == 1)
    total_pixels = mask0.shape[0] * mask0.shape[1]
    severity = white_pixels / total_pixels

    # Assign a severity level from 1 to 5
    if severity < 0.2:
        level = 1
    elif severity < 0.4:
        level = 2
    elif severity < 0.6:
        level = 3
    elif severity < 0.8:
        level = 4
    else:
        level = 5

    mask0_image = Image.fromarray(mask0 * 255).convert('L')
    return mask0_image, level

# Streamlit UI
st.title('Crack Segmentation App 🚨')

# Introduction
st.write('**Welcome to the Crack Segmentation App!**')
st.write('**This app is designed to perform crack segmentation on images.**')

# Image upload
st.write('Upload an image for crack segmentation:')
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Add a button for the user to perform segmentation
    if st.button('Perform Segmentation'):
        # Perform crack segmentation
        image = Image.open(uploaded_image)
        segmented_image, severity_level = segment_image(image)

        # Display the segmented image
        st.write('**Segmented Image:**')
        st.image(segmented_image, caption='Segmented Image', use_column_width=True)

        # Display the severity level
        st.header(f'Severity level: {severity_level}')
st.text('')  # Add an empty line to separate sections
