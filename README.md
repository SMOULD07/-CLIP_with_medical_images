# CLIP for Medical Image Understanding — ROCO Evaluation and Fine-Tuning

Exploring OpenAI's [CLIP](https://github.com/openai/CLIP) model for image–text matching, with a focus on medical images from the [ROCO](https://github.com/razorx89/roco-dataset) radiology dataset.

This project was carried out as part of the **Master 2 VMI – Multimodalité et IA Générative** at Université Paris Cité.

*Based on the CLIP tutorial developed in 2024 with the help of Xiaoyang Wei, PhD student at LIPADE, Université Paris Cité.*

## Objective

Evaluate the performance of different CLIP architectures on medical images, compare pretrained and fine-tuned models on ROCO, and measure CLIP's zero-shot classification performance on natural images.

## What This Notebook Covers

### 1. CLIP on Natural Images

- Load the pretrained `ViT-B/16` model.
- Compute cosine similarity between 8 natural images from scikit-image and their text descriptions.
- Visualize the image–text similarity matrix.

### 2. CLIP on Medical Images

- Test CLIP on medical radiology images from ROCO.
- Compute image–text similarities to evaluate alignment between medical images and their descriptions.
- Observe the limitations of pretrained CLIP on medical images.

### 3. Comparison of CLIP Architectures

- Explore the 9 available CLIP architectures, including ResNet and Vision Transformer models.
- Evaluate `RN50`, `ViT-B/32`, `ViT-B/16` and `ViT-L/14` on 640 image–caption pairs from the ROCO test set.
- Compare Top-1 accuracy, Top-5 accuracy, contrastive loss, inference time and parameter count.
- Compare pretrained models with a fine-tuned `ViT-B/16` checkpoint.

### 4. Fine-Tuning CLIP on ROCO

- Implement a custom `CLRDataset` class to load ROCO images and captions.
- Define a contrastive learning objective using symmetric cross-entropy.
- Configure training with Adam, cosine annealing, batch size 16 and learning rate `3e-6`.
- Load a fine-tuned `ViT-B/16` checkpoint for evaluation.

**Note:** The training execution recorded in this notebook was interrupted. The fine-tuned evaluation results were obtained using a separately loaded checkpoint, not from the interrupted training run.

### 5. Medical Image–Text Matching After Fine-Tuning

- Select 8 additional images from the ROCO test set.
- Compare similarity matrices before and after fine-tuning.
- Evaluate image–text retrieval using a pool of 100 captions.
- Analyze the improvement in medical image–text alignment.

### 6. Zero-Shot Image Classification

- Evaluate pretrained `ViT-B/16` on the Oxford-IIIT Pet dataset (37 classes).
- Compute Top-1 and Top-5 classification accuracy without task-specific training.
- Perform an additional evaluation on CIFAR-100.

## Datasets

| Dataset | Type | Usage |
|---|---|---|
| [ROCO](https://github.com/razorx89/roco-dataset) | Medical radiology images and captions | Medical image–text retrieval and fine-tuning |
| [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) | Natural images (37 classes) | Zero-shot classification |
| CIFAR-100 | Natural images (100 classes) | Additional classification evaluation |
| scikit-image samples | Natural images (8 examples) | Image–text similarity demonstration |

## Tech Stack

Python, PyTorch, OpenAI CLIP, torchvision, scikit-image, NumPy, Pandas, Matplotlib, Google Colab (GPU).

## Key Results

### CLIP Model Comparison on ROCO

| Model | Top-1 | Top-5 |
|---|---:|---:|
| RN50 | 19.22% | 51.56% |
| ViT-B/32 | 23.44% | 55.00% |
| ViT-B/16 | 26.41% | 61.25% |
| ViT-L/14 | 35.94% | 74.38% |
| ViT-B/16 Fine-tuned | **63.28%** | **95.62%** |

*Evaluation on 640 image–caption pairs. Each image is matched against 64 candidate captions within its batch. The comparison uses shared image preprocessing, so results across different architectures should be considered indicative.*

### Medical Image–Text Matching

On 8 additional ROCO test images:

- Pretrained ViT-B/16: **3/8** correct matches.
- Fine-tuned ViT-B/16: **7/8** correct matches.

On a retrieval task with 100 image–caption pairs:

- Top-1 accuracy increased from **19% to 56%**.
- Top-5 accuracy increased from **48% to 92%**.

### Zero-Shot Classification

| Dataset | Top-1 | Top-5 |
|---|---:|---:|
| Oxford-IIIT Pet | 80.24% | 91.44% |
| CIFAR-100 | 62.76% | 85.53% |

## Conclusion

The experiments show that CLIP achieves good results on natural image classification but has more difficulties with medical image–text matching.

Among the evaluated pretrained models, `ViT-L/14` obtains the highest retrieval accuracy on ROCO.

The fine-tuned `ViT-B/16` checkpoint achieves better results on medical images, with Top-1 accuracy increasing from **26.41% to 63.28%** in the batch-based ROCO evaluation.

These results highlight the importance of domain adaptation for improving CLIP's performance on specialized medical images.
