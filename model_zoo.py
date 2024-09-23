# coding=utf-8
"""Model definition."""
# pylint: disable=g-multiple-import,g-importing-member,g-bad-import-order,missing-class-docstring

from typing import Optional
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

#This is a base class skeleton for all models in the model zoo
class ModelZoo:

  # Transform an image input
  def transform(self, image):
    pass
  
  # Process an image tensor
  def transform_tensor(self, image_tensor):
    pass

  # Calculate the loss between the output and target images
  def calculate_loss(
      self, output, target_images
  ):
    pass

  # Compute the probability of similarity between an output and target images
  def get_probability(
      self, output, target_images
  ):
    pass

#This class is a wrapper for the CLIP model
#The main objective of this class is to compute image features, losses and probabilities using the CLIP model to measure image similarity.
class CLIPImageSimilarity(ModelZoo):

  # The constructor loads a pre-trained CLIP model from Hugging Face
  def __init__(self):
    # initialize classifier
    self.clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to("cuda")
    # Initialize the CLIP processor, which will pre-process the input images
    self.clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

  # Takes in an image, preprocesses it with CLIP processor. Returns a tensor of image pixel values that can be fed into the CLIP model.
  # The pre-processed image is moved to the GPU
  def transform(self, image):
    images_processed = self.clip_processor(images=image, return_tensors="pt")[
        "pixel_values"
    ].cuda()
    return images_processed

  # Takes in an image tensor and resizes it to 224x224 pixels, which is the input size required for CLIP.
  def transform_tensor(self, image_tensor):
    image_tensor = torch.nn.functional.interpolate(
        image_tensor, size=(224, 224), mode="bicubic", align_corners=False
    )
    # Normalize the image tensor using CLIP's standard mean and deviation values.
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    image_tensor = normalize(image_tensor)
    return image_tensor

  # Calculate the Cosime similarity loss between the output and target images
  def calculate_loss(
      self, output, target_images
  ):
    # calculate CLIP loss
    output = self.clip_model.get_image_features(output)
    # loss = -torch.cosine_similarity(output, input_clip_embedding, axis=1)

    mean_target_image = target_images.mean(dim=0).reshape(1, -1)
    loss = torch.mean(
        torch.cosine_similarity(
            output[None], mean_target_image[:, None], axis=2
        ),
        axis=1,
    )
    loss = 1 - loss.mean()
    return loss

  def get_probability(
      self, output, target_images
  ):
    output = self.clip_model.get_image_features(output)
    mean_target_image = target_images.mean(dim=0).reshape(1, -1)
    loss = torch.mean(
        torch.cosine_similarity(output[None], mean_target_image, axis=2), axis=1
    )
    return loss.mean()