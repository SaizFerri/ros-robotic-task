#!/usr/bin/env python

from __future__ import print_function
import os
import time
import copy

import numpy as np
import torch
# import torchvision
import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from PIL import Image

from smart_camera.srv import Ai, AiResponse
import rospy

def handle_request(req):
    response = 0
    rospy.loginfo(rospy.get_caller_id() + " Request -> %s", req.image)
    return AiResponse(response)

def ai_server():
    rospy.init_node('ai_server')
    s = rospy.Service('ai', Ai, handle_request)
    print("Ready to classify an image.")
    rospy.spin()

def load_image(img):
    loader = transforms.Compose([transforms.ToTensor()])
    return loader(image).float()

def generate_model():
  return nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
  )

def validate(img):
  model = generate_model()
  model = model.to(device)
  model.load_state_dict(torch.load('best_mnist_w.pt'))

  with torch.no_grad():
    image = img.to(device)

    image = image.view(image.shape[0], -1)

    output = model(image)
    _, preds = torch.max(output, 1)

    return preds.item()

if __name__ == "__main__":
    ai_server()