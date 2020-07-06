#!/usr/bin/env python

from __future__ import print_function
import os
import time
import copy

import rospy

import cv2
from cv_bridge import CvBridge

import numpy as np
import torch
import torchvision
import torch.nn as nn
# import torch.optim as optim
from torchvision import transforms
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from PIL import Image

from smart_camera.srv import Ai, AiResponse
import rospy

def handle_request(req):
  bridge = CvBridge()
  image = bridge.imgmsg_to_cv2(req.image)
  np_image = np.array(image, dtype=float) / 255

  tensor_image = load_image(np_image)
  response = validate(tensor_image)
  # rospy.loginfo(rospy.get_caller_id() + " Request -> %s", image)
  return AiResponse(response)

def ai_server():
  rospy.init_node('ai_server')
  s = rospy.Service('ai', Ai, handle_request)
  print("Ready to classify an image.")
  rospy.spin()

def load_image(img):
  loader = transforms.Compose([transforms.ToTensor()])
  return loader(img).float()

def generate_model():
  return nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
  )

def validate(image):
  device = torch.device('cpu')
  model = generate_model()
  model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + '/weights.pt', map_location=device))

  with torch.no_grad():
    image = image.view(image.shape[0], -1)

    output = model(image)
    _, preds = torch.max(output, 1)

    return preds.item()

if __name__ == "__main__":
  ai_server()