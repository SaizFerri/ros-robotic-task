# Prüfungsvorleistung

## Structure

* [Description](#description)
* [Task 1](#task-1)
  * [ROS Introduction](#ROS Introduction)
  * [ROS Basics](#ROS Basics)
    * [Nodes](#Nodes)
    * [Topics](#Topics)
    * [Messages](#Messages)
  * [Task goal](#Task goal)
* Task 2
* Task 3



## Description

In this project we develop a ROS application capable of taking a single image or a video stream of images with handwritten digits, process it and predict the digit using a neuronal network. The project is divided in three tasks. Task 1 is responsible for setting the basic structure of the ROS application. In task 2 we develop the synchronization of the topics and the service for the neuronal network. Finally in task 3, we build and train the model used to predict the digits on the images using pytorch.

## Task 1

### ROS Introduction

ROS is an open-source, meta-operating system for robots. It provides the services from an operating system and also tools and libraries for obtaining, building, writing and running code across multiple computers.

### ROS Basics

#### Nodes

Nodes are one of the core building blocks of ROS. They are executables within a ROS package which can publish or subscribe to topics and can also provide or consume services.

#### Topics

Topics are channels in which messages are published. A node can publish or subscribe to a topic to send or recive messages.

#### Messages

Messages are sent through topics to nodes subscribed to certain topics. Messages can vary from primitive types such as  `int` to custom messages with custom fields and types.

#### Services

Services are another way that nodes can communicate with each other. Services allow nodes to send a request and receive a response. 

### Task goal

In this task we set up the structure of the project.

1. Create a `cam` and a `processor` nodes.
2. We set up the `cam`node to publish 2 topics:
   * `/camera/image` which publishes an image with a handwritten digit
   * `/camera/class` which publishes the value of the digit with a custom message type
3. The `processor` node subscribes to the `/camera/image` and processes the image. The image is converted to a greyscale image and is croped. Afterwards the processed image is published to the `/processed/image`.
4. We write a launch file to start both nodes with a single command

At the end of this task, our program structure looks like this:

<img src="task1-png" alt="Task 1" />