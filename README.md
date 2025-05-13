# ML_Final_Project
This project implements a complete end-to-end pipeline for detecting and generating fake news using machine learning and large language models (LLMs). It covers data preparation, synthetic fake news generation using the OpenAI API, model training, evaluation, and deployment of a simple prediction and generation interface.

Key Features：

Real News Dataset: Utilizes a large set of real news articles as the base dataset.

Synthetic Fake News Generation: Automatically generates 15,000 high-quality fake news articles using OpenAI’s GPT API.

Fake News Detector: Trains a Logistic Regression classifier with TF–IDF vectorization to effectively distinguish between real and fake news, achieving near-perfect evaluation metrics.

ROC-AUC Evaluation: Includes advanced performance evaluation using precision, recall, F1-score, and ROC-AUC.

Fake News Generator: Provides a standalone fake news generator using OpenAI API, allowing realistic fake articles to be created from custom prompts.

Deployment Ready: Includes scripts for saving and loading models, and a simple command-line interface for real-time fake news prediction.

Team member: Ziyi Liu, Yide Fang, Hanzhe Zhou
