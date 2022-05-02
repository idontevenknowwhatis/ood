import numpy as np
import pandas as pd
import transformers
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import re
import tqdm

from transformers import DistilBertModel, DistilBertTokenizer

from sklearn.metrics import f1_score, accuracy_score

class base_model(torch.nn.Module):

  def train_model(self, train_loader, optimizer,):
    with tqdm.notebook.tqdm(
      train_loader,
      unit="batch",
      total=len(train_loader)) as batch_iterator:

      total_loss = 0.0
      losses = []
      for iteration, data in enumerate(batch_iterator, start=1):
        ids = data['ids'].to(self.device)
        masks = data['mask'].to(self.device)
        labels = data['label'].to(self.device)

        optimizer.zero_grad()
        self.zero_grad()

        loss = self.compute_loss(ids, masks, labels)

        total_loss += loss.item()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        batch_iterator.set_postfix(mean_loss=total_loss / iteration, current_loss=loss.item())
      return losses, total_loss
  def test_model(self, test_loader, tolerance = 0.02):
    with tqdm.notebook.tqdm(
      test_loader,
      unit="batch",
      total=len(test_loader)) as batch_iterator:

      predictions_all = []
      labels_all = []

      for iteration, data in enumerate(batch_iterator, start=1):
        ids = data['ids'].to(self.device)
        masks = data['mask'].to(self.device)
        labels = data['label'].to(self.device)

        predictions = self.predict(ids, masks, labels, tolerance)

        predictions_all += predictions.tolist()
        labels_all += labels.tolist()

        accuracy = accuracy_score(predictions.tolist(), labels.tolist())

        batch_iterator.set_postfix(batch_accuracy=accuracy)

      f1 = f1_score(predictions_all, labels_all, average=None)
      accuracy = accuracy_score(predictions_all, labels_all)
      return  f1, accuracy
  def tolerance_spread(self, test_loader, lower, upper, step):
    f1s_per_epoch = []
    accuracy_per_epoch = []

    best_threshold = -100
    best_f1 = np.zeros(self.TOP_K)
    accuracy_of_best_f1 = -1

    for i in np.arange(lower, upper, step):
      f1, accuracy =  self.test_model(test_loader, tolerance=i)
      f1s_per_epoch.append(f1)
      if f1[-1] > best_f1[-1]:
        best_threshold = i
        best_f1 = f1
        accuracy_of_best_f1 = accuracy
      accuracy_per_epoch.append(accuracy)

    return np.asarray(f1s_per_epoch), accuracy_per_epoch, best_threshold, best_f1, accuracy_of_best_f1

class DECODER_OOD(base_model):
  def get_label_indices(self, sorted_labels_by_popularity, tokenizer, TOP_K):
    kept_labels = [sorted_labels_by_popularity[i].lower() for i in range(TOP_K)]

    tokenized_labels = tokenizer.encode_plus(kept_labels,
                                             None,
                                             add_special_tokens=False)['input_ids']
    tokenized_labels = torch.tensor(tokenized_labels, dtype=torch.long).unsqueeze(axis=-1)
    return tokenized_labels

  def __init__(self, labels, tokenizer, TOP_K, loss_train, loss_inference, device):
      super(DECODER_OOD, self).__init__()

      self.loss_train = loss_train
      self.loss_inference = loss_inference
      self.device = device
      self.TOP_K = TOP_K

      self.l1 = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
      self.l2 = torch.nn.Linear(768, 256).to(device)
      self.l3 = torch.nn.ReLU().to(device)
      self.l4 = torch.nn.Linear(256, 768).to(device)

      label_index_tensor = self.get_label_indices(labels, tokenizer, TOP_K).to(device)

      print('Label Indices: \n', label_index_tensor,'\n')

      self.label_embeddings = torch.squeeze( self.l1(label_index_tensor,
                                            torch.ones_like(label_index_tensor)
                                            ).last_hidden_state).detach()
      print("Label Embeddings Initialized", self.label_embeddings.shape)
      print("Label Embeddings:", self.label_embeddings)
  
  def forward(self, ids, mask):
    output_l1 = self.l1(ids, mask).last_hidden_state[:, 0]
    output_l2= self.l2(output_l1)
    output_l3 = self.l3(output_l2)
    output_l4 = self.l4(output_l3)
    return output_l4

  def compute_loss(self, ids, masks, labels):
    outputs = self.forward(ids, masks).to(self.device)
    loss = self.loss_train(outputs,
                           self.label_embeddings[labels])
    return loss
  
  def predict(self, ids, masks, labels, tolerance = 0.02):
    ood_label = self.label_embeddings.shape[0]

    comparison_embeddings = torch.unsqueeze(self.label_embeddings, axis=0)
    comparison_embeddings = comparison_embeddings.repeat(labels.shape[0], 1, 1).to(self.device)
    
    outputs = self.forward(ids, masks)
    outputs = torch.unsqueeze(outputs, 1).repeat(1, self.TOP_K, 1)

    # mean of element wise squared error
    distances = torch.mean(
        self.loss_inference(outputs, comparison_embeddings), axis=-1
    )

    past_tolerance, predictions = torch.min(distances, axis=1)
    past_tolerance = past_tolerance > tolerance

    ood_predictions = torch.where(past_tolerance, ood_label, predictions )

    return ood_predictions

class MSP_OOD(base_model):
  def __init__(self, TOP_K, device, use_softmax_for_prediction = False):
    super(MSP_OOD, self).__init__()
    self.device = device
    
    self.l1 = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
    self.l2 = torch.nn.Linear(768, 256).to(self.device)
    self.l3 = torch.nn.ReLU().to(self.device)
    self.l4 = torch.nn.Linear(256, TOP_K).to(self.device)

    self.use_softmax = use_softmax_for_prediction
    self.softmax = torch.nn.Softmax(dim=-1)

    self.loss = torch.nn.CrossEntropyLoss()
    self.TOP_K = TOP_K

    self.record_logits = False
    self.mean_logits = []
    

  def forward(self, ids, mask):
    output_l1 = self.l1(ids, mask).last_hidden_state[:, 0]
    output_l2= self.l2(output_l1)
    output_l3 = self.l3(output_l2)
    output_l4 = self.l4(output_l3)
    return output_l4

  def compute_loss(self, ids, masks, labels):
    
    outputs = self.forward(ids, masks).to(self.device)

    loss = self.loss(outputs, labels)

    return loss

  def predict(self, ids, masks, labels, tolerance = 0.2):
    ood_label = self.TOP_K

    outputs = self.forward(ids, masks).to(self.device)
    if self.use_softmax:
      outputs = self.softmax(outputs).to(self.device)

    past_tolerance, predictions = torch.max(outputs, axis=1)

    past_tolerance = past_tolerance < tolerance

    ood_predictions = torch.where(past_tolerance, ood_label, predictions )

    return ood_predictions
    
