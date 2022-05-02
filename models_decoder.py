import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import tqdm
import os
import bisect

from transformers import AutoModel, AutoTokenizer, AutoConfig

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


MODEL_NAME = 'decoder_ood'
CHECKPOINT_FOLDER = 'checkpoints'

# if recompute embeddings, do not give gradients to label embedding tensor
# and pass them to the language model instead. If the label embedding tensor
# should receive gradients, then the language model should not. If none
# then neither the language model nor the embedding tensor should receive.

class DECODER_OOD(nn.Module):
  def __init__(self, lm, hidden_dim, dropout, label_indices, TOP_K, device ):
      super(DECODER_OOD, self).__init__()

      self.embedding_loss = nn.MSELoss()
      self.classification_loss = nn.CrossEntropyLoss()

      self.label_index_tensor = torch.tensor(label_indices).to(device)
      self.TOP_K = TOP_K
      self.device = device
      
      # Initialize LM model type
      # config = AutoConfig.from_pretrained(lm, num_labels = 768, output_hidden_states=True)
      self.l1 = AutoModel.from_pretrained(lm).to(device)

      # Causes issues with regular text inputs without use_fast
      tokenizer = AutoTokenizer.from_pretrained(lm, use_fast=False)

      # Initialize the Label Embeddings

      forward = self.l1(self.label_index_tensor)

      print(forward.last_hidden_state[:, 0].shape)
      
      
      self.label_embeddings = torch.mean(forward.last_hidden_state, dim = 1).detach().to(device)

      # Rest of the layers
      self.l2 = nn.Dropout(p=dropout).to(device)
      self.l3 = nn.Linear(in_features = self.label_embeddings.shape[-1], out_features=self.label_embeddings.shape[-1],).to(device)
      self.l4 = nn.GELU().to(device)
      self.l5 = nn.LayerNorm(self.label_embeddings.shape[-1])
      self.l6 = nn.Dropout(p=dropout).to(device)
      self.l7 = nn.Linear(in_features = self.label_embeddings.shape[-1], out_features=TOP_K).to(device)

      # Calibrated logits for prediction percentages
      self.baseline_logits = None
      self.softmax = nn.Softmax(dim = -1)

  def get_optimizer_parameters(self,):
    return list(self.l3.parameters()) + list(self.l1.parameters()) + list(self.l7.parameters())
    # return self.parameters()
    

  def forward(self, ids, mask):
    x1 = self.l1(ids, mask).last_hidden_state[:, 0]
    x2 = self.l2(x1)
    x3 = self.l3(x2)
    x4 = self.l4(x3)
    x5 = self.l5(x4)
    x6 = self.l6(x5)
    x7 = self.l7(x6)
    return x3.to(self.device), x7.to(self.device)

  def predict(self, ids, masks, labels):
    self.eval()
    with torch.no_grad():
      ood_label = self.TOP_K
      _, logits = self.forward(ids, masks)

      logits = torch.sigmoid(logits)

      maximal_logits, id_predictions = torch.max(logits, axis=1)
      prediction_logits = torch.cat((logits, maximal_logits[:, None]), dim=1)

      return prediction_logits, id_predictions,

  def compute_confidences(self, test_loader):
    self.eval()
    torch.manual_seed(0)

    with tqdm.notebook.tqdm(
      test_loader,
      unit="batch",
      total=len(test_loader)) as batch_iterator:

      confidences = []
      labels_all = []

      self.eval()

      for iteration, data in enumerate(batch_iterator, start=1):
        ids, masks, labels = data
        ids = ids.to(self.device)
        masks = masks.to(self.device)
        labels_all += labels.tolist()

        prediction_logits, predictions = self.predict(ids, masks, labels)

        prediction_logits = 1- prediction_logits
        
        confidences += prediction_logits[:, -1].tolist()

        accuracy = accuracy_score(predictions.tolist(), labels.tolist())
        batch_iterator.set_postfix(accuracy = accuracy)
        
    return confidences, labels_all

  def compute_auc(self, loader):
    confidences, labels = self.compute_confidences(loader)
    filtered_labels = []
    for label in labels:
      if label >= self.TOP_K:
        filtered_labels.append(1)
      else:
        filtered_labels.append(0)
    return roc_auc_score(filtered_labels, confidences)


  def train_model(self, train_loader, optimizer, epochs=1, val_loader = None, test_loader = None , eval_interval = -1):
    epoch_losses = []

    torch.manual_seed(0)

    with tqdm.notebook.tqdm(range(epochs), unit="epoch", total = epochs) as epoch_iterator:
      for epoch in epoch_iterator:
        with tqdm.notebook.tqdm(
          train_loader,
          unit="batch",
          total=len(train_loader)) as batch_iterator:

          total_loss = 0.0
          losses = []
          accuracies = []

          self.train()
          self.debug = False

          for iteration, data in enumerate(batch_iterator, start=1):
            ids, masks, labels = data
            ids = ids.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            self.zero_grad()

            embedding, logits = self.forward(ids, masks)
            embedding_loss = self.embedding_loss(embedding, self.label_embeddings[labels])
            classification_loss = self.classification_loss(logits, labels)
            loss = 0 * embedding_loss + classification_loss * 0.2

            id_predictions = torch.argmax(logits, dim = 1)
            accuracy = accuracy_score(id_predictions.tolist(), labels.tolist())

            accuracies.append(accuracy)

            losses.append(loss.item())
            total_loss += loss.item()

            if eval_interval > 0 and iteration % eval_interval == 0:
              if self.debug:
                print("ids", ids[0])
                print("mask", masks[0])
                print("labels", labels[0])
                print("distances", distances[0])
                print("predictions", predictions[0])

              if val_loader is not None:
                print("Iteration:", iteration)
                print("VAL:", self.compute_auc(val_loader))
              if test_loader is not None:
                print("TEST:", self.compute_auc(test_loader))

              

            loss.backward()
            optimizer.step()
            batch_iterator.set_postfix(mean_loss=total_loss / iteration, current_loss=loss.item(), accuracy = accuracy, mean_accuracy = np.mean(accuracies))

          epoch_losses.append(losses)
          
    return epoch_losses


      



