#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cat, no_grad, manual_seed
from torch.utils.data import Dataset, DataLoader
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)

import torchvision.models as models
from torchvision import datasets, transforms

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import real_amplitudes, zz_feature_map, z_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
# qiskit_algorithms imports removed: unused here, and the package is not part of
# the qiskit 2.x / qiskit-machine-learning 0.9 stack.
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

algorithm_globals.random_seed = 3407
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        tumor_dir = os.path.join(root_dir, 'yes')
        for img_name in os.listdir(tumor_dir):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):  # Assuming image formats are PNG or JPG
                self.images.append(os.path.join(tumor_dir, img_name))
                self.labels.append(1)  

        no_tumor_dir = os.path.join(root_dir, 'no')
        for img_name in os.listdir(no_tumor_dir):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                self.images.append(os.path.join(no_tumor_dir, img_name))
                self.labels.append(0)  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize(130),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = BrainDataset(root_dir='./data/Br35H/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = BrainDataset(root_dir='./data/Br35H/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

val_dataset = BrainDataset(root_dir='./data/Br35H/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


# In[3]:


n_samples_show = 6

data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[n_samples_show - 1].imshow(images[0, 0].numpy().squeeze(), cmap="gray")
    axes[n_samples_show - 1].set_xticks([])
    axes[n_samples_show - 1].set_yticks([])
    axes[n_samples_show - 1].set_title("Labeled: {}".format(targets[0].item()))

    n_samples_show -= 1


# In[3]:


###### Define and create QCNN ######

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


# In[4]:


def create_qnn():

    feature_map = z_feature_map(8)

    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn


qnn4 = create_qnn()


# In[5]:


class Net(nn.Module):
    def __init__(self, qnn):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d()
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 8)    
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)   

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        return torch.cat((x, 1 - x), -1)

model4 = Net(qnn4)


# In[7]:


# Define model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model4.to(device)

optimizer = optim.NAdam(model4.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

# Start training
epochs = 300  # Set number of epochs
loss_list = []  # Store loss history
model4.train()  # Set model to training mode

for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = model4(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss
    loss_list.append(sum(total_loss) / len(total_loss))
           
    avg_train_loss = sum(total_loss) / len(total_loss)

    model4.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model4(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)   

    val_accuracy = 100. * correct / total

    with open("training_performance_q8.txt", "a") as file:
        file.write(f"{epoch+1}, {avg_train_loss:.4f}, {val_accuracy:.2f}%\n")

    print(f"Epoch: {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    if epoch > 2:
        if loss_list[-1] < min(loss_list[:-1]):
            torch.save(model4.state_dict(), "q8.pt")
            print("Saved")
            


# In[8]:


# Plot loss convergence
plt.plot(loss_list)
plt.title("Hybrid NN Training Convergence")
plt.xlabel("Training Iterations")
plt.ylabel("Neg. Log Likelihood Loss")
plt.show()


# In[6]:


model4.load_state_dict(torch.load("q8.pt"))

precision_list = []
recall_list = []
f1_list = []

model4.eval()  # set model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        output = model4(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        all_preds.extend(pred.view(-1).tolist())
        all_targets.extend(target.view(-1).tolist())

    accuracy = 100. * correct / total


    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    print(
        "Performance on test data:\n\tAccuracy: {:.1f}%\n\tPrecision: {:.1f}%\n\tRecall: {:.1f}%\n\tF1 Score: {:.1f}%".format(
            accuracy, precision * 100, recall * 100, f1 * 100
        )
    )

with open("model_performance_q8.txt", "w") as file:
    file.write("Performance on test data:\n")
    file.write("\tAccuracy: {:.1f}%\n".format(accuracy))
    file.write("\tPrecision: {:.1f}%\n".format(precision * 100))
    file.write("\tRecall: {:.1f}%\n".format(recall * 100))
    file.write("\tF1 Score: {:.1f}%\n".format(f1 * 100))


# In[ ]:


# Plot predicted labels

n_samples_show = 8
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model4.eval()
with no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model4(data[0:1])
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)

        axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title("Predicted {}".format(pred.item()))

        count += 1


# In[ ]:





# In[ ]:




