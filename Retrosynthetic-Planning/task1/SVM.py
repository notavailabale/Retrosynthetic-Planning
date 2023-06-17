import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

path = ''

train_dataset = torch.load(path + "train_filter_dataset.pt")
test_dataset = torch.load(path + "test_filter_dataset.pt")
val_dataset = torch.load(path + "val_filter_dataset.pt")

train_features = []
train_labels = []
test_features = []
test_labels = []
val_features = []
val_labels = []

for train_sample in train_dataset:
    train_feature, train_label = train_sample
    train_features.append(train_feature)
    train_labels.append(train_label)

for test_sample in test_dataset:
    test_feature, test_label = test_sample
    test_features.append(test_feature)
    test_labels.append(test_label)

for val_sample in val_dataset:
    val_feature, val_label = val_sample
    val_features.append(val_feature)
    val_labels.append(val_label)

label_num = preprocessing.LabelEncoder()
label_num.fit(train_labels + test_labels + val_labels)
templates_labels = list(label_num.classes_)
templates_nums = list(label_num.transform(templates_labels))
labels_nums = [None] * len(label_num.classes_)
for ind, template in zip(templates_nums, templates_labels):
    labels_nums[int(ind)] = template
with open(path + 'labels_nums_filter_SVM.csv', "w") as file:
    file.write("\n".join(labels_nums))

train_labels_num = label_num.transform(train_labels)
test_labels_num = label_num.transform(test_labels)
val_labels_num = label_num.transform(val_labels)

train_dataset = TensorDataset(torch.tensor(np.array(train_features, dtype=np.float32)),
                              torch.tensor(np.array(train_labels_num)))
test_dataset = TensorDataset(torch.tensor(np.array(test_features, dtype=np.float32)),
                             torch.tensor(np.array(test_labels_num)))
val_dataset = TensorDataset(torch.tensor(np.array(val_features, dtype=np.float32)),
                            torch.tensor(np.array(val_labels_num)))

train_dataset = ConcatDataset([train_dataset, val_dataset])
train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)

train_features, train_labels = next(iter(train_loader))
train_features = train_features.numpy()
train_labels = train_labels.numpy()

# Train the SVM model
model = SVC(kernel='rbf')
num_epochs = 1
loss_list = []

for epoch in range(num_epochs):
    # Train the model
    model.fit(train_features, train_labels)

    # Compute the loss
    train_loss = 1 - model.score(train_features, train_labels)
    loss_list.append(train_loss)

    # Print the loss for the current epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss))

plt.plot(loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# Evaluate the model on the test set
test_features = test_dataset.tensors[0].numpy()
test_labels = test_dataset.tensors[1].numpy()

accuracy = model.score(test_features, test_labels)
print('Accuracy of the SVM model: {} %'.format(100 * accuracy))
