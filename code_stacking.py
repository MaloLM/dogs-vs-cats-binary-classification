#######
# Stacking old code that could be usefull 
#######


#===================================================

class CNNBinaryClassifier(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super(CNNBinaryClassifier, self).__init__()
        
        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(64) 

        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.pool22 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4 (with an additional convolutional layer)
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bnorm4 = nn.BatchNorm2d(256)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output (1, 1) spatially for each channel

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bnorm1(x)
        x = self.pool22(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.bnorm3(x)
        x = self.pool22(x)

        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.bnorm4(x)
        x = self.pool22(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(-1, 256)  # Flatten for fully connected layers

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Binary classification output

        return x

summary(CNNBinaryClassifier().cuda(), input_size = [(3, 128, 128)])

#===================================================

class CNNBinaryClassifier(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super(CNNBinaryClassifier, self).__init__()
        
        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)  # BatchNorm after last conv in block
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling by 2

        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4 (with an additional convolutional layer)
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # Extra layer
        self.bnorm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output (1, 1) spatially for each channel

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bnorm1(x)
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bnorm2(x)
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.bnorm3(x)
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))  # Additional layer
        x = self.bnorm4(x)
        x = self.pool4(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(-1, 256)  # Flatten for fully connected layers

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Binary classification output

        return x

summary(CNNBinaryClassifier().cuda(), input_size = [(3, 128, 128)])

#===================================================

predictions = []
true_labels = []

ort_session = ort.InferenceSession(import_path, providers=['CPUExecutionProvider'])

classes = ['Dog', 'Cat']


for images, labels in test_loader:

    for image, label in zip(images, labels):
  
        image = image.unsqueeze(0).cpu()

        outputs = ort_session.run(None, {'input': image.numpy()})
  
        predicted = classes[0] if outputs[0][0] <= 0.5 else classes[1]

        label_value = label.cpu().detach().numpy().item()
        actual = classes[label_value]

        predictions.append(predicted)
        true_labels.append(actual)

accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)

cm = confusion_matrix(true_labels, predictions, labels=classes)

plot_confusion_matrix(cm, title=f"Model Confusion matrix\nAccuracy: {accuracy * 100:.1f}%", normalize=True)

#===================================================

#TRAIN_SIZE = 12000
#VAL_SIZE = 100
#TEST_SIZE = 1000

#===================================================

def create_data_loaders(dataset, batch_size, train_size, val_size, test_size, generator=None):
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)

    return train_loader, val_loader, test_loader


class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, dogs_dir, cats_dir, transform=None):
        self.nb_classes = 2
        self.dogs_dir = dogs_dir
        self.cats_dir = cats_dir
        self.transform = transform
        self.dog_images = [os.path.join(dogs_dir, img) for img in os.listdir(dogs_dir) if img.endswith(".jpg")]
        self.cat_images = [os.path.join(cats_dir, img) for img in os.listdir(cats_dir) if img.endswith(".jpg")]

        nb_sample_per_class = int((train_size / 2) + (val_size / 2) + (test_size / 2))
  
        self.images = self.dog_images[:nb_sample_per_class] + \
                      self.cat_images[:nb_sample_per_class]
        self.labels = [0] * len(self.dog_images[:nb_sample_per_class]) + \
                      [1] * len(self.cat_images[:nb_sample_per_class])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# Dataset init
# dataset = CatDogDataset(dogs_dir=dogs_path, cats_dir=cats_path, transform=data_transform)

# DataLoaders creation
#if train_on_gpu:
#    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=32, train_size=train_size, val_size=val_size, test_size=test_size, generator=generator)
#else:
#    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=32, train_size=train_size, val_size=val_size, test_size=test_size)


# Dataset init
# train_dataset = CatDogDataset(dogs_dir=dogs_path, cats_dir=cats_path, transform=train_transforms)
dataset = CatDogDataset(dogs_dir=dogs_path, cats_dir=cats_path, transform=data_transform)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

batch_size = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)