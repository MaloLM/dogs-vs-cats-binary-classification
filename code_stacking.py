#######
# Stacking old code that could be usefull 
#######



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