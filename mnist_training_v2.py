import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import itertools

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

device = torch.device("cpu")

class BinarizeTransform:
    """
    A class to binarize the input MNIST data.
    """
    def __call__(self, img):
        # Values are between 0 and 1 so I have binarized with threshold of 0.5
        return (img>0.5).float()

# Transform to be applied on to the data immediately after loading from location.
transform = transforms.Compose([
    transforms.ToTensor(),
    BinarizeTransform()
])

# Load and transform MNIST data
mnist_data_train = datasets.MNIST(root="data/MNIST", train=True, download=True, transform=transform)
mnist_data_test = datasets.MNIST(root="data/MNIST", train=False, download=True, transform=transform)


train_size = int(0.9 * len(mnist_data_train))       # Size of the train split
val_size = len(mnist_data_train) - train_size       # Size of the validation split


train_data, val_data = random_split(mnist_data_train, [train_size, val_size])

json_data = {
    "365nm":{
        "I1":range(0, 5),
        "I2":range(10, 15),
        "I3":range(18, 23),
        "I4":range(25, 30)
    },
    "455nm":{
        "I1":range(0, 5),
        "I2":range(7, 12),
        "I3":range(14, 19),
        "I4":range(21, 26)
    },
    "White":{
        "I1":range(0, 5),
        "I2":range(9, 14),
        "I3":range(16, 21),
        "I4":range(24, 29)
    }
}


class CustomDataset(Dataset):
    def __init__(self, data, combined_table:pd.DataFrame, device_masks, mask_set):
        self.processed_data = []
        self.labels = []

        for idx in tqdm(range(len(data))):
            image, label = data[idx]
            image = image.reshape(196, 4)
            optical_pulses = (
                image[ :, 0] * 1000 + 
                image[ :, 1] * 100 + 
                image[ :, 2] * 10 + 
                image[ :, 3]
            ).repeat(len(mask_set))
            row_indices = torch.tensor([(device_masks[i]+mask_set[i]*5).tolist() for i in range(len(mask_set))]).flatten()
            column_indices = combined_table.columns.get_indexer(optical_pulses.tolist())
            
            self.processed_data.append(   combined_table.values[row_indices, column_indices] *1e9 )
            self.labels.append(label)
        self.processed_data = torch.tensor(np.array(self.processed_data)).to(device=device)
        self.labels = torch.tensor(self.labels).to(device=device)

    def __len__(self):
        return self.processed_data.shape[0]
    def __getitem__(self, index):
        return self.processed_data[index], self.labels[index]


class ReadoutLayer(nn.Module):
    """Readout layer.
    Parameters - 

    Attributes - 
    fc : `nn.Linear`
            Fully connected layer to be trained for reservoir computing.
    activation : `nn.functional.leaky_relu`
            Activation layer to be applied
    
    softmax : `nn.Softmax`
            Softmax activation function to get the one-hot encoded results.
    
    """
    def __init__(self, input_size):
        # super function to initialize the constructors of the parent classes.
        super(ReadoutLayer, self).__init__()
        # Class Attributes
        self.fc = nn.Linear(input_size, 10) 
        self.activation = nn.functional.leaky_relu
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward method to be executed on function call.
        Parameters - 
        x : torch.tensor
            Shape( Batch_size, 196 * len(mask_sets) )
        
        Returns : torch.tensor
            Shape( Batch_size, 10 )
        """
        x = self.fc(x)
        x = self.activation(x)
        x = self.softmax(x)
        return x
    

def get_metrics(filename, mask_set, device_masks):
    path = "data/"+filename+".xlsx" # Excel doc path

    df = pd.read_excel(path, usecols='B:Q') # Read the excel sheet
    # Break the sheet down into different `DataFrame`s for every Mask-set in the table
    tables = [df.iloc[json_data[filename][key]].copy().reset_index(drop=True) for key in list(json_data[filename].keys())]
    combined_table = pd.concat(tables, axis=0)

    train_dataset = CustomDataset(train_data, combined_table, device_masks, mask_set)
    validation_dataset = CustomDataset(val_data, combined_table, device_masks, mask_set)
    test_dataset = CustomDataset(mnist_data_test, combined_table, device_masks, mask_set)

    BATCH_SIZE = 64
    EPOCHS = 100
    learning_rate = 0.001
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    model = ReadoutLayer(len(mask_set)*196).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    val_accuracy, val_precision, val_recall, val_fscore = [], [], [], []

    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", num_classes=10, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=10, average='macro').to(device)
    f1_score = F1Score(task="multiclass", num_classes=10, average='macro').to(device)

    # Class-wise Confusion matrix
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=10).to(device)

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to device
            outputs = model(images.float())  # Forward pass
            loss = criterion(outputs, labels)  # Loss calculation
            optimizer.zero_grad()
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.float())
                preds = outputs.argmax(dim=1)

                # Update metrics
                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1_score.update(preds, labels)

            # Print validation metrics
            print(f'Validation Accuracy: {accuracy.compute().item():.4f} Precision: {precision.compute().item():.4f} ', end="")
            print(f'Validation Recall: {recall.compute().item():.4f} F1 Score: {f1_score.compute().item():.4f}')

            # Updating the list to save current metrics
            val_accuracy.append(accuracy.compute().item())
            val_precision.append(precision.compute().item())
            val_recall.append(recall.compute().item())
            val_fscore.append(f1_score.compute().item())

            # Reset metrics for the next epoch
            accuracy.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()
            confusion_matrix.reset()

    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Move images and labels to GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(torch.tensor(images, dtype=torch.float32))
            _, predicted = torch.max(outputs, 1)

            # Append predictions and labels for metric calculations
            all_preds.append(predicted)
            all_labels.append(labels)


    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds).to(device)
    all_labels = torch.cat(all_labels).to(device)

    # Calculate metrics
    test_accuracy = accuracy(all_preds, all_labels)
    test_precision = precision(all_preds, all_labels)
    test_recall = recall(all_preds, all_labels)
    test_f1 = f1_score(all_preds, all_labels)
    test_confusion_matrix = confusion_matrix(all_preds, all_labels)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test Precision: {test_precision*100:.4f}%')
    print(f'Test Recall: {test_recall*100:.4f}%')
    print(f'Test F1 Score: {test_f1:.4f}')
    print("Confusion Matrix:")
    print(test_confusion_matrix)

    save_path = "data/mnist_results_debug/" + filename + '_' + ''.join(map(str, mask_set)) + '.npz'
    np.savez(save_path,
            predictions = all_preds,
            labels = all_labels,
            validation_accuracy = val_accuracy,
            validation_precision = val_precision,
            validation_recall = val_recall,
            validation_fscore = val_fscore
            
    )
    return test_accuracy, test_precision, test_recall, test_f1

        

def get_all_combinations(lis):
    all_combinations = []
    for r in range(1, len(lis)+1):
        combs = list(itertools.combinations(lis, r))
        all_combinations.extend(combs)
    return [list(ele) for ele in all_combinations]

metrics_json = {
    "White":{},
    "365nm":{},
    "455nm":{}
}
mask_sets = [0, 1, 2, 3]
combinations = get_all_combinations(mask_sets)
for filename in ["White", "365nm", "455nm"]:
    for ms in combinations:

        device_masks = torch.randint(0, 5, (len(ms), 196))

        accuracy, precision, recall, fscore = get_metrics(filename=filename, mask_set=ms, device_masks=device_masks)
        metrics_json[filename][''.join(map(str, ms)) ] = {
            "Accuracy":accuracy.item(),
            "Trainable Parameters": (len(ms)*196+1)*10,
            "Precision":precision.item(),
            "Recall":recall.item(),
            "F-score":fscore.item()
            
        }




filename = 'data/metrics_v2.xlsx'

with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    for wavelength, results in metrics_json.items():
        # Flatten the data for each wavelength
        flattened_data = []
        for index, metrics in results.items():
            if metrics:  # Check if the metrics are not empty
                flattened_data.append({
                    'Index': index,
                    'Accuracy': metrics['Accuracy'],  # Convert tensor to float
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F-score': metrics['F-score']
                })
        
        # Create a DataFrame for the current wavelength
        if flattened_data:  # Check if there's data to save
            df = pd.DataFrame(flattened_data)
            # Write the DataFrame to a specific sheet named after the wavelength
            df.to_excel(writer, sheet_name=wavelength, index=False)

print("Data saved to", filename)

