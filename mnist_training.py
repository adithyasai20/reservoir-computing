import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import itertools

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, ConfusionMatrix


# Defines the device on which the processing is going to take place.
device = torch.device("cpu")
# Preprocessing takes most of the time. 
# No significant reduction in training time if we use GPU as the model is not very big.


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


# Dictionary of all the rows of each mask-set in every file 
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

class CustomMNISTDataset(Dataset):
    """Dataset object to save the preprocessed mnist dataset

    Parameters -
    mnist_data : `torch.utils.data.dataset`
                Contains images and corresponding labels of MNIST dataset               
    tables : `list[pd.DataFrame]`
                conductance tables for every mask-set.
    device_indices : `torch.tensor`
                Initial conductance states of every device. Shape of `((len(mask_sets), 196))`
    mask_sets : `List[int]`
                List of all the mask-sets to be used

    Attributes - 

    mask_sets : `List[int]`
                A list of all the mask-sets used for simulation.
    processed_data : `torch.tensor`
                MNIST images after preprocessing. Shape of ((N_samples, 1, 196 * len(mask_sets)))
    labels : `torch.tensor`
                label of each corresponding image. Shape of ((N_samples, num_classes))   (num_classes = 10)
    device_indices : List[List]
                Initial states of each device for every mask-set
    """
    def __init__(self, mnist_data, tables, device_indices, mask_sets ):
        self.mask_sets = mask_sets
        self.processed_data = []
        self.labels = []

        self.device_indices = device_indices.int().tolist()
        
        # Preprocessing step. Same as discussed in the paper given.
        for idx in tqdm(range(len(mnist_data))):
            image, label = mnist_data[idx]
            image = image.reshape(1, 196, 4)
            image_combined = (
                image[:, :, 0] * 1000 + 
                image[:, :, 1] * 100 + 
                image[:, :, 2] * 10 + 
                image[:, :, 3]
            )
            # x = []
            # for m_idx, mask in enumerate(self.mask_sets):
            #     for i, device in enumerate(self.device_indices[m_idx]):                
            #         x.append(tables[mask].loc[int(device), int(image_combined[0][i])]*1e9)
            x = [
                tables[mask].loc[int(device), int(image_combined[0][i])] * 1e9
                for mask_idx, mask in enumerate(self.mask_sets)
                for i, device in enumerate(self.device_indices[mask_idx])
            ]
            
            self.processed_data.append(x)
            self.labels.append(label)
        
        # Convert the lists to `torch.tensor`
        self.processed_data = torch.tensor(self.processed_data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        """ Function to access length of the dataset object.

        Returns: `int`
                    Length of the dataset object
        """
        return len(self.processed_data)

    def __getitem__(self, idx):
        """ Returns element's image and label at a given index.
        Parameters - 
        idx : int
            index of the element to be accessed

        Returns - torch.tensor, torch.tensor
        """
        return self.processed_data[idx], self.labels[idx]

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
    

# Model hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
learning_rate = 0.001

def get_metrics(filename, mask_sets, device_indices):
    # filename = "365nm"  # Excel document name from which data has to be extracted.
    path = "data/"+filename+".xlsx" # Excel doc path

    df = pd.read_excel(path, usecols='B:Q') # Read the excel sheet
    # Break the sheet down into different `DataFrame`s for every Mask-set in the table
    tables = [df.iloc[json_data[filename][key]].copy().reset_index(drop=True) for key in list(json_data[filename].keys())]

    train_dataset = CustomMNISTDataset(train_data, tables=tables, mask_sets=mask_sets, device_indices=device_indices)
    validation_dataset = CustomMNISTDataset(val_data, tables=tables, mask_sets=mask_sets, device_indices=device_indices)
    test_dataset = CustomMNISTDataset(mnist_data_test, tables=tables, mask_sets=mask_sets, device_indices=device_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    model = ReadoutLayer(len(mask_sets)*196)
    # Loss function, Categorical crossentropy loss for multi-class classification problem
    criterion = nn.CrossEntropyLoss()
    # Optimizer to update the gradients.
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    val_accuracy, val_precision, val_recall, val_fscore = [], [], [], []
    # Evaluation metrics
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
    precision = Precision(task="multiclass", num_classes=10, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=10, average='macro').to(device)
    f1_score = F1Score(task="multiclass", num_classes=10, average='macro').to(device)
    # Class-wise Confusion matrix
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=10).to(device)

    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
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

    save_path = "data/mnist_results/" + filename + '_' + ''.join(map(str, mask_sets)) + '.npz'
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
for filename in ["White","455nm", "365nm"]:
    for ms in combinations:

        dev_ind = torch.randint(0, 5, (len(ms), 196))

        accuracy, precision, recall, fscore = get_metrics(filename=filename, mask_sets=ms, device_indices=dev_ind)
        metrics_json[filename][''.join(map(str, ms)) ] = {
            "Accuracy":accuracy.item(),
            "Precision":precision.item(),
            "Recall":recall.item(),
            "F-score":fscore.item()
        }


# Specify the file name
filename = 'data/metrics.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine
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