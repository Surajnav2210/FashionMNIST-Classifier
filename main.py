import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
     TODO: implement this function.

     INPUT:
         An optional boolean argument (default value is True for training dataset)

     RETURNS:
         Dataloader for the training set (if training = True) or the test set (if training = False)
     """

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./data', train=True,
                                      download=True, transform=custom_transform)
    test_set = datasets.FashionMNIST('./data', train=False,
                                     transform=custom_transform)

    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    # criterion = nn.CrossEntropyLoss()
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        total_correct = 0
        total_loss = 0.0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_loss += loss.item()

        avg_Accuracy = total_correct / total * 100
        actual_loss = total_loss/1000
        print(f'Train Epoch: {epoch} Accuracy: {total_correct}/{total} ({avg_Accuracy:.2f}%) Loss: {actual_loss:.3f}')

def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    Evaluate the model on the test dataset.

    INPUT:
        model - the trained model produced by the previous function
        test_loader - the test DataLoader
        criterion - cross-entropy
        show_loss - whether to print the test loss (default is True)

    RETURNS:
        None
    """
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            total += labels.size(0)

    accuracy = total_correct / total * 100.0

    if show_loss == False:
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        average_loss = total_loss / len(test_loader)
        print(f"Average loss: {average_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")


def predict_label(model, test_images, index):
    """
    Predict the labels for a single test image using a data loader.

    INPUT:
        model - the trained model
        test_loader - the test DataLoader
        index - specific index i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    # """

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle Boot']

    model.eval()
    with torch.no_grad():
        test_image = test_images[index]
        logits = model(test_image.unsqueeze(0))
        probabilities = F.softmax(logits, dim=1)
        sorted_prob, sorted_indices = torch.sort(probabilities, descending=True, dim=1)

        top_labels = []
        for i in sorted_indices[0]:
            top_labels.append(class_names[i])

        top_probs = []
        for j in sorted_prob[0]:
            top_probs.append(j * 100)

        for i in range(3):
            print(f"{top_labels[i]}: {top_probs[i]:.2f}%")


if __name__ == '__main__':
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)

    test_loader = get_data_loader(training=False)
    print(type(test_loader))
    print(test_loader.dataset)

    model = build_model()
    print(model)

    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, T=5)
    evaluate_model(model, test_loader, criterion, show_loss=True)

    test_loader = get_data_loader(training=False)
    test_images, _ = next(iter(test_loader))
    predict_label(model, test_images, 1)