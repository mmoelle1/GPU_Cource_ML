# Torch and TorchVision
import torch
import torchvision
import time

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparmeters
batch_size = 128

# Import the Fashion MNIST dataset: 70,000 grayscale images of clothes
# (28 by 28 pixels low-resolution images) in 10 categories
fashion_mnist = torchvision.datasets.FashionMNIST(root='.',
                                                  train=True,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)

# Separate the dataset into 50,000 images for training and 10,000 for testing
ds_train, ds_val = torch.utils.data.random_split(fashion_mnist, [50000, 10000])
ds_test = torchvision.datasets.FashionMNIST(root='.',
                                            train=False,
                                            transform=torchvision.transforms.ToTensor())

# Print the number of training/testing images
print(f"Using {len(ds_train.indices)} datasets for training")
print(f"Using {len(ds_test.data)} datasets for testing")

# Specify the categories
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Dataloaders
train_loader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, pin_memory=True, num_workers = 8, prefetch_factor=8)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size*2, pin_memory=True, num_workers = 8, prefetch_factor=8)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size*2, pin_memory=True, num_workers = 8, prefetch_factor=8)

class FMnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28*28, 10)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)                  # Generate predictions
        loss = torch.nn.functional.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)                    # Generate predictions
        loss = torch.nn.functional.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = FMnistModel().to(device)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

epochs = 10
lr = 0.1
tic = time.perf_counter()
history = fit(epochs, lr, model, train_loader, val_loader)
toc = time.perf_counter()
print(f"Spent {(toc-tic)} seconds on training using {torch.cuda.max_memory_allocated()} bytes of memory")
