[33mcommit 1fb46b3d21c24c8a8fd9d7c34ea3833174e1882d[m[33m ([m[1;36mHEAD -> [m[1;32mmaster[m[33m)[m
Author: gitders222 <anders.ullsfoss.torp@gmail.com>
Date:   Wed Feb 20 11:44:17 2019 +0100

    Adding files and alittle test

[1mdiff --git a/dataloaders.py b/dataloaders.py[m
[1mnew file mode 100644[m
[1mindex 0000000..48f9493[m
[1m--- /dev/null[m
[1m+++ b/dataloaders.py[m
[36m@@ -0,0 +1,53 @@[m
[32m+[m[32mfrom torchvision import transforms, datasets[m
[32m+[m[32mfrom torch.utils.data.sampler import SubsetRandomSampler[m
[32m+[m[32mimport torch[m
[32m+[m[32mimport numpy as np[m
[32m+[m
[32m+[m[32mmean = (0.4914, 0.4822, 0.4465)[m
[32m+[m[32mstd = (0.2023, 0.1994, 0.2010)[m
[32m+[m
[32m+[m
[32m+[m[32mdef load_cifar10(batch_size, validation_fraction=0.1):[m
[32m+[m[32m    transform = [[m
[32m+[m[32m        transforms.ToTensor(),[m
[32m+[m[32m        transforms.Normalize(mean, std)[m
[32m+[m[32m    ][m
[32m+[m[32m    transform = transforms.Compose(transform)[m
[32m+[m[32m    data_train = datasets.CIFAR10('data/cifar10',[m
[32m+[m[32m                                  train=True,[m
[32m+[m[32m                                  download=True,[m
[32m+[m[32m                                  transform=transform)[m
[32m+[m
[32m+[m[32m    data_test = datasets.CIFAR10('data/cifar10',[m
[32m+[m[32m                                 train=False,[m
[32m+[m[32m                                 download=True,[m
[32m+[m[32m                                 transform=transform)[m
[32m+[m
[32m+[m[32m    indices = list(range(len(data_train)))[m
[32m+[m[32m    split_idx = int(np.floor(validation_fraction * len(data_train)))[m
[32m+[m[32m    #  Uncomment to yield the same shuffle of the dataset each time[m
[32m+[m[32m    # Note that the order of the samples will still be random, since the sampler[m
[32m+[m[32m    # returns random indices from the list[m
[32m+[m[32m    # np.random.seed(42)[m
[32m+[m[32m    val_indices = np.random.choice(indices, size=split_idx, replace=False)[m
[32m+[m[32m    train_indices = list(set(indices) - set(val_indices))[m
[32m+[m
[32m+[m[32m    train_sampler = SubsetRandomSampler(train_indices)[m
[32m+[m[32m    validation_sampler = SubsetRandomSampler(val_indices)[m
[32m+[m
[32m+[m[32m    dataloader_train = torch.utils.data.DataLoader(data_train,[m
[32m+[m[32m                                                   sampler=train_sampler,[m
[32m+[m[32m                                                   batch_size=batch_size,[m
[32m+[m[32m                                                   num_workers=2)[m
[32m+[m
[32m+[m[32m    dataloader_val = torch.utils.data.DataLoader(data_train,[m
[32m+[m[32m                                                 sampler=validation_sampler,[m
[32m+[m[32m                                                 batch_size=batch_size,[m
[32m+[m[32m                                                 num_workers=2)[m
[32m+[m
[32m+[m[32m    dataloader_test = torch.utils.data.DataLoader(data_test,[m
[32m+[m[32m                                                  batch_size=batch_size,[m
[32m+[m[32m                                                  shuffle=False,[m
[32m+[m[32m                                                  num_workers=2)[m
[32m+[m
[32m+[m[32m    return dataloader_train, dataloader_val, dataloader_test[m
[1mdiff --git a/starter_code(5).zip b/starter_code(5).zip[m
[1mnew file mode 100644[m
[1mindex 0000000..701f5cf[m
Binary files /dev/null and b/starter_code(5).zip differ
[1mdiff --git a/starter_code.py b/starter_code.py[m
[1mnew file mode 100644[m
[1mindex 0000000..94c4885[m
[1m--- /dev/null[m
[1m+++ b/starter_code.py[m
[36m@@ -0,0 +1,210 @@[m
[32m+[m[32mimport os[m
[32m+[m[32mimport matplotlib.pyplot as plt[m
[32m+[m[32mimport torch[m
[32m+[m[32mfrom torch import nn[m
[32m+[m[32mfrom dataloaders import load_cifar10[m
[32m+[m[32mfrom utils import to_cuda, compute_loss_and_accuracy[m
[32m+[m
[32m+[m
[32m+[m[32mclass ExampleModel(nn.Module):[m
[32m+[m
[32m+[m[32m    def __init__(self,[m
[32m+[m[32m                 image_channels,[m
[32m+[m[32m                 num_classes):[m
[32m+[m[32m        """[m
[32m+[m[32m            Is called when model is initialized.[m
[32m+[m[32m            Args:[m
[32m+[m[32m                image_channels. Number of color channels in image (3)[m
[32m+[m[32m                num_classes: Number of classes we want to predict (10)[m
[32m+[m[32m        """[m
[32m+[m[32m        super().__init__()[m
[32m+[m[32m        num_filters = 32  # Set number of filters in first conv layer[m
[32m+[m
[32m+[m[32m        # Define the convolutional layers[m
[32m+[m[32m        self.feature_extractor = nn.Sequential([m
[32m+[m[32m            nn.Conv2d([m
[32m+[m[32m                in_channels=image_channels,[m
[32m+[m[32m                out_channels=num_filters,[m
[32m+[m[32m                kernel_size=5,[m
[32m+[m[32m                stride=1,[m
[32m+[m[32m                padding=2[m
[32m+[m[32m            ),[m
[32m+[m[32m            nn.MaxPool2d(kernel_size=2, stride=2),[m
[32m+[m[32m            nn.ReLU()[m
[32m+[m[32m        )[m
[32m+[m[32m        # The output of feature_extractor will be [batch_size, num_filters, 16, 16][m
[32m+[m[32m        self.num_output_features = 32*16*16[m
[32m+[m[32m        # Initialize our last fully connected layer[m
[32m+[m[32m        # Inputs all extracted features from the convolutional layers[m
[32m+[m[32m        # Outputs num_classes predictions, 1 for each class.[m
[32m+[m[32m        # There is no need for softmax activation function, as this is[m
[32m+[m[32m        # included with nn.CrossEntropyLoss[m
[32m+[m[32m        self.classifier = nn.Sequential([m
[32m+[m[32m            nn.Linear(self.num_output_features, num_classes),[m
[32m+[m[32m        )[m
[32m+[m
[32m+[m[32m    def forward(self, x):[m
[32m+[m[32m        """[m
[32m+[m[32m        Performs a forward pass through the model[m
[32m+[m[32m        Args:[m
[32m+[m[32m            x: Input image, shape: [batch_size, 3, 32, 32][m
[32m+[m[32m        """[m
[32m+[m
[32m+[m[32m        # Run image through convolutional layers[m
[32m+[m[32m        x = self.feature_extractor(x)[m
[32m+[m[32m        # Reshape our input to (batch_size, num_output_features)[m
[32m+[m[32m        x = x.view(-1, self.num_output_features)[m
[32m+[m[32m        # Forward pass through the fully-connected layers.[m
[32m+[m[32m        x = self.classifier(x)[m
[32m+[m[32m        return x[m
[32m+[m
[32m+[m
[32m+[m[32mclass Trainer:[m
[32m+[m
[32m+[m[32m    def __init__(self):[m
[32m+[m[32m        """[m
[32m+[m[32m        Initialize our trainer class.[m
[32m+[m[32m        Set hyperparameters, architecture, tracking variables etc.[m
[32m+[m[32m        """[m
[32m+[m[32m        # Define hyperparameters[m
[32m+[m[32m        self.epochs = 100[m
[32m+[m[32m        self.batch_size = 64[m
[32m+[m[32m        self.learning_rate = 5e-2[m
[32m+[m[32m        self.early_stop_count = 4[m
[32m+[m
[32m+[m[32m        # Architecture[m
[32m+[m
[32m+[m[32m        # Since we are doing multi-class classification, we use the CrossEntropyLoss[m
[32m+[m[32m        self.loss_criterion = nn.CrossEntropyLoss()[m
[32m+[m[32m        # Initialize the mode[m
[32m+[m[32m        self.model = ExampleModel(image_channels=3, num_classes=10)[m
[32m+[m[32m        # Transfer model to GPU VRAM, if possible.[m
[32m+[m[32m        self.model = to_cuda(self.model)[m
[32m+[m
[32m+[m[32m        # Define our optimizer. SGD = Stochastich Gradient Descent[m
[32m+[m[32m        self.optimizer = torch.optim.SGD(self.model.parameters(),[m
[32m+[m[32m                                         self.learning_rate)[m
[32m+[m
[32m+[m[32m        # Load our dataset[m
[32m+[m[32m        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)[m
[32m+[m
[32m+[m[32m        self.validation_check = len(self.dataloader_train) // 2[m
[32m+[m
[32m+[m[32m        # Tracking variables[m
[32m+[m[32m        self.VALIDATION_LOSS = [][m
[32m+[m[32m        self.TEST_LOSS = [][m
[32m+[m[32m        self.TRAIN_LOSS = [][m
[32m+[m[32m        self.TRAIN_ACC = [][m
[32m+[m[32m        self.VALIDATION_ACC = [][m
[32m+[m[32m        self.TEST_ACC = [][m
[32m+[m
[32m+[m[32m    def validation_epoch(self):[m
[32m+[m[32m        """[m
[32m+[m[32m            Computes the loss/accuracy for all three datasets.[m
[32m+[m[32m            Train, validation and test.[m
[32m+[m[32m        """[m
[32m+[m[32m        self.model.eval()[m
[32m+[m
[32m+[m[32m        # Compute for training set[m
[32m+[m[32m        train_loss, train_acc = compute_loss_and_accuracy([m
[32m+[m[32m            self.dataloader_train, self.model, self.loss_criterion[m
[32m+[m[32m        )[m
[32m+[m[32m        self.TRAIN_ACC.append(train_acc)[m
[32m+[m[32m        self.TRAIN_LOSS.append(train_loss)[m
[32m+[m
[32m+[m[32m        # Compute for validation set[m
[32m+[m[32m        validation_loss, validation_acc = compute_loss_and_accuracy([m
[32m+[m[32m            self.dataloader_val, self.model, self.loss_criterion[m
[32m+[m[32m        )[m
[32m+[m[32m        self.VALIDATION_ACC.append(validation_acc)[m
[32m+[m[32m        self.VALIDATION_LOSS.append(validation_loss)[m
[32m+[m[32m        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)[m
[32m+[m[32m        # Compute for testing set[m
[32m+[m[32m        test_loss, test_acc = compute_loss_and_accuracy([m
[32m+[m[32m            self.dataloader_test, self.model, self.loss_criterion[m
[32m+[m[32m        )[m
[32m+[m[32m        self.TEST_ACC.append(test_acc)[m
[32m+[m[32m        self.TEST_LOSS.append(test_loss)[m
[32m+[m
[32m+[m[32m        self.model.train()[m
[32m+[m
[32m+[m[32m    def should_early_stop(self):[m
[32m+[m[32m        """[m
[32m+[m[32m        Checks if validation loss doesn't improve over early_stop_count epochs.[m
[32m+[m[32m        """[m
[32m+[m[32m        # Check if we have more than early_stop_count elements in our validation_loss list.[m
[32m+[m[32m        if len(self.VALIDATION_LOSS) < self.early_stop_count:[m
[32m+[m[32m            return False[m
[32m+[m[32m        # We only care about the last [early_stop_count] losses.[m
[32m+[m[32m        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:][m
[32m+[m[32m        previous_loss = relevant_loss[0][m
[32m+[m[32m        for current_loss in relevant_loss[1:]:[m
[32m+[m[32m            # If the next loss decrease, early stopping criteria is not met.[m
[32m+[m[32m            if current_loss < previous_loss:[m
[32m+[m[32m                return False[m
[32m+[m[32m            previous_loss = current_loss[m
[32m+[m[32m        return True[m
[32m+[m
[32m+[m[32m    def train(self):[m
[32m+[m[32m        """[m
[32m+[m[32m        Trains the model for [self.epochs] epochs.[m
[32m+[m[32m        """[m
[32m+[m[32m        # Track initial loss/accuracy[m
[32m+[m[32m        self.validation_epoch()[m
[32m+[m[32m        for epoch in range(self.epochs):[m
[32m+[m[32m            # Perform a full pass through all the training samples[m
[32m+[m[32m            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):[m
[32m+[m[32m                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32][m
[32m+[m[32m                # Y_batch is the CIFAR10 image label. Shape: [batch_size][m
[32m+[m[32m                # Transfer images / labels to GPU VRAM, if possible[m
[32m+[m[32m                X_batch = to_cuda(X_batch)[m
[32m+[m[32m                Y_batch = to_cuda(Y_batch)[m
[32m+[m
[32m+[m[32m                # Perform the forward pass[m
[32m+[m[32m                predictions = self.model(X_batch)[m
[32m+[m[32m                # Compute the cross entropy loss for the batch[m
[32m+[m[32m                loss = self.loss_criterion(predictions, Y_batch)[m
[32m+[m
[32m+[m[32m                # Backpropagation[m
[32m+[m[32m                loss.backward()[m
[32m+[m
[32m+[m[32m                # Gradient descent step[m
[32m+[m[32m                self.optimizer.step()[m
[32m+[m[41m                [m
[32m+[m[32m                # Reset all computed gradients to 0[m
[32m+[m[32m                self.optimizer.zero_grad()[m
[32m+[m[32m                 # Compute loss/accuracy for all three datasets.[m
[32m+[m[32m                if batch_it % self.validation_check == 0:[m
[32m+[m[32m                    self.validation_epoch()[m
[32m+[m[32m                    # Check early stopping criteria.[m
[32m+[m[32m                    if self.should_early_stop():[m
[32m+[m[32m                        print("Early stopping.")[m
[32m+[m[32m                        return[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    trainer = Trainer()[m
[32m+[m[32m    trainer.train()[m
[32m+[m
[32m+[m[32m    os.makedirs("plots", exist_ok=True)[m
[32m+[m[32m    # Save plots and show them[m
[32m+[m[32m    plt.figure(figsize=(12, 8))[m
[32m+[m[32m    plt.title("Cross Entropy Loss")[m
[32m+[m[32m    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")[m
[32m+[m[32m    plt.plot(trainer.TRAIN_LOSS, label="Training loss")[m
[32m+[m[32m    plt.plot(trainer.TEST_LOSS, label="Testing Loss")[m
[32m+[m[32m    plt.legend()[m
[32m+[m[32m    plt.savefig(os.path.join("plots", "final_loss.png"))[m
[32m+[m[32m    plt.show()[m
[32m+[m
[32m+[m[32m    plt.figure(figsize=(12, 8))[m
[32m+[m[32m    plt.title("Accuracy")[m
[32m+[m[32m    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")[m
[32m+[m[32m    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")[m
[32m+[m[32m    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")[m
[32m+[m[32m    plt.legend()[m
[32m+[m[32m    plt.savefig(os.path.join("plots", "final_accuracy.png"))[m
[32m+[m[32m    plt.show()[m
[32m+[m
[32m+[m[32m    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])[m
[32m+[m[32m    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])[m
[1mdiff --git a/utils.py b/utils.py[m
[1mnew file mode 100644[m
[1mindex 0000000..090e103[m
[1m--- /dev/null[m
[1m+++ b/utils.py[m
[36m@@ -0,0 +1,55 @@[m
[32m+[m[32mimport torch[m
[32m+[m
[32m+[m[32mdef to_cuda(elements):[m
[32m+[m[32m    """[m
[32m+[m[32m    Transfers elements to GPU memory, if a nvidia- GPU is available.[m
[32m+[m[32m    Args:[m
[32m+[m[32m        elements: A list or a single pytorch module.[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        The same list transferred to GPU memory[m
[32m+[m[32m    """[m
[32m+[m
[32m+[m[32m    if torch.cuda.is_available(): # Checks if a GPU is available for pytorch[m
[32m+[m[32m        if isinstance(elements, (list, tuple)):[m
[32m+[m[32m            return [x.cuda() for x in elements] # Transfer each index of the list to GPU memory[m
[32m+[m[32m        return elements.cuda()[m
[32m+[m[32m    return elements[m
[32m+[m
[32m+[m
[32m+[m[32mdef compute_loss_and_accuracy(dataloader, model, loss_criterion):[m
[32m+[m[32m    """[m
[32m+[m[32m    Computes the total loss and accuracy over the whole dataloader[m
[32m+[m[32m    Args:[m
[32m+[m[32m        dataloder: Validation/Test dataloader[m
[32m+[m[32m        model: torch.nn.Module[m
[32m+[m[32m        loss_criterion: The loss criterion, e.g: nn.CrossEntropyLoss()[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        [loss_avg, accuracy]: both scalar.[m
[32m+[m[32m    """[m
[32m+[m[32m    # Tracking variables[m
[32m+[m[32m    loss_avg = 0[m
[32m+[m[32m    total_correct = 0[m
[32m+[m[32m    total_images = 0[m
[32m+[m[32m    total_steps = 0[m
[32m+[m
[32m+[m[32m    for (X_batch, Y_batch) in dataloader:[m
[32m+[m[32m        # Transfer images/labels to GPU VRAM, if possible[m
[32m+[m[32m        X_batch = to_cuda(X_batch)[m
[32m+[m[32m        Y_batch = to_cuda(Y_batch)[m
[32m+[m[32m        # Forward pass the images through our model[m
[32m+[m[32m        output_probs = model(X_batch)[m
[32m+[m[32m        # Compute loss[m
[32m+[m[32m        loss = loss_criterion(output_probs, Y_batch)[m
[32m+[m
[32m+[m[32m        # Predicted class is the max index over the column dimension[m
[32m+[m[32m        predictions = output_probs.argmax(dim=1).squeeze()[m
[32m+[m[32m        Y_batch = Y_batch.squeeze()[m
[32m+[m
[32m+[m[32m        # Update tracking variables[m
[32m+[m[32m        loss_avg += loss.item()[m
[32m+[m[32m        total_steps += 1[m
[32m+[m[32m        total_correct += (predictions == Y_batch).sum().item()[m
[32m+[m[32m        total_images += predictions.shape[0][m
[32m+[m[32m    loss_avg = loss_avg / total_steps[m
[32m+[m[32m    accuracy = total_correct / total_images[m
[32m+[m[32m    return loss_avg, accuracy[m
