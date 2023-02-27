import torch
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import numpy as np

# load the pre-trained Pix2Pix model
model = torch.load('pix2pix_model.pt')

# set the model to evaluation mode
model.eval()

# load the test dataset
test_dataset = MyDataset('test_data_path', transform=transforms.Compose([
    Resize((256, 256)),
    ToTensor()
]))

# create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# define the classification labels
classes = ['roads', 'water', 'land', 'buildings', 'forest']

# initialize the confusion matrix
confusion_matrix = np.zeros((len(classes), len(classes)))

# iterate over the test dataset
for i, (input_image, target_map) in enumerate(test_loader):
    # pass the input image through the Pix2Pix model to generate the output map
    output_map = model(input_image)

    # convert the output map to a numpy array
    output_map = output_map.detach().numpy()[0]

    # compute the predicted class labels for each pixel in the output map
    predicted_labels = np.argmax(output_map, axis=0)

    # compute the true class labels for each pixel in the target map
    true_labels = np.argmax(target_map.numpy()[0], axis=0)

    # update the confusion matrix
    for j in range(len(classes)):
        for k in range(len(classes)):
            if np.logical_and(predicted_labels == j, true_labels == k).sum() > 0:
                confusion_matrix[j, k] += 1

# compute the overall accuracy and class-wise accuracy
overall_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
class_accuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

# print the confusion matrix and accuracy metrics
print('Confusion Matrix:')
print(confusion_matrix)
print('Overall Accuracy:', overall_accuracy)
print('Class-wise Accuracy:', class_accuracy)
