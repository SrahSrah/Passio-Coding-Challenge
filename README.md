
# Food Recognition System For Passio
## By: Sarah Hernandez

### Complied in the Google Cloud Platform


```python
import tensorflow as tf
import numpy as np
import pickle
import os
import cv2
import math
from random import shuffle
import matplotlib.pyplot as plt
import skimage
    

```

### Goal 1: Prepare a Dataset:

#### Step 1: Explore Dataset



```python
currentDir = os.getcwd()
foodDir = currentDir + "/Food"
notFoodDir = currentDir + "/Not Food"

foodList = os.listdir(foodDir)
numFoods = len(foodList)

notFoodList = os.listdir(notFoodDir)
numNotFoods = len(notFoodList)

print("Foods: " + str(numFoods))
print("Not Foods: " + str(numNotFoods))

for i in range(3):
    testDir = foodDir + "/" + foodList[i]
    img = cv2.imread(testDir)
    print("Example Food Shape: " + str(img.shape))
    testDir2 = notFoodDir + "/" + notFoodList[i]
    img2 = cv2.imread(testDir2)
    print("Example Not Food Shape: " + str(img2.shape))
    

```

    Foods: 144
    Not Foods: 125
    Example Food Shape: (768, 1024, 3)
    Example Not Food Shape: (4032, 3024, 3)
    Example Food Shape: (1024, 1024, 3)
    Example Not Food Shape: (640, 480, 3)
    Example Food Shape: (640, 640, 3)
    Example Not Food Shape: (360, 640, 3)


So we have fewer than 300 images of food and not food, of various square and rectangular sizes. This is a rather small data set, so we'll augment the data using a few tricks:


#### Step 2: Augment Data


```python
# Get array of directories of food and not food
def get_image_dirs(foodList, notFoodList):
    
    foodDirs = []
    for food in foodList:
        if not food.startswith('.'):
            foodDirs.append(foodDir + "/" + food)
    
    notFoodDirs = []
    for notFood in notFoodList:
        if not notFood.startswith('.'):
            notFoodDirs.append(notFoodDir + "/" + notFood)
        
    
    return foodDirs, notFoodDirs
        

foodDirs, notFoodDirs = get_image_dirs(foodList, notFoodList)

print("Directories Created")
```

    Directories Created



```python

# First trick: flip and rotate images
for i in range(numFoods):
    img = cv2.imread(foodDirs[i])
    img2 = np.fliplr(img)
    img3 = np.flipud(img)
    img4 = np.rot90(img)
    cv2.imwrite(foodDir + "/lr" + str(i) + ".jpg", img2)
    cv2.imwrite(foodDir + "/ud" + str(i) + ".jpg", img3)
    cv2.imwrite(foodDir + "/rot90" + str(i) + ".jpg", img4)
    
print("Food Flipped Images Created")

for i in range(numNotFoods):
    img = cv2.imread(notFoodDirs[i])
    img2 = np.fliplr(img)
    img3 = np.flipud(img)
    img4 = np.rot90(img)
    cv2.imwrite(notFoodDir + "/lr" + str(i) + ".jpg", img2)
    cv2.imwrite(notFoodDir + "/ud" + str(i) + ".jpg", img3)
    cv2.imwrite(notFoodDir + "/rot90" + str(i) + ".jpg", img4)
    
print("Not Food Flipped Images Created")

```

    Food Flipped Images Created
    Not Food Flipped Images Created



```python
# Update Food and NotFood Dirs:
foodList = os.listdir(foodDir)

notFoodList = os.listdir(notFoodDir)

foodDirs, notFoodDirs = get_image_dirs(foodList, notFoodList)

print("Directories created")
    
```

    Directories created



```python
# Second trick: add noise to images:
for i in range(len(foodDirs)):
    
    img = cv2.imread(foodDirs[i])
    img = cv2.resize(img,(256,256))
    row,col,ch= img.shape
    mean = 0
    gauss = np.random.normal(mean,30,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    cv2.imwrite(foodDir + "/noisy" + str(i) + ".jpg", noisy)


print("Noisy Food Created")

for i in range(len(notFoodDirs)):
    
    img = cv2.imread(notFoodDirs[i])
    img = cv2.resize(img,(256,256))
    row,col,ch= img.shape
    mean = math.ceil(255/2)
    gauss = np.random.normal(mean,50,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    cv2.imwrite(notFoodDir + "/noisy" + str(i) + ".jpg", noisy)
       
print("Noisy Not Food Created")

```

    Noisy Food Created
    Noisy Not Food Created



```python
foodList = os.listdir(foodDir)

notFoodList = os.listdir(notFoodDir)

foodDirs, notFoodDirs = get_image_dirs(foodList, notFoodList)

print("Number of Foods: " + str(len(foodDirs)))
print("Number of Not Foods: " + str(len(notFoodDirs)))

```

    Number of Foods: 1152
    Number of Not Foods: 1000


Now we've got something to work with! Next, let's generate the dataset:

#### Step 3: Generate Dataset


```python
import cv2
class Generate_Dataset:
    
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
        self.labels = []
        self.data_paths = []
        self.images = []
        
        # Now we put all data paths in a single matrix and shuffle it: 
        self.data_paths = np.concatenate([self.data_dirs[0], self.data_dirs[1]])
        shuffle(self.data_paths)
        
        #Next, generate labels:
        for path in self.data_paths:
            self.labels.append(self.generate_data_labels(path))
            self.images.append(self.get_image(path))
        
        
    # Returns label of specified file
    def generate_data_labels(self, directory):
        labels = []
        # Because we're doing a simple binary classification, we can one-hot-encode here:
        if "Not Food" in directory:
            label = [1, 0]
        else:
            label = [0, 1]
        
        return label
    
    
    def get_data_paths(self, startIndex, endIndex):
        return self.data_paths[startIndex:endIndex]
    
    def get_data_labels(self, startIndex, endIndex):
        return self.labels[startIndex: endIndex]
        
        
    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img,(256,256))
        return img
    
    def get_data(self):
        return self.images, self.labels
    
    def get_all_dirs(self):
        return self.data_paths
        
        
        
        
dirs = [foodDirs, notFoodDirs]
dataset = Generate_Dataset(dirs)
images, labels = dataset.get_data()
paths = dataset.get_all_dirs()

print("Dataset Generated")


```

    Dataset Generated


Okay, we've got an array of images, all scaled down to 256x256, and an array of labels that correspond to the images. But we still need to do some preprocessing:

### Step 4: Preprocess Data


```python
def normalize(x):
    #Returns a normalized image, x: input image data in numpy array [256, 256, 3]
    
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x
```


```python
def preprocess_helper(some_images, some_labels, some_paths, filename, isTraining = False):
    some_images = normalize(some_images)
    #some_images = some_images.reshape((len(some_images), 3, 256, 256))
    
    num_images = len(some_images)
    
    if not isTraining:
        pickle.dump((some_images, some_labels, some_paths), open(filename, "wb"))
    else:
        # break training images into five batches
        for i in range(5):
            newFileName = filename + str(i) + ".p"
            first_index = int(num_images*i/5)
            second_index = int(num_images*(i+1)/5)
            pickle.dump((some_images[first_index:second_index], some_labels[first_index:second_index], some_paths[first_index:second_index]), open(newFileName, "wb"))
        
        
    

def preprocess(currentDir):
    
    validation_images = []
    validation_labels = []
    validation_paths = []
    test_images = []
    test_labels = []
    test_paths = []
    training_images = []
    training_labels = []
    training_paths = []
    # Save 10% of data for validation, and another 10% for testing:
    first_index = int(len(images)*0.1)
    second_index = int(len(images)*0.2)
    
    
    # Save validation set:
    validation_images.extend(images[0:first_index])
    validation_labels.extend(labels[0:first_index])
    validation_paths.extend(paths[0:first_index])
    filename = currentDir + "/" + "preprocess_validation.p"
    preprocess_helper(np.array(validation_images), np.array(validation_labels), np.array(validation_paths), filename)
    print("Validation Set Saved")
    
    # Save testing set:
    test_images.extend(images[first_index:second_index])
    test_labels.extend(labels[first_index:second_index])
    test_paths.extend(paths[first_index:second_index])
    filename = currentDir + "/" + "preprocess_testing.p"
    preprocess_helper(np.array(test_images), np.array(test_labels), np.array(test_paths), filename)
    print("Testing Set Saved")
    
    # Save training set!
    training_images.extend(images[second_index:])
    training_labels.extend(labels[second_index:])
    training_paths.extend(paths[second_index:])
    filename = currentDir + "/" + "preprocess_training"
    preprocess_helper(np.array(training_images), np.array(training_labels), np.array(training_paths), filename, True)
    print("Training Set Saved")
    
    

```


```python
preprocess(currentDir)
test_images, test_labels, test_paths = pickle.load(open(currentDir + "/" + "preprocess_testing.p", mode = "rb"))
valid_images, valid_labels, valid_paths = pickle.load(open(currentDir + "/" + "preprocess_validation.p", mode = "rb"))

# Split training images and labels into five batches:
train_images = []
train_labels = []
train_paths = []
for i in range(5): 
    batch_images, batch_labels, batch_paths = pickle.load(open(currentDir + "/" + "preprocess_training" + str(i) + ".p", mode = "rb"))
    train_images.append(batch_images)
    train_labels.append(batch_labels)
    train_paths.append(batch_paths)
    
print("All sets created and loaded")
```

    Validation Set Saved
    Testing Set Saved
    Training Set Saved
    All sets created and loaded


### Goal 2: Implement a neural network for classifying food vs non-food

#### Step 1: Prepare Model 
We will prepare the model by creating several helper functions. The first of these will help us get mini-batches as needed for training. The remaining are methods that will ech define a layer of the model: a convolutional layer, a flattening layer, a fully connected layer, or the final output layer. 




```python
# Create methods to get mini-batches
def get_mini_batches(batch_size, batch_images, batch_labels, batch_paths):
    # Returns images and labels in batches
   
    for start in range(0, len(batch_images), batch_size):
        end = min(start + batch_size, len(batch_images))
        
        temp_img = list(batch_images[start:end])
        
        temp_labels = list(batch_labels[start:end])
        
        temp_paths = list(batch_paths[start:end])
        
        yield temp_img, temp_labels, temp_paths
        
    
```


```python

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :-param x_tensor: TensorFlow Tensor
    :-param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :-param conv_strides: Stride 2-D Tuple for convolution
    :-param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    
    import math
    
    h_in =  int(x_tensor.shape[1])
    w_in =  int(x_tensor.shape[2])
    h = math.ceil(float(h_in - conv_strides[0] + 1) / float(conv_strides[0]))
    w = math.ceil(float(w_in - conv_strides[1] + 1) / float(conv_strides[1]))
    
    mean = 0.0
    stddev = 0.01
    weights_init = tf.random_normal([*conv_ksize, int(x_tensor.shape[3]), conv_num_outputs], mean=0.0, stddev=0.01, dtype=tf.float32)
    weights = tf.Variable(weights_init)
    
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    c_strides = [1, conv_strides[0], conv_strides[1], 1]
    p_strides = [1, pool_strides[0], pool_strides[1], 1]
    p_ksize = [1, pool_ksize[0], pool_ksize[1], 1]
    padding = "SAME"
    
    
    conv = tf.nn.conv2d(tf.to_float(x_tensor), weights, c_strides, padding)
    conv = tf.nn.bias_add(conv, bias)
    conv = tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv, ksize = p_ksize , strides = p_strides, padding = padding)
    
    
    
    return conv 


```


```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    
    shape = x_tensor.get_shape().as_list()        
    dim = np.prod(shape[1:])            
    x2 = tf.reshape(x_tensor, [-1, dim])          
    
    return x2
```


```python
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """ 
    
    weights = tf.Variable(tf.truncated_normal([int(x_tensor.shape[-1]), num_outputs], mean=0.0, stddev=0.01))

    bias = tf.Variable(tf.zeros([num_outputs]))

    layer = tf.add(tf.matmul(x_tensor, weights), bias)

    layer = tf.nn.relu(layer)
    
    return layer
```


```python
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    mean = 0.0
    stddev = 0.01
    weight_init = tf.truncated_normal([int(x_tensor.shape[-1]), num_outputs], mean=mean, stddev= stddev)
    weights = tf.Variable(weight_init)

    
    bias = tf.Variable(tf.zeros([num_outputs]))


    output = tf.add(tf.matmul(x_tensor, weights), bias)
    
    return output 

```

#### Step 2: Build the Model
Now, we'll combine the helper function above to create a multi-layered CNN model.
The model is laregly based off of prior succesfull image classification models, as shown here:

<img src = "cnn_network.jpg">


Like in the image above, I will start off with a few convolutional layers (followed the addition of bias and the application of max pooling), followed by a flattening layer. Next will be several fully connected layers, increasing in size until the final layer, the output layer, converges into two logit outputs.


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    
    conv_num_outputs = 2
    conv_ksize = (3,3)
    conv_strides = (1,1)
    pool_ksize = (2,2)
    pool_strides = (2,2)
    
    conv1 = conv2d_maxpool(x, 256, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv2 = conv2d_maxpool(conv1, 512, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv3 = conv2d_maxpool(conv1, 1024, conv_ksize, conv_strides, pool_ksize, pool_strides)

    
    flat = flatten(conv3)
    
    
    fullycon1 = fully_conn(flat, 256)
    fullycon2 = fully_conn(fullycon1, 512)
    fullycon3 = fully_conn(fullycon2, 1024)
    
    dropout = tf.nn.dropout(fullycon3, tf.to_float(keep_prob))
    
    num_outputs = 2
    outputs = output(dropout, num_outputs)
    
    return outputs

```


```python
batch_size = 16
epochs = 125
keep_probability = .5
learning_rate = 0.001

```


```python
def train_neural_network(session, optimizer, keep_probability, image_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    
    transposed_images = np.array(image_batch).transpose(0, 3, 1, 2)
    
    session.run(optimizer, feed_dict = {"x:0":transposed_images, "y:0": np.array(label_batch), "keep_prob:0": keep_probability})
    
```


```python
def print_stats(session, image_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    
    transposed_images = np.array(image_batch).transpose(0, 3, 1, 2)
    
    loss = session.run(cost, feed_dict={x: transposed_images, y: np.array(label_batch), keep_prob: 1.0})
    valid_acc = session.run(accuracy, feed_dict={"x:0": transposed_images, "y:0": np.array(label_batch), "keep_prob:0": 1.0})
    
    
    print('Loss: {:>10.4f} Accuracy: {:.6f}'.format(loss,valid_acc))
```

### Step 3: Train the Model

Next, we'll train the model. We'll do so by using the Adam Optimizer for gradient descent, and by shuffling each batch as we train it to increase learning. Here, we'll generate and create the tensorflow graph and run the session in one swift motion.
    


```python
import random

# Removes prior weights, biases, etc.
tf.reset_default_graph()


with tf.Graph().as_default():
# Create placeholders:
    x = tf.placeholder(tf.float32, shape = (None, 3,256,256), name = "x")
    y = tf.placeholder(tf.float32, shape = (None, 2), name = "y")
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()


    logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name="logits")

    # Loss and Optimizer, using the sigmoid function for binary classification rather than softmax
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name= "accuracy")

    print("Training...")
    with tf.Session() as sess:
        # Initializing the variables
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        # Training cycle
        for epoch in range(epochs):
            print("Epoch Number: " + str(epoch))
            
            for batch in range(5):
                for batch_images, batch_labels, batch_paths in get_mini_batches(batch_size, train_images[batch], train_labels[batch], train_paths[batch]):
                    c = list(zip(batch_images, batch_labels))
                    random.shuffle(c)
                    batch_images, batch_labels = zip(*c)
                    train_neural_network(sess, optimizer, keep_probability, batch_images, batch_labels)

                print("Epoch " + str(epoch) + ", Batch " + str(batch) + ": ")   
                print_stats(sess, valid_images, valid_labels, cost, accuracy)
            #learning_rate = learning_rate/1.05
   
                
            
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./training_sess")
        
print("Training Completed")
```

    Training...
    Epoch Number: 0
    Epoch 0, Batch 0: 
    Loss:     0.6807 Accuracy: 0.460465
    Epoch 0, Batch 1: 
    Loss:     0.6791 Accuracy: 0.804651
    Epoch 0, Batch 2: 
    Loss:     0.5938 Accuracy: 0.618605
    Epoch 0, Batch 3: 
    Loss:     0.5853 Accuracy: 0.660465
    Epoch 0, Batch 4: 
    Loss:     0.4902 Accuracy: 0.837209
    Epoch Number: 1
    Epoch 1, Batch 0: 
    Loss:     0.5472 Accuracy: 0.688372
    Epoch 1, Batch 1: 
    Loss:     0.3875 Accuracy: 0.841860
    Epoch 1, Batch 2: 
    Loss:     0.3462 Accuracy: 0.865116
    Epoch 1, Batch 3: 
    Loss:     0.3294 Accuracy: 0.860465
    Epoch 1, Batch 4: 
    Loss:     0.4580 Accuracy: 0.776744
    Epoch Number: 2
    Epoch 2, Batch 0: 
    Loss:     0.3373 Accuracy: 0.837209
    Epoch 2, Batch 1: 
    Loss:     0.3872 Accuracy: 0.841860
    Epoch 2, Batch 2: 
    Loss:     0.3080 Accuracy: 0.855814
    Epoch 2, Batch 3: 
    Loss:     0.3254 Accuracy: 0.869767
    Epoch 2, Batch 4: 
    Loss:     0.4109 Accuracy: 0.823256
    Epoch Number: 3
    Epoch 3, Batch 0: 
    Loss:     0.3599 Accuracy: 0.837209
    Epoch 3, Batch 1: 
    Loss:     0.5133 Accuracy: 0.702326
    Epoch 3, Batch 2: 
    Loss:     0.4510 Accuracy: 0.786047
    Epoch 3, Batch 3: 
    Loss:     0.2729 Accuracy: 0.888372
    Epoch 3, Batch 4: 
    Loss:     0.3531 Accuracy: 0.832558
    Epoch Number: 4
    Epoch 4, Batch 0: 
    Loss:     0.3961 Accuracy: 0.800000
    Epoch 4, Batch 1: 
    Loss:     0.4901 Accuracy: 0.744186
    Epoch 4, Batch 2: 
    Loss:     0.3442 Accuracy: 0.860465
    Epoch 4, Batch 3: 
    Loss:     0.3390 Accuracy: 0.855814
    Epoch 4, Batch 4: 
    Loss:     0.2734 Accuracy: 0.865116
    Epoch Number: 5
    Epoch 5, Batch 0: 
    Loss:     0.3111 Accuracy: 0.846512
    Epoch 5, Batch 1: 
    Loss:     0.3092 Accuracy: 0.869767
    Epoch 5, Batch 2: 
    Loss:     0.6058 Accuracy: 0.548837
    Epoch 5, Batch 3: 
    Loss:     0.3391 Accuracy: 0.832558
    Epoch 5, Batch 4: 
    Loss:     0.2875 Accuracy: 0.883721
    Epoch Number: 6
    Epoch 6, Batch 0: 
    Loss:     0.2670 Accuracy: 0.855814
    Epoch 6, Batch 1: 
    Loss:     0.2822 Accuracy: 0.883721
    Epoch 6, Batch 2: 
    Loss:     0.2888 Accuracy: 0.869767
    Epoch 6, Batch 3: 
    Loss:     0.2613 Accuracy: 0.883721
    Epoch 6, Batch 4: 
    Loss:     0.2784 Accuracy: 0.865116
    Epoch Number: 7
    Epoch 7, Batch 0: 
    Loss:     0.2631 Accuracy: 0.883721
    Epoch 7, Batch 1: 
    Loss:     0.2882 Accuracy: 0.888372
    Epoch 7, Batch 2: 
    Loss:     0.3000 Accuracy: 0.888372
    Epoch 7, Batch 3: 
    Loss:     0.2263 Accuracy: 0.883721
    Epoch 7, Batch 4: 
    Loss:     0.2525 Accuracy: 0.897674
    Epoch Number: 8
    Epoch 8, Batch 0: 
    Loss:     0.3138 Accuracy: 0.851163
    Epoch 8, Batch 1: 
    Loss:     0.2820 Accuracy: 0.855814
    Epoch 8, Batch 2: 
    Loss:     2.3751 Accuracy: 0.604651
    Epoch 8, Batch 3: 
    Loss:     0.4320 Accuracy: 0.818605
    Epoch 8, Batch 4: 
    Loss:     0.4245 Accuracy: 0.813953
    Epoch Number: 9
    Epoch 9, Batch 0: 
    Loss:     0.3067 Accuracy: 0.851163
    Epoch 9, Batch 1: 
    Loss:     0.3360 Accuracy: 0.846512
    Epoch 9, Batch 2: 
    Loss:     0.3654 Accuracy: 0.832558
    Epoch 9, Batch 3: 
    Loss:     0.3555 Accuracy: 0.841860
    Epoch 9, Batch 4: 
    Loss:     0.2701 Accuracy: 0.874419
    Epoch Number: 10
    Epoch 10, Batch 0: 
    Loss:     0.3090 Accuracy: 0.874419
    Epoch 10, Batch 1: 
    Loss:     0.2658 Accuracy: 0.879070
    Epoch 10, Batch 2: 
    Loss:     0.3061 Accuracy: 0.865116
    Epoch 10, Batch 3: 
    Loss:     0.2918 Accuracy: 0.855814
    Epoch 10, Batch 4: 
    Loss:     0.2521 Accuracy: 0.902326
    Epoch Number: 11
    Epoch 11, Batch 0: 
    Loss:     0.2526 Accuracy: 0.888372
    Epoch 11, Batch 1: 
    Loss:     0.2378 Accuracy: 0.869767
    Epoch 11, Batch 2: 
    Loss:     0.2488 Accuracy: 0.883721
    Epoch 11, Batch 3: 
    Loss:     0.2381 Accuracy: 0.906977
    Epoch 11, Batch 4: 
    Loss:     0.2855 Accuracy: 0.893023
    Epoch Number: 12
    Epoch 12, Batch 0: 
    Loss:     0.2600 Accuracy: 0.869767
    Epoch 12, Batch 1: 
    Loss:     0.2685 Accuracy: 0.883721
    Epoch 12, Batch 2: 
    Loss:     0.4552 Accuracy: 0.883721
    Epoch 12, Batch 3: 
    Loss:     0.3571 Accuracy: 0.865116
    Epoch 12, Batch 4: 
    Loss:     0.2606 Accuracy: 0.906977
    Epoch Number: 13
    Epoch 13, Batch 0: 
    Loss:     0.2215 Accuracy: 0.902326
    Epoch 13, Batch 1: 
    Loss:     0.2849 Accuracy: 0.888372
    Epoch 13, Batch 2: 
    Loss:     0.2477 Accuracy: 0.888372
    Epoch 13, Batch 3: 
    Loss:     0.2979 Accuracy: 0.883721
    Epoch 13, Batch 4: 
    Loss:     0.2587 Accuracy: 0.920930
    Epoch Number: 14
    Epoch 14, Batch 0: 
    Loss:     0.3781 Accuracy: 0.851163
    Epoch 14, Batch 1: 
    Loss:     0.2662 Accuracy: 0.888372
    Epoch 14, Batch 2: 
    Loss:     0.2153 Accuracy: 0.911628
    Epoch 14, Batch 3: 
    Loss:     0.2298 Accuracy: 0.906977
    Epoch 14, Batch 4: 
    Loss:     0.2386 Accuracy: 0.916279
    Epoch Number: 15
    Epoch 15, Batch 0: 
    Loss:     0.2622 Accuracy: 0.893023
    Epoch 15, Batch 1: 
    Loss:     0.2321 Accuracy: 0.906977
    Epoch 15, Batch 2: 
    Loss:     0.2549 Accuracy: 0.906977
    Epoch 15, Batch 3: 
    Loss:     0.2087 Accuracy: 0.911628
    Epoch 15, Batch 4: 
    Loss:     0.3612 Accuracy: 0.865116
    Epoch Number: 16
    Epoch 16, Batch 0: 
    Loss:     0.2563 Accuracy: 0.874419
    Epoch 16, Batch 1: 
    Loss:     0.3244 Accuracy: 0.869767
    Epoch 16, Batch 2: 
    Loss:     0.2397 Accuracy: 0.893023
    Epoch 16, Batch 3: 
    Loss:     0.2235 Accuracy: 0.925581
    Epoch 16, Batch 4: 
    Loss:     0.2310 Accuracy: 0.902326
    Epoch Number: 17
    Epoch 17, Batch 0: 
    Loss:     0.2448 Accuracy: 0.911628
    Epoch 17, Batch 1: 
    Loss:     0.3025 Accuracy: 0.888372
    Epoch 17, Batch 2: 
    Loss:     0.2331 Accuracy: 0.902326
    Epoch 17, Batch 3: 
    Loss:     0.2708 Accuracy: 0.893023
    Epoch 17, Batch 4: 
    Loss:     0.2397 Accuracy: 0.893023
    Epoch Number: 18
    Epoch 18, Batch 0: 
    Loss:     0.2707 Accuracy: 0.897674
    Epoch 18, Batch 1: 
    Loss:     0.2685 Accuracy: 0.893023
    Epoch 18, Batch 2: 
    Loss:     0.2209 Accuracy: 0.916279
    Epoch 18, Batch 3: 
    Loss:     0.2441 Accuracy: 0.934884
    Epoch 18, Batch 4: 
    Loss:     0.2657 Accuracy: 0.869767
    Epoch Number: 19
    Epoch 19, Batch 0: 
    Loss:     0.2566 Accuracy: 0.897674
    Epoch 19, Batch 1: 
    Loss:     0.3040 Accuracy: 0.897674
    Epoch 19, Batch 2: 
    Loss:     0.2454 Accuracy: 0.916279
    Epoch 19, Batch 3: 
    Loss:     0.2251 Accuracy: 0.916279
    Epoch 19, Batch 4: 
    Loss:     0.2413 Accuracy: 0.911628
    Epoch Number: 20
    Epoch 20, Batch 0: 
    Loss:     0.3752 Accuracy: 0.865116
    Epoch 20, Batch 1: 
    Loss:     0.4064 Accuracy: 0.906977
    Epoch 20, Batch 2: 
    Loss:     0.3179 Accuracy: 0.902326
    Epoch 20, Batch 3: 
    Loss:     0.3142 Accuracy: 0.860465
    Epoch 20, Batch 4: 
    Loss:     0.2450 Accuracy: 0.906977
    Epoch Number: 21
    Epoch 21, Batch 0: 
    Loss:     0.3358 Accuracy: 0.874419
    Epoch 21, Batch 1: 
    Loss:     0.2201 Accuracy: 0.906977
    Epoch 21, Batch 2: 
    Loss:     0.2332 Accuracy: 0.930233
    Epoch 21, Batch 3: 
    Loss:     0.2312 Accuracy: 0.888372
    Epoch 21, Batch 4: 
    Loss:     0.3021 Accuracy: 0.897674
    Epoch Number: 22
    Epoch 22, Batch 0: 
    Loss:     0.2453 Accuracy: 0.897674
    Epoch 22, Batch 1: 
    Loss:     0.1922 Accuracy: 0.934884
    Epoch 22, Batch 2: 
    Loss:     0.1954 Accuracy: 0.934884
    Epoch 22, Batch 3: 
    Loss:     0.2187 Accuracy: 0.944186
    Epoch 22, Batch 4: 
    Loss:     0.3704 Accuracy: 0.874419
    Epoch Number: 23
    Epoch 23, Batch 0: 
    Loss:     0.7545 Accuracy: 0.883721
    Epoch 23, Batch 1: 
    Loss:     0.3112 Accuracy: 0.851163
    Epoch 23, Batch 2: 
    Loss:     0.2281 Accuracy: 0.925581
    Epoch 23, Batch 3: 
    Loss:     0.3374 Accuracy: 0.934884
    Epoch 23, Batch 4: 
    Loss:     0.2202 Accuracy: 0.888372
    Epoch Number: 24
    Epoch 24, Batch 0: 
    Loss:     0.2792 Accuracy: 0.911628
    Epoch 24, Batch 1: 
    Loss:     0.3020 Accuracy: 0.897674
    Epoch 24, Batch 2: 
    Loss:     0.2552 Accuracy: 0.944186
    Epoch 24, Batch 3: 
    Loss:     0.2982 Accuracy: 0.911628
    Epoch 24, Batch 4: 
    Loss:     0.3849 Accuracy: 0.888372
    Epoch Number: 25
    Epoch 25, Batch 0: 
    Loss:     0.3182 Accuracy: 0.902326
    Epoch 25, Batch 1: 
    Loss:     0.4059 Accuracy: 0.902326
    Epoch 25, Batch 2: 
    Loss:     0.3054 Accuracy: 0.925581
    Epoch 25, Batch 3: 
    Loss:     0.2552 Accuracy: 0.897674
    Epoch 25, Batch 4: 
    Loss:     0.2913 Accuracy: 0.911628
    Epoch Number: 26
    Epoch 26, Batch 0: 
    Loss:     0.2402 Accuracy: 0.902326
    Epoch 26, Batch 1: 
    Loss:     0.2726 Accuracy: 0.916279
    Epoch 26, Batch 2: 
    Loss:     0.2459 Accuracy: 0.911628
    Epoch 26, Batch 3: 
    Loss:     0.3474 Accuracy: 0.920930
    Epoch 26, Batch 4: 
    Loss:     0.3827 Accuracy: 0.897674
    Epoch Number: 27
    Epoch 27, Batch 0: 
    Loss:     0.4583 Accuracy: 0.846512
    Epoch 27, Batch 1: 
    Loss:     0.3169 Accuracy: 0.925581
    Epoch 27, Batch 2: 
    Loss:     0.1770 Accuracy: 0.916279
    Epoch 27, Batch 3: 
    Loss:     0.4149 Accuracy: 0.916279
    Epoch 27, Batch 4: 
    Loss:     0.2469 Accuracy: 0.911628
    Epoch Number: 28
    Epoch 28, Batch 0: 
    Loss:     0.3696 Accuracy: 0.911628
    Epoch 28, Batch 1: 
    Loss:     0.2879 Accuracy: 0.906977
    Epoch 28, Batch 2: 
    Loss:     0.3327 Accuracy: 0.930233
    Epoch 28, Batch 3: 
    Loss:     0.2654 Accuracy: 0.920930
    Epoch 28, Batch 4: 
    Loss:     0.2968 Accuracy: 0.939535
    Epoch Number: 29
    Epoch 29, Batch 0: 
    Loss:     0.2039 Accuracy: 0.944186
    Epoch 29, Batch 1: 
    Loss:     0.1878 Accuracy: 0.925581
    Epoch 29, Batch 2: 
    Loss:     0.1798 Accuracy: 0.948837
    Epoch 29, Batch 3: 
    Loss:     0.3496 Accuracy: 0.944186
    Epoch 29, Batch 4: 
    Loss:     0.3034 Accuracy: 0.902326
    Epoch Number: 30
    Epoch 30, Batch 0: 
    Loss:     0.2745 Accuracy: 0.930233
    Epoch 30, Batch 1: 
    Loss:     0.2296 Accuracy: 0.930233
    Epoch 30, Batch 2: 
    Loss:     0.1637 Accuracy: 0.948837
    Epoch 30, Batch 3: 
    Loss:     0.2211 Accuracy: 0.948837
    Epoch 30, Batch 4: 
    Loss:     0.2207 Accuracy: 0.948837
    Epoch Number: 31
    Epoch 31, Batch 0: 
    Loss:     0.2539 Accuracy: 0.916279
    Epoch 31, Batch 1: 
    Loss:     0.2309 Accuracy: 0.911628
    Epoch 31, Batch 2: 
    Loss:     0.1960 Accuracy: 0.930233
    Epoch 31, Batch 3: 
    Loss:     0.1660 Accuracy: 0.953488
    Epoch 31, Batch 4: 
    Loss:     0.2020 Accuracy: 0.934884
    Epoch Number: 32
    Epoch 32, Batch 0: 
    Loss:     0.2588 Accuracy: 0.944186
    Epoch 32, Batch 1: 
    Loss:     0.3699 Accuracy: 0.934884
    Epoch 32, Batch 2: 
    Loss:     0.2672 Accuracy: 0.953488
    Epoch 32, Batch 3: 
    Loss:     0.3084 Accuracy: 0.958140
    Epoch 32, Batch 4: 
    Loss:     0.4670 Accuracy: 0.893023
    Epoch Number: 33
    Epoch 33, Batch 0: 
    Loss:     2.3302 Accuracy: 0.832558
    Epoch 33, Batch 1: 
    Loss:     0.5536 Accuracy: 0.702326
    Epoch 33, Batch 2: 
    Loss:     0.5205 Accuracy: 0.711628
    Epoch 33, Batch 3: 
    Loss:     0.5239 Accuracy: 0.706977
    Epoch 33, Batch 4: 
    Loss:     0.4649 Accuracy: 0.776744
    Epoch Number: 34
    Epoch 34, Batch 0: 
    Loss:     0.4516 Accuracy: 0.772093
    Epoch 34, Batch 1: 
    Loss:     0.4279 Accuracy: 0.786047
    Epoch 34, Batch 2: 
    Loss:     0.3886 Accuracy: 0.837209
    Epoch 34, Batch 3: 
    Loss:     0.3157 Accuracy: 0.855814
    Epoch 34, Batch 4: 
    Loss:     0.2706 Accuracy: 0.888372
    Epoch Number: 35
    Epoch 35, Batch 0: 
    Loss:     0.3596 Accuracy: 0.888372
    Epoch 35, Batch 1: 
    Loss:     0.2702 Accuracy: 0.902326
    Epoch 35, Batch 2: 
    Loss:     0.2357 Accuracy: 0.930233
    Epoch 35, Batch 3: 
    Loss:     0.3069 Accuracy: 0.920930
    Epoch 35, Batch 4: 
    Loss:     0.2663 Accuracy: 0.953488
    Epoch Number: 36
    Epoch 36, Batch 0: 
    Loss:     0.3707 Accuracy: 0.911628
    Epoch 36, Batch 1: 
    Loss:     0.3556 Accuracy: 0.934884
    Epoch 36, Batch 2: 
    Loss:     0.3403 Accuracy: 0.920930
    Epoch 36, Batch 3: 
    Loss:     0.3379 Accuracy: 0.916279
    Epoch 36, Batch 4: 
    Loss:     0.4710 Accuracy: 0.911628
    Epoch Number: 37
    Epoch 37, Batch 0: 
    Loss:     0.3905 Accuracy: 0.930233
    Epoch 37, Batch 1: 
    Loss:     0.5770 Accuracy: 0.860465
    Epoch 37, Batch 2: 
    Loss:     0.2415 Accuracy: 0.939535
    Epoch 37, Batch 3: 
    Loss:     0.2485 Accuracy: 0.948837
    Epoch 37, Batch 4: 
    Loss:     0.2794 Accuracy: 0.939535
    Epoch Number: 38
    Epoch 38, Batch 0: 
    Loss:     0.3799 Accuracy: 0.911628
    Epoch 38, Batch 1: 
    Loss:     0.2574 Accuracy: 0.939535
    Epoch 38, Batch 2: 
    Loss:     0.2528 Accuracy: 0.934884
    Epoch 38, Batch 3: 
    Loss:     0.1997 Accuracy: 0.962791
    Epoch 38, Batch 4: 
    Loss:     0.2515 Accuracy: 0.944186
    Epoch Number: 39
    Epoch 39, Batch 0: 
    Loss:     0.3510 Accuracy: 0.916279
    Epoch 39, Batch 1: 
    Loss:     0.2926 Accuracy: 0.930233
    Epoch 39, Batch 2: 
    Loss:     0.2098 Accuracy: 0.934884
    Epoch 39, Batch 3: 
    Loss:     0.2298 Accuracy: 0.925581
    Epoch 39, Batch 4: 
    Loss:     0.2227 Accuracy: 0.934884
    Epoch Number: 40
    Epoch 40, Batch 0: 
    Loss:     0.2941 Accuracy: 0.925581
    Epoch 40, Batch 1: 
    Loss:     0.2550 Accuracy: 0.939535
    Epoch 40, Batch 2: 
    Loss:     0.1914 Accuracy: 0.934884
    Epoch 40, Batch 3: 
    Loss:     0.2208 Accuracy: 0.958140
    Epoch 40, Batch 4: 
    Loss:     0.2569 Accuracy: 0.948837
    Epoch Number: 41
    Epoch 41, Batch 0: 
    Loss:     0.3169 Accuracy: 0.902326
    Epoch 41, Batch 1: 
    Loss:     0.3237 Accuracy: 0.925581
    Epoch 41, Batch 2: 
    Loss:     0.4862 Accuracy: 0.920930
    Epoch 41, Batch 3: 
    Loss:     0.2128 Accuracy: 0.925581
    Epoch 41, Batch 4: 
    Loss:     0.1582 Accuracy: 0.958140
    Epoch Number: 42
    Epoch 42, Batch 0: 
    Loss:     0.1703 Accuracy: 0.953488
    Epoch 42, Batch 1: 
    Loss:     0.1983 Accuracy: 0.953488
    Epoch 42, Batch 2: 
    Loss:     0.2045 Accuracy: 0.939535
    Epoch 42, Batch 3: 
    Loss:     0.2302 Accuracy: 0.925581
    Epoch 42, Batch 4: 
    Loss:     0.1981 Accuracy: 0.930233
    Epoch Number: 43
    Epoch 43, Batch 0: 
    Loss:     0.2726 Accuracy: 0.930233
    Epoch 43, Batch 1: 
    Loss:     0.2631 Accuracy: 0.948837
    Epoch 43, Batch 2: 
    Loss:     0.2505 Accuracy: 0.939535
    Epoch 43, Batch 3: 
    Loss:     0.2325 Accuracy: 0.953488
    Epoch 43, Batch 4: 
    Loss:     0.3075 Accuracy: 0.911628
    Epoch Number: 44
    Epoch 44, Batch 0: 
    Loss:     0.3368 Accuracy: 0.916279
    Epoch 44, Batch 1: 
    Loss:     0.2411 Accuracy: 0.930233
    Epoch 44, Batch 2: 
    Loss:     0.3628 Accuracy: 0.939535
    Epoch 44, Batch 3: 
    Loss:     0.2218 Accuracy: 0.934884
    Epoch 44, Batch 4: 
    Loss:     0.1932 Accuracy: 0.944186
    Epoch Number: 45
    Epoch 45, Batch 0: 
    Loss:     0.1429 Accuracy: 0.958140
    Epoch 45, Batch 1: 
    Loss:     0.2420 Accuracy: 0.939535
    Epoch 45, Batch 2: 
    Loss:     0.4523 Accuracy: 0.911628
    Epoch 45, Batch 3: 
    Loss:     0.2500 Accuracy: 0.958140
    Epoch 45, Batch 4: 
    Loss:     0.4512 Accuracy: 0.916279
    Epoch Number: 46
    Epoch 46, Batch 0: 
    Loss:     0.6088 Accuracy: 0.841860
    Epoch 46, Batch 1: 
    Loss:     0.2631 Accuracy: 0.953488
    Epoch 46, Batch 2: 
    Loss:     0.2107 Accuracy: 0.958140
    Epoch 46, Batch 3: 
    Loss:     0.2349 Accuracy: 0.934884
    Epoch 46, Batch 4: 
    Loss:     0.2344 Accuracy: 0.953488
    Epoch Number: 47
    Epoch 47, Batch 0: 
    Loss:     0.2173 Accuracy: 0.958140
    Epoch 47, Batch 1: 
    Loss:     0.2190 Accuracy: 0.953488
    Epoch 47, Batch 2: 
    Loss:     0.1830 Accuracy: 0.944186
    Epoch 47, Batch 3: 
    Loss:     0.2256 Accuracy: 0.953488
    Epoch 47, Batch 4: 
    Loss:     0.2833 Accuracy: 0.939535
    Epoch Number: 48
    Epoch 48, Batch 0: 
    Loss:     0.1778 Accuracy: 0.953488
    Epoch 48, Batch 1: 
    Loss:     0.1340 Accuracy: 0.962791
    Epoch 48, Batch 2: 
    Loss:     0.1449 Accuracy: 0.962791
    Epoch 48, Batch 3: 
    Loss:     0.2180 Accuracy: 0.953488
    Epoch 48, Batch 4: 
    Loss:     0.3785 Accuracy: 0.934884
    Epoch Number: 49
    Epoch 49, Batch 0: 
    Loss:     0.2368 Accuracy: 0.944186
    Epoch 49, Batch 1: 
    Loss:     0.2491 Accuracy: 0.953488
    Epoch 49, Batch 2: 
    Loss:     0.1399 Accuracy: 0.934884
    Epoch 49, Batch 3: 
    Loss:     0.2133 Accuracy: 0.944186
    Epoch 49, Batch 4: 
    Loss:     0.1558 Accuracy: 0.944186
    Epoch Number: 50
    Epoch 50, Batch 0: 
    Loss:     0.1980 Accuracy: 0.944186
    Epoch 50, Batch 1: 
    Loss:     0.1844 Accuracy: 0.948837
    Epoch 50, Batch 2: 
    Loss:     0.2020 Accuracy: 0.948837
    Epoch 50, Batch 3: 
    Loss:     0.4052 Accuracy: 0.902326
    Epoch 50, Batch 4: 
    Loss:     0.3898 Accuracy: 0.809302
    Epoch Number: 51
    Epoch 51, Batch 0: 
    Loss:     0.6018 Accuracy: 0.800000
    Epoch 51, Batch 1: 
    Loss:     0.4207 Accuracy: 0.869767
    Epoch 51, Batch 2: 
    Loss:     0.2397 Accuracy: 0.883721
    Epoch 51, Batch 3: 
    Loss:     0.3317 Accuracy: 0.897674
    Epoch 51, Batch 4: 
    Loss:     0.2998 Accuracy: 0.888372
    Epoch Number: 52
    Epoch 52, Batch 0: 
    Loss:     0.2112 Accuracy: 0.916279
    Epoch 52, Batch 1: 
    Loss:     0.3054 Accuracy: 0.893023
    Epoch 52, Batch 2: 
    Loss:     0.1961 Accuracy: 0.920930
    Epoch 52, Batch 3: 
    Loss:     0.2663 Accuracy: 0.911628
    Epoch 52, Batch 4: 
    Loss:     0.2870 Accuracy: 0.925581
    Epoch Number: 53
    Epoch 53, Batch 0: 
    Loss:     0.2508 Accuracy: 0.934884
    Epoch 53, Batch 1: 
    Loss:     0.2463 Accuracy: 0.925581
    Epoch 53, Batch 2: 
    Loss:     0.2467 Accuracy: 0.944186
    Epoch 53, Batch 3: 
    Loss:     0.2414 Accuracy: 0.916279
    Epoch 53, Batch 4: 
    Loss:     0.4022 Accuracy: 0.939535
    Epoch Number: 54
    Epoch 54, Batch 0: 
    Loss:     0.2510 Accuracy: 0.944186
    Epoch 54, Batch 1: 
    Loss:     0.1928 Accuracy: 0.934884
    Epoch 54, Batch 2: 
    Loss:     0.3784 Accuracy: 0.911628
    Epoch 54, Batch 3: 
    Loss:     0.3736 Accuracy: 0.897674
    Epoch 54, Batch 4: 
    Loss:     0.3613 Accuracy: 0.953488
    Epoch Number: 55
    Epoch 55, Batch 0: 
    Loss:     0.4541 Accuracy: 0.934884
    Epoch 55, Batch 1: 
    Loss:     0.2323 Accuracy: 0.934884
    Epoch 55, Batch 2: 
    Loss:     0.2362 Accuracy: 0.948837
    Epoch 55, Batch 3: 
    Loss:     0.3191 Accuracy: 0.920930
    Epoch 55, Batch 4: 
    Loss:     0.3481 Accuracy: 0.939535
    Epoch Number: 56
    Epoch 56, Batch 0: 
    Loss:     0.8660 Accuracy: 0.888372
    Epoch 56, Batch 1: 
    Loss:     0.2202 Accuracy: 0.934884
    Epoch 56, Batch 2: 
    Loss:     0.1693 Accuracy: 0.930233
    Epoch 56, Batch 3: 
    Loss:     0.6013 Accuracy: 0.916279
    Epoch 56, Batch 4: 
    Loss:     0.2169 Accuracy: 0.930233
    Epoch Number: 57
    Epoch 57, Batch 0: 
    Loss:     0.2485 Accuracy: 0.948837
    Epoch 57, Batch 1: 
    Loss:     0.2368 Accuracy: 0.962791
    Epoch 57, Batch 2: 
    Loss:     0.3511 Accuracy: 0.958140
    Epoch 57, Batch 3: 
    Loss:     0.2719 Accuracy: 0.930233
    Epoch 57, Batch 4: 
    Loss:     0.3863 Accuracy: 0.953488
    Epoch Number: 58
    Epoch 58, Batch 0: 
    Loss:     0.3608 Accuracy: 0.944186
    Epoch 58, Batch 1: 
    Loss:     0.2896 Accuracy: 0.948837
    Epoch 58, Batch 2: 
    Loss:     0.2592 Accuracy: 0.953488
    Epoch 58, Batch 3: 
    Loss:     0.3868 Accuracy: 0.948837
    Epoch 58, Batch 4: 
    Loss:     0.4212 Accuracy: 0.953488
    Epoch Number: 59
    Epoch 59, Batch 0: 
    Loss:     0.3768 Accuracy: 0.930233
    Epoch 59, Batch 1: 
    Loss:     0.4117 Accuracy: 0.939535
    Epoch 59, Batch 2: 
    Loss:     0.3371 Accuracy: 0.944186
    Epoch 59, Batch 3: 
    Loss:     0.3110 Accuracy: 0.962791
    Epoch 59, Batch 4: 
    Loss:     0.3508 Accuracy: 0.944186
    Epoch Number: 60
    Epoch 60, Batch 0: 
    Loss:     0.3979 Accuracy: 0.930233
    Epoch 60, Batch 1: 
    Loss:     0.3293 Accuracy: 0.953488
    Epoch 60, Batch 2: 
    Loss:     0.2709 Accuracy: 0.948837
    Epoch 60, Batch 3: 
    Loss:     0.2694 Accuracy: 0.944186
    Epoch 60, Batch 4: 
    Loss:     0.2904 Accuracy: 0.939535
    Epoch Number: 61
    Epoch 61, Batch 0: 
    Loss:     0.2579 Accuracy: 0.944186
    Epoch 61, Batch 1: 
    Loss:     0.2038 Accuracy: 0.958140
    Epoch 61, Batch 2: 
    Loss:     0.2115 Accuracy: 0.958140
    Epoch 61, Batch 3: 
    Loss:     0.2034 Accuracy: 0.958140
    Epoch 61, Batch 4: 
    Loss:     0.2935 Accuracy: 0.953488
    Epoch Number: 62
    Epoch 62, Batch 0: 
    Loss:     0.4342 Accuracy: 0.962791
    Epoch 62, Batch 1: 
    Loss:     0.2803 Accuracy: 0.948837
    Epoch 62, Batch 2: 
    Loss:     0.2357 Accuracy: 0.953488
    Epoch 62, Batch 3: 
    Loss:     0.2941 Accuracy: 0.953488
    Epoch 62, Batch 4: 
    Loss:     0.3100 Accuracy: 0.939535
    Epoch Number: 63
    Epoch 63, Batch 0: 
    Loss:     0.3674 Accuracy: 0.948837
    Epoch 63, Batch 1: 
    Loss:     0.3590 Accuracy: 0.962791
    Epoch 63, Batch 2: 
    Loss:     0.4243 Accuracy: 0.953488
    Epoch 63, Batch 3: 
    Loss:     0.6699 Accuracy: 0.869767
    Epoch 63, Batch 4: 
    Loss:     0.3654 Accuracy: 0.934884
    Epoch Number: 64
    Epoch 64, Batch 0: 
    Loss:     0.2628 Accuracy: 0.934884
    Epoch 64, Batch 1: 
    Loss:     0.1699 Accuracy: 0.930233
    Epoch 64, Batch 2: 
    Loss:     0.2107 Accuracy: 0.944186
    Epoch 64, Batch 3: 
    Loss:     0.3443 Accuracy: 0.934884
    Epoch 64, Batch 4: 
    Loss:     0.1497 Accuracy: 0.953488
    Epoch Number: 65
    Epoch 65, Batch 0: 
    Loss:     0.2117 Accuracy: 0.948837
    Epoch 65, Batch 1: 
    Loss:     0.3474 Accuracy: 0.948837
    Epoch 65, Batch 2: 
    Loss:     0.1992 Accuracy: 0.934884
    Epoch 65, Batch 3: 
    Loss:     0.2240 Accuracy: 0.958140
    Epoch 65, Batch 4: 
    Loss:     0.2229 Accuracy: 0.953488
    Epoch Number: 66
    Epoch 66, Batch 0: 
    Loss:     0.2373 Accuracy: 0.953488
    Epoch 66, Batch 1: 
    Loss:     0.3188 Accuracy: 0.944186
    Epoch 66, Batch 2: 
    Loss:     0.3149 Accuracy: 0.958140
    Epoch 66, Batch 3: 
    Loss:     0.3437 Accuracy: 0.948837
    Epoch 66, Batch 4: 
    Loss:     0.3797 Accuracy: 0.953488
    Epoch Number: 67
    Epoch 67, Batch 0: 
    Loss:     0.3467 Accuracy: 0.962791
    Epoch 67, Batch 1: 
    Loss:     0.3228 Accuracy: 0.967442
    Epoch 67, Batch 2: 
    Loss:     0.3959 Accuracy: 0.958140
    Epoch 67, Batch 3: 
    Loss:     0.4809 Accuracy: 0.944186
    Epoch 67, Batch 4: 
    Loss:     0.5338 Accuracy: 0.920930
    Epoch Number: 68
    Epoch 68, Batch 0: 
    Loss:     0.7027 Accuracy: 0.934884
    Epoch 68, Batch 1: 
    Loss:     0.7062 Accuracy: 0.846512
    Epoch 68, Batch 2: 
    Loss:     0.8437 Accuracy: 0.883721
    Epoch 68, Batch 3: 
    Loss:     0.2314 Accuracy: 0.897674
    Epoch 68, Batch 4: 
    Loss:     0.2648 Accuracy: 0.925581
    Epoch Number: 69
    Epoch 69, Batch 0: 
    Loss:     0.2976 Accuracy: 0.902326
    Epoch 69, Batch 1: 
    Loss:     0.2541 Accuracy: 0.925581
    Epoch 69, Batch 2: 
    Loss:     0.2758 Accuracy: 0.934884
    Epoch 69, Batch 3: 
    Loss:     0.2810 Accuracy: 0.939535
    Epoch 69, Batch 4: 
    Loss:     0.2626 Accuracy: 0.911628
    Epoch Number: 70
    Epoch 70, Batch 0: 
    Loss:     0.2128 Accuracy: 0.944186
    Epoch 70, Batch 1: 
    Loss:     0.2118 Accuracy: 0.948837
    Epoch 70, Batch 2: 
    Loss:     0.2658 Accuracy: 0.948837
    Epoch 70, Batch 3: 
    Loss:     0.3105 Accuracy: 0.958140
    Epoch 70, Batch 4: 
    Loss:     0.2980 Accuracy: 0.934884
    Epoch Number: 71
    Epoch 71, Batch 0: 
    Loss:     0.2794 Accuracy: 0.930233
    Epoch 71, Batch 1: 
    Loss:     0.3128 Accuracy: 0.948837
    Epoch 71, Batch 2: 
    Loss:     0.4253 Accuracy: 0.916279
    Epoch 71, Batch 3: 
    Loss:     0.2760 Accuracy: 0.944186
    Epoch 71, Batch 4: 
    Loss:     0.3296 Accuracy: 0.953488
    Epoch Number: 72
    Epoch 72, Batch 0: 
    Loss:     0.3584 Accuracy: 0.958140
    Epoch 72, Batch 1: 
    Loss:     0.4186 Accuracy: 0.930233
    Epoch 72, Batch 2: 
    Loss:     0.2927 Accuracy: 0.920930
    Epoch 72, Batch 3: 
    Loss:     0.3178 Accuracy: 0.930233
    Epoch 72, Batch 4: 
    Loss:     0.3110 Accuracy: 0.934884
    Epoch Number: 73
    Epoch 73, Batch 0: 
    Loss:     0.4884 Accuracy: 0.939535
    Epoch 73, Batch 1: 
    Loss:     0.3680 Accuracy: 0.939535
    Epoch 73, Batch 2: 
    Loss:     0.2346 Accuracy: 0.958140
    Epoch 73, Batch 3: 
    Loss:     0.2364 Accuracy: 0.953488
    Epoch 73, Batch 4: 
    Loss:     0.2710 Accuracy: 0.948837
    Epoch Number: 74
    Epoch 74, Batch 0: 
    Loss:     0.2804 Accuracy: 0.958140
    Epoch 74, Batch 1: 
    Loss:     0.2529 Accuracy: 0.953488
    Epoch 74, Batch 2: 
    Loss:     0.2099 Accuracy: 0.953488
    Epoch 74, Batch 3: 
    Loss:     0.2424 Accuracy: 0.958140
    Epoch 74, Batch 4: 
    Loss:     0.2462 Accuracy: 0.939535
    Epoch Number: 75
    Epoch 75, Batch 0: 
    Loss:     0.3696 Accuracy: 0.948837
    Epoch 75, Batch 1: 
    Loss:     0.3152 Accuracy: 0.920930
    Epoch 75, Batch 2: 
    Loss:     0.2246 Accuracy: 0.953488
    Epoch 75, Batch 3: 
    Loss:     0.2601 Accuracy: 0.934884
    Epoch 75, Batch 4: 
    Loss:     0.2776 Accuracy: 0.944186
    Epoch Number: 76
    Epoch 76, Batch 0: 
    Loss:     0.3021 Accuracy: 0.962791
    Epoch 76, Batch 1: 
    Loss:     0.3420 Accuracy: 0.967442
    Epoch 76, Batch 2: 
    Loss:     0.3697 Accuracy: 0.953488
    Epoch 76, Batch 3: 
    Loss:     0.3339 Accuracy: 0.948837
    Epoch 76, Batch 4: 
    Loss:     0.3678 Accuracy: 0.948837
    Epoch Number: 77
    Epoch 77, Batch 0: 
    Loss:     0.5419 Accuracy: 0.911628
    Epoch 77, Batch 1: 
    Loss:     0.4337 Accuracy: 0.934884
    Epoch 77, Batch 2: 
    Loss:     0.3589 Accuracy: 0.939535
    Epoch 77, Batch 3: 
    Loss:     0.4386 Accuracy: 0.944186
    Epoch 77, Batch 4: 
    Loss:     0.5043 Accuracy: 0.920930
    Epoch Number: 78
    Epoch 78, Batch 0: 
    Loss:     0.5063 Accuracy: 0.939535
    Epoch 78, Batch 1: 
    Loss:     0.4255 Accuracy: 0.939535
    Epoch 78, Batch 2: 
    Loss:     0.3956 Accuracy: 0.934884
    Epoch 78, Batch 3: 
    Loss:     0.3178 Accuracy: 0.962791
    Epoch 78, Batch 4: 
    Loss:     0.4245 Accuracy: 0.948837
    Epoch Number: 79
    Epoch 79, Batch 0: 
    Loss:     0.4694 Accuracy: 0.939535
    Epoch 79, Batch 1: 
    Loss:     0.5149 Accuracy: 0.944186
    Epoch 79, Batch 2: 
    Loss:     0.3716 Accuracy: 0.948837
    Epoch 79, Batch 3: 
    Loss:     0.3513 Accuracy: 0.948837
    Epoch 79, Batch 4: 
    Loss:     0.5002 Accuracy: 0.920930
    Epoch Number: 80
    Epoch 80, Batch 0: 
    Loss:     0.3704 Accuracy: 0.948837
    Epoch 80, Batch 1: 
    Loss:     0.4132 Accuracy: 0.948837
    Epoch 80, Batch 2: 
    Loss:     0.4380 Accuracy: 0.948837
    Epoch 80, Batch 3: 
    Loss:     0.4981 Accuracy: 0.930233
    Epoch 80, Batch 4: 
    Loss:     0.3605 Accuracy: 0.958140
    Epoch Number: 81
    Epoch 81, Batch 0: 
    Loss:     0.2695 Accuracy: 0.944186
    Epoch 81, Batch 1: 
    Loss:     0.2840 Accuracy: 0.953488
    Epoch 81, Batch 2: 
    Loss:     0.3014 Accuracy: 0.944186
    Epoch 81, Batch 3: 
    Loss:     0.3232 Accuracy: 0.939535
    Epoch 81, Batch 4: 
    Loss:     0.3901 Accuracy: 0.930233
    Epoch Number: 82
    Epoch 82, Batch 0: 
    Loss:     0.4340 Accuracy: 0.944186
    Epoch 82, Batch 1: 
    Loss:     0.3872 Accuracy: 0.958140
    Epoch 82, Batch 2: 
    Loss:     0.3748 Accuracy: 0.962791
    Epoch 82, Batch 3: 
    Loss:     0.3995 Accuracy: 0.967442
    Epoch 82, Batch 4: 
    Loss:     0.5296 Accuracy: 0.958140
    Epoch Number: 83
    Epoch 83, Batch 0: 
    Loss:     1.0387 Accuracy: 0.851163
    Epoch 83, Batch 1: 
    Loss:     0.2581 Accuracy: 0.925581
    Epoch 83, Batch 2: 
    Loss:     0.2639 Accuracy: 0.944186
    Epoch 83, Batch 3: 
    Loss:     0.3061 Accuracy: 0.948837
    Epoch 83, Batch 4: 
    Loss:     0.3497 Accuracy: 0.939535
    Epoch Number: 84
    Epoch 84, Batch 0: 
    Loss:     1.0478 Accuracy: 0.818605
    Epoch 84, Batch 1: 
    Loss:     0.4734 Accuracy: 0.851163
    Epoch 84, Batch 2: 
    Loss:     0.3150 Accuracy: 0.897674
    Epoch 84, Batch 3: 
    Loss:     0.2625 Accuracy: 0.906977
    Epoch 84, Batch 4: 
    Loss:     0.5208 Accuracy: 0.916279
    Epoch Number: 85
    Epoch 85, Batch 0: 
    Loss:     0.3538 Accuracy: 0.934884
    Epoch 85, Batch 1: 
    Loss:     0.3365 Accuracy: 0.944186
    Epoch 85, Batch 2: 
    Loss:     0.2561 Accuracy: 0.948837
    Epoch 85, Batch 3: 
    Loss:     0.3515 Accuracy: 0.930233
    Epoch 85, Batch 4: 
    Loss:     0.3775 Accuracy: 0.934884
    Epoch Number: 86
    Epoch 86, Batch 0: 
    Loss:     0.4413 Accuracy: 0.930233
    Epoch 86, Batch 1: 
    Loss:     0.3733 Accuracy: 0.948837
    Epoch 86, Batch 2: 
    Loss:     0.4337 Accuracy: 0.967442
    Epoch 86, Batch 3: 
    Loss:     0.4591 Accuracy: 0.967442
    Epoch 86, Batch 4: 
    Loss:     0.4240 Accuracy: 0.944186
    Epoch Number: 87
    Epoch 87, Batch 0: 
    Loss:     0.5933 Accuracy: 0.930233
    Epoch 87, Batch 1: 
    Loss:     0.4057 Accuracy: 0.920930
    Epoch 87, Batch 2: 
    Loss:     0.2050 Accuracy: 0.930233
    Epoch 87, Batch 3: 
    Loss:     0.2016 Accuracy: 0.967442
    Epoch 87, Batch 4: 
    Loss:     0.2541 Accuracy: 0.953488
    Epoch Number: 88
    Epoch 88, Batch 0: 
    Loss:     0.2014 Accuracy: 0.958140
    Epoch 88, Batch 1: 
    Loss:     0.2385 Accuracy: 0.953488
    Epoch 88, Batch 2: 
    Loss:     0.2634 Accuracy: 0.958140
    Epoch 88, Batch 3: 
    Loss:     0.3832 Accuracy: 0.948837
    Epoch 88, Batch 4: 
    Loss:     0.4510 Accuracy: 0.948837
    Epoch Number: 89
    Epoch 89, Batch 0: 
    Loss:     0.3482 Accuracy: 0.967442
    Epoch 89, Batch 1: 
    Loss:     0.3461 Accuracy: 0.972093
    Epoch 89, Batch 2: 
    Loss:     0.3486 Accuracy: 0.958140
    Epoch 89, Batch 3: 
    Loss:     0.4382 Accuracy: 0.962791
    Epoch 89, Batch 4: 
    Loss:     0.4426 Accuracy: 0.953488
    Epoch Number: 90
    Epoch 90, Batch 0: 
    Loss:     0.4304 Accuracy: 0.953488
    Epoch 90, Batch 1: 
    Loss:     0.4155 Accuracy: 0.962791
    Epoch 90, Batch 2: 
    Loss:     0.4208 Accuracy: 0.948837
    Epoch 90, Batch 3: 
    Loss:     0.4959 Accuracy: 0.962791
    Epoch 90, Batch 4: 
    Loss:     0.5171 Accuracy: 0.962791
    Epoch Number: 91
    Epoch 91, Batch 0: 
    Loss:     0.5254 Accuracy: 0.962791
    Epoch 91, Batch 1: 
    Loss:     0.5230 Accuracy: 0.967442
    Epoch 91, Batch 2: 
    Loss:     0.4986 Accuracy: 0.953488
    Epoch 91, Batch 3: 
    Loss:     0.5108 Accuracy: 0.953488
    Epoch 91, Batch 4: 
    Loss:     0.5131 Accuracy: 0.953488
    Epoch Number: 92
    Epoch 92, Batch 0: 
    Loss:     0.5284 Accuracy: 0.962791
    Epoch 92, Batch 1: 
    Loss:     0.5599 Accuracy: 0.958140
    Epoch 92, Batch 2: 
    Loss:     0.5934 Accuracy: 0.953488
    Epoch 92, Batch 3: 
    Loss:     0.6106 Accuracy: 0.953488
    Epoch 92, Batch 4: 
    Loss:     0.9552 Accuracy: 0.944186
    Epoch Number: 93
    Epoch 93, Batch 0: 
    Loss:     1.1987 Accuracy: 0.827907
    Epoch 93, Batch 1: 
    Loss:     0.4372 Accuracy: 0.851163
    Epoch 93, Batch 2: 
    Loss:     0.2480 Accuracy: 0.930233
    Epoch 93, Batch 3: 
    Loss:     0.3106 Accuracy: 0.934884
    Epoch 93, Batch 4: 
    Loss:     0.3252 Accuracy: 0.948837
    Epoch Number: 94
    Epoch 94, Batch 0: 
    Loss:     0.5682 Accuracy: 0.888372
    Epoch 94, Batch 1: 
    Loss:     0.5111 Accuracy: 0.920930
    Epoch 94, Batch 2: 
    Loss:     0.3057 Accuracy: 0.939535
    Epoch 94, Batch 3: 
    Loss:     0.4308 Accuracy: 0.930233
    Epoch 94, Batch 4: 
    Loss:     0.3245 Accuracy: 0.906977
    Epoch Number: 95
    Epoch 95, Batch 0: 
    Loss:     0.2151 Accuracy: 0.944186
    Epoch 95, Batch 1: 
    Loss:     0.2173 Accuracy: 0.958140
    Epoch 95, Batch 2: 
    Loss:     0.2835 Accuracy: 0.944186
    Epoch 95, Batch 3: 
    Loss:     0.4240 Accuracy: 0.934884
    Epoch 95, Batch 4: 
    Loss:     0.4927 Accuracy: 0.939535
    Epoch Number: 96
    Epoch 96, Batch 0: 
    Loss:     0.2514 Accuracy: 0.962791
    Epoch 96, Batch 1: 
    Loss:     0.2909 Accuracy: 0.967442
    Epoch 96, Batch 2: 
    Loss:     0.3441 Accuracy: 0.958140
    Epoch 96, Batch 3: 
    Loss:     0.4875 Accuracy: 0.953488
    Epoch 96, Batch 4: 
    Loss:     0.6379 Accuracy: 0.958140
    Epoch Number: 97
    Epoch 97, Batch 0: 
    Loss:     0.8955 Accuracy: 0.939535
    Epoch 97, Batch 1: 
    Loss:     0.6847 Accuracy: 0.948837
    Epoch 97, Batch 2: 
    Loss:     0.3208 Accuracy: 0.916279
    Epoch 97, Batch 3: 
    Loss:     0.5979 Accuracy: 0.939535
    Epoch 97, Batch 4: 
    Loss:     0.5437 Accuracy: 0.925581
    Epoch Number: 98
    Epoch 98, Batch 0: 
    Loss:     0.7126 Accuracy: 0.888372
    Epoch 98, Batch 1: 
    Loss:     0.3424 Accuracy: 0.897674
    Epoch 98, Batch 2: 
    Loss:     0.1775 Accuracy: 0.944186
    Epoch 98, Batch 3: 
    Loss:     0.1991 Accuracy: 0.972093
    Epoch 98, Batch 4: 
    Loss:     0.2128 Accuracy: 0.972093
    Epoch Number: 99
    Epoch 99, Batch 0: 
    Loss:     0.3904 Accuracy: 0.958140
    Epoch 99, Batch 1: 
    Loss:     0.2812 Accuracy: 0.934884
    Epoch 99, Batch 2: 
    Loss:     0.2483 Accuracy: 0.953488
    Epoch 99, Batch 3: 
    Loss:     0.3509 Accuracy: 0.972093
    Epoch 99, Batch 4: 
    Loss:     0.3626 Accuracy: 0.958140
    Epoch Number: 100
    Epoch 100, Batch 0: 
    Loss:     0.3177 Accuracy: 0.962791
    Epoch 100, Batch 1: 
    Loss:     0.2636 Accuracy: 0.967442
    Epoch 100, Batch 2: 
    Loss:     0.2646 Accuracy: 0.939535
    Epoch 100, Batch 3: 
    Loss:     0.3276 Accuracy: 0.962791
    Epoch 100, Batch 4: 
    Loss:     0.3588 Accuracy: 0.962791
    Epoch Number: 101
    Epoch 101, Batch 0: 
    Loss:     0.3759 Accuracy: 0.953488
    Epoch 101, Batch 1: 
    Loss:     0.3953 Accuracy: 0.958140
    Epoch 101, Batch 2: 
    Loss:     0.4636 Accuracy: 0.916279
    Epoch 101, Batch 3: 
    Loss:     0.3216 Accuracy: 0.934884
    Epoch 101, Batch 4: 
    Loss:     0.2991 Accuracy: 0.939535
    Epoch Number: 102
    Epoch 102, Batch 0: 
    Loss:     0.4198 Accuracy: 0.897674
    Epoch 102, Batch 1: 
    Loss:     0.2184 Accuracy: 0.920930
    Epoch 102, Batch 2: 
    Loss:     0.2398 Accuracy: 0.962791
    Epoch 102, Batch 3: 
    Loss:     0.4082 Accuracy: 0.846512
    Epoch 102, Batch 4: 
    Loss:     0.3731 Accuracy: 0.851163
    Epoch Number: 103
    Epoch 103, Batch 0: 
    Loss:     0.3006 Accuracy: 0.860465
    Epoch 103, Batch 1: 
    Loss:     0.2645 Accuracy: 0.888372
    Epoch 103, Batch 2: 
    Loss:     0.2641 Accuracy: 0.902326
    Epoch 103, Batch 3: 
    Loss:     0.3380 Accuracy: 0.860465
    Epoch 103, Batch 4: 
    Loss:     0.2573 Accuracy: 0.925581
    Epoch Number: 104
    Epoch 104, Batch 0: 
    Loss:     0.2940 Accuracy: 0.911628
    Epoch 104, Batch 1: 
    Loss:     0.1744 Accuracy: 0.948837
    Epoch 104, Batch 2: 
    Loss:     0.2012 Accuracy: 0.930233
    Epoch 104, Batch 3: 
    Loss:     0.2573 Accuracy: 0.930233
    Epoch 104, Batch 4: 
    Loss:     0.2727 Accuracy: 0.925581
    Epoch Number: 105
    Epoch 105, Batch 0: 
    Loss:     0.2138 Accuracy: 0.944186
    Epoch 105, Batch 1: 
    Loss:     0.2682 Accuracy: 0.916279
    Epoch 105, Batch 2: 
    Loss:     0.2086 Accuracy: 0.948837
    Epoch 105, Batch 3: 
    Loss:     0.2148 Accuracy: 0.925581
    Epoch 105, Batch 4: 
    Loss:     0.2636 Accuracy: 0.948837
    Epoch Number: 106
    Epoch 106, Batch 0: 
    Loss:     0.2318 Accuracy: 0.948837
    Epoch 106, Batch 1: 
    Loss:     0.2394 Accuracy: 0.962791
    Epoch 106, Batch 2: 
    Loss:     0.2278 Accuracy: 0.953488
    Epoch 106, Batch 3: 
    Loss:     0.2433 Accuracy: 0.948837
    Epoch 106, Batch 4: 
    Loss:     0.2421 Accuracy: 0.967442
    Epoch Number: 107
    Epoch 107, Batch 0: 
    Loss:     0.2098 Accuracy: 0.958140
    Epoch 107, Batch 1: 
    Loss:     0.2204 Accuracy: 0.958140
    Epoch 107, Batch 2: 
    Loss:     0.2671 Accuracy: 0.948837
    Epoch 107, Batch 3: 
    Loss:     0.2421 Accuracy: 0.962791
    Epoch 107, Batch 4: 
    Loss:     0.2410 Accuracy: 0.962791
    Epoch Number: 108
    Epoch 108, Batch 0: 
    Loss:     0.2529 Accuracy: 0.958140
    Epoch 108, Batch 1: 
    Loss:     0.2421 Accuracy: 0.962791
    Epoch 108, Batch 2: 
    Loss:     0.3023 Accuracy: 0.953488
    Epoch 108, Batch 3: 
    Loss:     0.3101 Accuracy: 0.958140
    Epoch 108, Batch 4: 
    Loss:     0.3190 Accuracy: 0.953488
    Epoch Number: 109
    Epoch 109, Batch 0: 
    Loss:     0.3221 Accuracy: 0.958140
    Epoch 109, Batch 1: 
    Loss:     0.3139 Accuracy: 0.953488
    Epoch 109, Batch 2: 
    Loss:     0.3493 Accuracy: 0.948837
    Epoch 109, Batch 3: 
    Loss:     0.3526 Accuracy: 0.953488
    Epoch 109, Batch 4: 
    Loss:     0.3629 Accuracy: 0.953488
    Epoch Number: 110
    Epoch 110, Batch 0: 
    Loss:     0.3705 Accuracy: 0.948837
    Epoch 110, Batch 1: 
    Loss:     0.3626 Accuracy: 0.958140
    Epoch 110, Batch 2: 
    Loss:     0.3830 Accuracy: 0.958140
    Epoch 110, Batch 3: 
    Loss:     0.3906 Accuracy: 0.953488
    Epoch 110, Batch 4: 
    Loss:     0.3994 Accuracy: 0.953488
    Epoch Number: 111
    Epoch 111, Batch 0: 
    Loss:     0.4134 Accuracy: 0.948837
    Epoch 111, Batch 1: 
    Loss:     0.4006 Accuracy: 0.953488
    Epoch 111, Batch 2: 
    Loss:     0.4095 Accuracy: 0.953488
    Epoch 111, Batch 3: 
    Loss:     0.4152 Accuracy: 0.953488
    Epoch 111, Batch 4: 
    Loss:     0.4188 Accuracy: 0.953488
    Epoch Number: 112
    Epoch 112, Batch 0: 
    Loss:     0.4688 Accuracy: 0.958140
    Epoch 112, Batch 1: 
    Loss:     0.4816 Accuracy: 0.953488
    Epoch 112, Batch 2: 
    Loss:     0.5278 Accuracy: 0.944186
    Epoch 112, Batch 3: 
    Loss:     0.3786 Accuracy: 0.893023
    Epoch 112, Batch 4: 
    Loss:     0.2498 Accuracy: 0.944186
    Epoch Number: 113
    Epoch 113, Batch 0: 
    Loss:     0.4262 Accuracy: 0.916279
    Epoch 113, Batch 1: 
    Loss:     0.3614 Accuracy: 0.953488
    Epoch 113, Batch 2: 
    Loss:     0.1884 Accuracy: 0.944186
    Epoch 113, Batch 3: 
    Loss:     0.2071 Accuracy: 0.925581
    Epoch 113, Batch 4: 
    Loss:     0.2024 Accuracy: 0.948837
    Epoch Number: 114
    Epoch 114, Batch 0: 
    Loss:     0.2564 Accuracy: 0.944186
    Epoch 114, Batch 1: 
    Loss:     0.3056 Accuracy: 0.934884
    Epoch 114, Batch 2: 
    Loss:     0.3084 Accuracy: 0.934884
    Epoch 114, Batch 3: 
    Loss:     0.4757 Accuracy: 0.934884
    Epoch 114, Batch 4: 
    Loss:     0.4978 Accuracy: 0.953488
    Epoch Number: 115
    Epoch 115, Batch 0: 
    Loss:     0.4161 Accuracy: 0.962791
    Epoch 115, Batch 1: 
    Loss:     0.4019 Accuracy: 0.972093
    Epoch 115, Batch 2: 
    Loss:     0.4803 Accuracy: 0.958140
    Epoch 115, Batch 3: 
    Loss:     0.5085 Accuracy: 0.944186
    Epoch 115, Batch 4: 
    Loss:     0.4995 Accuracy: 0.948837
    Epoch Number: 116
    Epoch 116, Batch 0: 
    Loss:     0.4295 Accuracy: 0.958140
    Epoch 116, Batch 1: 
    Loss:     0.4204 Accuracy: 0.967442
    Epoch 116, Batch 2: 
    Loss:     0.4279 Accuracy: 0.967442
    Epoch 116, Batch 3: 
    Loss:     0.4665 Accuracy: 0.958140
    Epoch 116, Batch 4: 
    Loss:     0.4502 Accuracy: 0.962791
    Epoch Number: 117
    Epoch 117, Batch 0: 
    Loss:     0.4936 Accuracy: 0.962791
    Epoch 117, Batch 1: 
    Loss:     0.3634 Accuracy: 0.953488
    Epoch 117, Batch 2: 
    Loss:     0.3484 Accuracy: 0.944186
    Epoch 117, Batch 3: 
    Loss:     0.4102 Accuracy: 0.958140
    Epoch 117, Batch 4: 
    Loss:     0.5322 Accuracy: 0.934884
    Epoch Number: 118
    Epoch 118, Batch 0: 
    Loss:     0.6111 Accuracy: 0.925581
    Epoch 118, Batch 1: 
    Loss:     0.4095 Accuracy: 0.948837
    Epoch 118, Batch 2: 
    Loss:     0.4219 Accuracy: 0.953488
    Epoch 118, Batch 3: 
    Loss:     0.4517 Accuracy: 0.953488
    Epoch 118, Batch 4: 
    Loss:     0.5744 Accuracy: 0.934884
    Epoch Number: 119
    Epoch 119, Batch 0: 
    Loss:     0.6023 Accuracy: 0.925581
    Epoch 119, Batch 1: 
    Loss:     0.6317 Accuracy: 0.920930
    Epoch 119, Batch 2: 
    Loss:     0.5149 Accuracy: 0.906977
    Epoch 119, Batch 3: 
    Loss:     0.5892 Accuracy: 0.962791
    Epoch 119, Batch 4: 
    Loss:     0.5051 Accuracy: 0.930233
    Epoch Number: 120
    Epoch 120, Batch 0: 
    Loss:     0.4471 Accuracy: 0.925581
    Epoch 120, Batch 1: 
    Loss:     0.4470 Accuracy: 0.939535
    Epoch 120, Batch 2: 
    Loss:     0.4181 Accuracy: 0.944186
    Epoch 120, Batch 3: 
    Loss:     0.4164 Accuracy: 0.944186
    Epoch 120, Batch 4: 
    Loss:     0.4378 Accuracy: 0.939535
    Epoch Number: 121
    Epoch 121, Batch 0: 
    Loss:     0.4410 Accuracy: 0.939535
    Epoch 121, Batch 1: 
    Loss:     0.4360 Accuracy: 0.948837
    Epoch 121, Batch 2: 
    Loss:     0.4516 Accuracy: 0.948837
    Epoch 121, Batch 3: 
    Loss:     0.4498 Accuracy: 0.948837
    Epoch 121, Batch 4: 
    Loss:     0.4531 Accuracy: 0.944186
    Epoch Number: 122
    Epoch 122, Batch 0: 
    Loss:     0.4546 Accuracy: 0.944186
    Epoch 122, Batch 1: 
    Loss:     0.4415 Accuracy: 0.948837
    Epoch 122, Batch 2: 
    Loss:     0.4535 Accuracy: 0.948837
    Epoch 122, Batch 3: 
    Loss:     0.4737 Accuracy: 0.948837
    Epoch 122, Batch 4: 
    Loss:     0.4827 Accuracy: 0.944186
    Epoch Number: 123
    Epoch 123, Batch 0: 
    Loss:     0.4876 Accuracy: 0.944186
    Epoch 123, Batch 1: 
    Loss:     0.4900 Accuracy: 0.944186
    Epoch 123, Batch 2: 
    Loss:     0.4946 Accuracy: 0.944186
    Epoch 123, Batch 3: 
    Loss:     0.5013 Accuracy: 0.944186
    Epoch 123, Batch 4: 
    Loss:     0.5070 Accuracy: 0.944186
    Epoch Number: 124
    Epoch 124, Batch 0: 
    Loss:     0.5121 Accuracy: 0.944186
    Epoch 124, Batch 1: 
    Loss:     0.5153 Accuracy: 0.944186
    Epoch 124, Batch 2: 
    Loss:     0.5176 Accuracy: 0.944186
    Epoch 124, Batch 3: 
    Loss:     0.5206 Accuracy: 0.944186
    Epoch 124, Batch 4: 
    Loss:     0.5230 Accuracy: 0.944186
    Training Completed


By tweaking the code and adjusting the hyperparameters, I've increased the accuracy from my prior submission by almost 20%. 


### Goal 3: Evaluate the neural network and show us the results
#### Testing the Model:

We will test the model using our test_images and test_labels data. We will do this by creating a method to test the model that will load the model with get_tensor_by_name and run it again our testing data. 


```python
save_model_path = "./training_sess"
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    #test_images, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_image_batch, test_label_batch, test_path_batch in get_mini_batches(batch_size, test_images, test_labels, test_paths):
            
            transposed_images = np.array(test_image_batch).transpose(0, 3, 1, 2)
            
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: transposed_images, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_images, random_test_labels = tuple(zip(*random.sample(list(zip(test_images, test_labels)), n_samples)))
        
        
        transposed_images = np.array(test_image_batch).transpose(0, 3, 1, 2)
            
            
        
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), 2),
            feed_dict={loaded_x: transposed_images, loaded_y: random_test_labels, loaded_keep_prob: 1.0})


test_model()
```

    Testing Accuracy: 0.9241071428571429
    


Woot! This testing accuracy is 8% higher than my previous submission. 

Although I could not get display_image_predictions to function, it was intended to display a few random images and the model's associated probabilites for each class in a table format. However, looking at the testing accuracy, the model's prospects look encouraging. 

### Goal 4: Testing the Network on Local Machines:

Running the following cells will allow you to input the file path of an image, and the model will provide you with a prediction.


```python
# Enter filepath of image
input_img = input()
```

    /Users/sarahhernandez/Documents/4. Important Docs/Passio/tissues.jpg



```python
# Enter 0 if it's Not Food, 1 if it's Food
input_label = input()
```

    0



```python
single_label = []
if input_label == str(0):
    single_label = [1, 0]
elif input_label == str(1):
    single_label = [0, 1]
else:
    print("Invalid label input, please run above cell again")
  
```


```python
img = cv2.imread(input_img)
img = cv2.resize(img,(256,256))
img = normalize(img)
img_stack = [img]
label_stack = [single_label]

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        

        test_batch_acc_total = 0
        test_batch_count = 0
        
       
        transposed_images = np.array(img_stack).transpose(0, 3, 1, 2)
            
        test_batch_acc_total += sess.run(loaded_acc, feed_dict={loaded_x: transposed_images, loaded_y: label_stack, loaded_keep_prob: 1.0})
        

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total))     
```

    1.0


Wonderful! If the testing accuracy is 1, then the model correctly predicted the given image, if the testing accuracy is 0, then the model failed to predict if the image was food or not food. 


#### Future Work

For further improvements to the program, I would implement the following:

* I would plot the training and validation accuracies vs. epochs to check for under and overfitting. 
* I would further expand the dataset by randomly cropping, translating and scaling images, or by using an online dataset
* I would implement early stopping to prevent overfitting 
* I'd like to see the results of implementing a k-fold cross validation, as it's good to use with limited datasets


```python

```
