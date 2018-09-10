import tensorflow as tf
import numpy as np
import pandas as pd

file_path = r'C:\MLproject\fashion-mnist_train.csv'

train = pd.read_csv(file_path, skiprows=1)
train_images = np.array(train.iloc[:, 1:])

test = pd.read_csv(file_path, skiprows=1)
test_images = np.array(test.iloc[:, 1:])


#Reshape input for CNN 28x28 array
train_images_cnn = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_labels_cnn = tf.keras.utils.to_categorical(np.array(train.iloc[:, 0]))

test_images_cnn = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_labels_cnn = tf.keras.utils.to_categorical(np.array(test.iloc[:, 0]))

#Parameters
OUTPUT_CLASSES = 10 #10 different items to classify
LEARNING_RATE = 0.0005
NUM_EPOCHS = 10
BATCH_SIZE = 10

def cnn_structure(input, labels, learning_rate):
    #input
    input_layer = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28, 28, 1])
    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10])

    #whether network is being trained
    isTraining = tf.placeholder(tf.bool)


    conv1 = tf.layers.conv2d(inputs=input_layer,  filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_fc_input = tf.reshape(pool2, shape=[-1, 7*7*64])

    fc_layer_1 = tf.layers.dense(inputs=pool2_fc_input, units=128, activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(inputs=fc_layer_1, rate=0.1, training= isTraining)  #add node dropout as network trains

    fc_layer_2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(inputs=fc_layer_2, rate=0.1, training= isTraining)  #add node dropout as network trains

    logits = tf.layers.dense(inputs=dropout2, units=10)

    with tf.name_scope("predictions"):
        predictions = {"classes" : tf.argmax(input=logits, axis=1), "probabilities" : tf.nn.softmax(logits=logits)}

    #loss function
    with tf.name_scope("loss_function"):
        cost = tf.losses.softmax_cross_entropy(labels, logits)

    #training backpropegation
    with tf.name_scope("Optimization"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))

    return {"input_layer":input_layer,
            "labels":labels,
            "logits": logits,
            "predictions": predictions,
            "cost": cost,
            "optimizer": optimizer,
            "accuracy": accuracy,
            "training": isTraining}

def next_batch(batch_size):

    '''
    if test == True:
        images = test_images
        labes = test_labels
    else:
        images = train_images
        labels = train_labels'''

    idx = np.arange(0 , 59999)
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [train_images_cnn[i] for i in idx]
    labels_shuffle = [train_labels_cnn[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def train(cnn):

    with tf.Session() as session:

        tf.summary.merge_all()
        session.run(tf.global_variables_initializer())

        for epoch in range(NUM_EPOCHS):
            avg_cost = 0.0
            avg_accuracy_pct = 0.0
            num_batches = int(60000/BATCH_SIZE)

            for i in range(num_batches):
                batch_x, batch_y = next_batch(BATCH_SIZE)

                _, batch_cost, batch_acc = session.run([cnn["optimizer"], cnn["cost"], cnn["accuracy"]], feed_dict = {cnn['input_layer']:batch_x, cnn['labels']:batch_y, cnn['training']:True})
                avg_cost += batch_cost / num_batches
                avg_accuracy_pct += batch_acc / num_batches


            print("cnn Epoch ", '%03d' % (epoch+1), ": error = ", '{:.3f}'.format(avg_cost), ", accuracy = ", '{:.2f}'.format(avg_accuracy_pct), "%", sep='')

def test(images,labels):

    with tf.Session() as session:
     accuracy =  session.run([cnn["accuracy"]], feed_dict = {cnn['input_layer']:batch_x, cnn['labels']:batch_y, cnn['training']:True})
     print(accuracy)

#Train and test model
cnn =  cnn_structure(train_images_cnn, train_labels_cnn, LEARNING_RATE)
train(cnn)
