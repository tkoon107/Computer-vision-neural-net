import tensorflow as tf
import numpy as np
import pandas as pd

train = pd.read_csv(r'C:\MLproject\fashion-mnist_train.csv', skiprows=1)
train_images = np.array(train.iloc[:, 1:])
train_labels = tf.keras.utils.to_categorical(np.array(train.iloc[:, 0]))

test = pd.read_csv(r'C:\MLproject\fashion-mnist_test.csv', skiprows=1)
test_images = np.array(test.iloc[:, 1:])
test_labels = tf.keras.utils.to_categorical(np.array(test.iloc[:, 0]))

#Parameters
HIDDEN1 = 128 #Nodes in our first hidden layer
HIDDEN2 = 128 #Nodes in our second hidden layer
INPUT_SIZE = 784 #One for each pixel in 28x28 image
OUTPUT_CLASSES = 10 #10 different items to classify
LEARNING_RATE = 0.0004
NUM_EPOCHS = 10
BATCH_SIZE = 10

#weights to be used at each layer
weights = {
    'wl1': tf.get_variable('wl1', [INPUT_SIZE, HIDDEN1], initializer=tf.contrib.layers.xavier_initializer()),
    'wl2':  tf.get_variable('wl2', [HIDDEN1, HIDDEN2], initializer=tf.contrib.layers.xavier_initializer()),
    'w_out': tf.get_variable('w_out', [HIDDEN2, OUTPUT_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
    }

#biases to be used at each layer
biases = {
    'b1': tf.Variable(tf.zeros([HIDDEN1])),
    'b2': tf.Variable(tf.zeros([HIDDEN2])),
    'b_out': tf.Variable(tf.zeros([OUTPUT_CLASSES])),
}

#Create a fully connected layer that uses matrix multiplicaiton for weights and input, a relu function is then applied to the result
def fc_layer(x, W, b):
    with tf.name_scope("fully_connected_layer"):

        out = tf.add(tf.matmul(x, W), b)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", out)

        return tf.nn.relu(out)




def nn_structure(x, weights, biases, INPUT_SIZE):

    fc_layer1 = fc_layer(x, weights['wl1'], biases['b1'])
    fc_layer2 = fc_layer(fc_layer1, weights['wl2'], biases['b2'])
    output = tf.add(tf.matmul(fc_layer2, weights['w_out']), biases['b_out'])

    return output

#input placeholder
x = tf.placeholder(tf.float32, [None,INPUT_SIZE])

#ooutput label placeholder
y = tf.placeholder(tf.float32, [None,OUTPUT_CLASSES])

with tf.name_scope("Predictions"):
    predictions = nn_structure(x, weights, biases, INPUT_SIZE) #forward pass through network through 2 fully connected layers 128 nodes at each layer

with tf.name_scope("loss_function"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y)) #Cross entropy loss function

with tf.name_scope("Optimization"):
    optimizer = tf.train.AdamOptimizer().minimize(cost) #back prop using Adam optimizer

with tf.name_scope("Accuracy"):
    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy_pct = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) * 100
    tf.summary.scalar("accuracy", accuracy_pct)


def next_batch(batch_size):

    idx = np.arange(0 , 59999)
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [train_images[i] for i in idx]
    labels_shuffle = [train_labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train():

    with tf.Session() as session:

        tf.summary.merge_all()

        session.run(tf.global_variables_initializer())

        for epoch in range(NUM_EPOCHS):
            avg_cost = 0.0
            avg_accuracy_pct = 0.0
            num_batches = int(60000/BATCH_SIZE)

            for i in range(num_batches):
                batch_x, batch_y = next_batch(BATCH_SIZE)

                _, batch_cost, batch_acc = session.run([optimizer, cost, accuracy_pct], feed_dict = {x:batch_x, y:batch_y})
                avg_cost += batch_cost / num_batches
                avg_accuracy_pct += batch_acc / num_batches


            print("Epoch ", '%03d' % (epoch+1), ": error = ", '{:.3f}'.format(avg_cost), ", accuracy = ", '{:.2f}'.format(avg_accuracy_pct), "%", sep='')


def test(images,labels):

    with tf.Session() as session:
        session.run([predictions, correct_predictions, accuracy_pct], feed_dict = {x : images[i:],  y : labels[i:] })

train()
