import tensorflow as tf
from Blocks import Block

class VGG(tf.keras.Model):
    def __init__(self):
        super(VGG, self).__init__()

        self.block_a = Block(filters = 64, kernel_size = 3, repetitions = 2)
        self.block_b = Block(filters = 128, kernel_size = 3, repetitions = 2)
        self.block_c = Block(filters = 256, kernel_size = 3, repetitions = 3)
        self.block_d = Block(filters = 512, kernel_size = 3, repetitions = 3)
        self.block_e = Block(filters = 512, kernel_size = 3, repetitions = 3)

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(256, activation = 'relu')

        self.classifier = tf.keras.layers.Dense(10, activation = 'softmax')

    def call(self, inputs):
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        
        return x




