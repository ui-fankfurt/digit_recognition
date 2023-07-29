import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class inception(tf.keras.Model):
    def __init__(self, nClasses):
        super(inception, self).__init__()
        self.layer_1a = Conv2D(10, (1,1), padding='same', activation='relu')
        self.layer_1b = Conv2D(10, (3,3), padding='same', activation='relu')

        ### 2nd layer
        self.layer_2a = Conv2D(10, (1,1), padding='same', activation='relu')
        self.layer_2b = Conv2D(10, (5,5), padding='same', activation='relu')

        ### 3rd layer
        self.layer_3a = MaxPooling2D((3,3), strides=(1,1), padding='same')
        self.layer_3b = Conv2D(10, (1,1), padding='same', activation='relu')

        ### Concatenate
        self.flat_1 = Flatten()

        self.dense_1 = Dense(1200, activation='relu')
        self.dense_2 = Dense(600, activation='relu')
        self.dense_3 = Dense(150, activation='relu')

        self.output_layer = Dense(nClasses, activation='softmax')  
    
    def call(self, inputs):
        x_1 = self.layer_1a(inputs)
        x_1 = self.layer_1b(x_1)

        x_2 = self.layer_2a(inputs)
        x_2 = self.layer_2b(x_2)

        x_3 = self.layer_3a(inputs)
        x_3 = self.layer_3b(x_3)

        mid_1 = tf.keras.layers.concatenate([x_1, x_2, x_3], axis=3)

        x = self.flat_1(mid_1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.output_layer(x)
        
        return x

class InceptionModel(tf.keras.Model):
    def __init__(self, nClasses):
        super(InceptionModel, self).__init__()
        self.inception_layer = inception(nClasses)

    def call(self, inputs):
        return self.inception_layer(inputs)
