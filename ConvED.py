from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
"test
l2 = tf.keras.regularizers.l2

class ConvBlock(tf.keras.Model):
  """Convolutional Block consisting of (batchnorm->relu->conv).
  """

  def __init__(self, output_filters, dropout_rate=0):
    super(ConvBlock, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(output_filters, (3, 3), padding="same", kernel_initializer="he_normal")
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.maxpool1 = tf.keras.layers.MaxPool2D((2, 2), 2, padding='same')

  def call(self, x, training=True):
    output = self.conv1(x)
    output = tf.nn.relu(self.batchnorm1(output))
    output = self.dropout1(output)
    output = self.maxpool1(output)
    return output 

class ConvTransposeBlock(tf.keras.Model):
  """Convolutional Transpose Block 
  """

  def __init__(self, output_filters, dropout_rate=0.3):
    super(ConvTransposeBlock, self).__init__()
    self.convTranspose = tf.keras.layers.Conv2DTranspose(output_filters, (3, 3), strides = (2, 2), padding="same", kernel_initializer="he_normal")
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, training=True):
    output = self.convTranspose(x)
    output = tf.nn.relu(self.batchnorm1(output))
    output = self.dropout1(output)
    return output

class ConvED(tf.keras.Model):
  """Simple Segnet model
  """

  def __init__(self, output_filters, dropout_rate=0.3):
    super(ConvED, self).__init__()
    self.encoder_blocks = []
    self.upsample = ConvTransposeBlock(1, dropout_rate)
    self.encoder_blocks.append(ConvBlock(64, dropout_rate))
    self.encoder_blocks.append(ConvBlock(128, dropout_rate))
    self.encoder_blocks.append(ConvBlock(256, dropout_rate))
    self.encoder_blocks.append(ConvBlock(512, dropout_rate))
    self.decoder_blocks = []
    self.decoder_blocks.append(ConvTransposeBlock(512, dropout_rate))
    self.decoder_blocks.append(ConvTransposeBlock(256, dropout_rate))
    self.decoder_blocks.append(ConvTransposeBlock(128, dropout_rate))
    self.decoder_blocks.append(ConvTransposeBlock(64, dropout_rate))
    self.downsample = tf.keras.layers.MaxPool2D((2, 2), 2, padding='same')
    self.conv = tf.keras.layers.Conv2D(output_filters, (3, 3), padding = "same", activation = "sigmoid", kernel_initializer="he_normal")

  def call(self, x, training=True):
    output = self.upsample(x)
    output = self.encoder_blocks[0](output)
    output = self.encoder_blocks[1](output, training = training)
    output = self.encoder_blocks[2](output, training = training)
    output = self.encoder_blocks[3](output, training = training)
    output = self.decoder_blocks[0](output, training = training)
    output = self.decoder_blocks[1](output, training = training)
    output = self.decoder_blocks[2](output, training = training)
    output = self.decoder_blocks[3](output, training = training)
    output = self.downsample(output)
    output = self.conv(output)
    return output

class ConvEDLSTM(tf.keras.Model):
  """ConvLSTM-SegNet model: Segnet + ConvLSTM layer
  """

  def __init__(self, output_filters, num_tstep, dropout_rate=0):
    super(ConvEDLSTM, self).__init__()
    self.encoder_blocks = []
    self.upsample = ConvTransposeBlock(1, dropout_rate)
    self.encoder_blocks.append(ConvBlock(64, dropout_rate))
    self.encoder_blocks.append(ConvBlock(128, dropout_rate))
    self.encoder_blocks.append(ConvBlock(256, dropout_rate))
    self.encoder_blocks.append(ConvBlock(512, dropout_rate))
    self.decoder_blocks = []
    self.decoder_blocks.append(ConvTransposeBlock(512, dropout_rate))
    self.decoder_blocks.append(ConvTransposeBlock(256, dropout_rate))
    self.decoder_blocks.append(ConvTransposeBlock(128, dropout_rate))
    self.decoder_blocks.append(ConvTransposeBlock(64, dropout_rate))
    self.downsample = tf.keras.layers.MaxPool2D((2, 2), 2, padding='same')
    self.convlstm1 = tf.keras.layers.ConvLSTM2D(5, (3, 3), padding="same", kernel_initializer="he_normal", return_sequences=True)
    self.conv1 = tf.keras.layers.Conv2D(num_tstep, (3, 3), padding = "same", activation = "relu", kernel_initializer="he_normal")
   # self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding = "same", activation = "relu", kernel_initializer="he_normal")
   # self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), padding = "same", activation = "relu", kernel_initializer="he_normal")
    self.conv2 = tf.keras.layers.Conv3D(output_filters, (3, 3, 3), padding = "same", activation = "sigmoid", kernel_initializer="he_normal")

  def call(self, x, training=True):
    output = self.upsample(x)
    output = self.encoder_blocks[0](output)
    output = self.encoder_blocks[1](output, training = training)
    output = self.encoder_blocks[2](output, training = training)
    output = self.encoder_blocks[3](output, training = training)
    output = self.decoder_blocks[0](output, training = training)
    output = self.decoder_blocks[1](output, training = training)
    output = self.decoder_blocks[2](output, training = training)
    output = self.decoder_blocks[3](output, training = training)
    output = self.downsample(output)
    output = self.conv1(output)
    output = tf.transpose(output, perm = [0, 3, 1, 2])
    output = tf.expand_dims(output, -1)
    output = self.convlstm1(output)
    output = self.conv2(output)

    return output
