import tensorflow as tf

class CBR(tf.keras.Model):
    def __init__(self, filters, kernel_size, **conv_params):
        super(CBR, self).__init__()
        self.conv = tf.layers.Conv2D(filters, kernel_size, activation=None, use_bias=False, **conv_params)
        self.bn = tf.layers.BatchNormalization(axis=-1)

    def __call__(self, inp, is_training=False):
        out = self.conv(inp)
        out = self.bn(out, training=is_training)
        out = tf.nn.relu(out)
        return out

class ASPP(tf.keras.Model):
    def __init__(self, **conv_params):
        super(ASPP, self).__init__()
        self.conv_1 = CBR(256, (1, 1), **conv_params)
        self.conv_2 = CBR(256, (3, 3), dilation_rate=6, **conv_params)
        self.conv_3 = CBR(256, (3, 3), dilation_rate=12, **conv_params)
        self.conv_4 = CBR(256, (3, 3), dilation_rate=18, **conv_params)
        self.conv_5 = CBR(256, (1, 1), **conv_params)

    def __call__(self, inp, is_training=False):
        out_1 = self.conv_1(inp, is_training)
        out_2 = self.conv_2(inp, is_training)
        out_3 = self.conv_3(inp, is_training)
        out_4 = self.conv_4(inp, is_training)

        out = tf.concat([out_1, out_2, out_3, out_4], axis=3)
        out = self.conv_5(out, is_training)

        return out

class Model(tf.keras.Model):
    def __init__(self, weight_decay=0.0):
        super(Model, self).__init__()

        conv_params = {
            'padding': 'same',
            'kernel_regularizer': tf.keras.regularizers.l2(weight_decay)
        }

        self.conv_1 = CBR(64, (3, 3), **conv_params)
        self.conv_2 = CBR(64, (3, 3), **conv_params)
        self.conv_3 = CBR(128, (3, 3), **conv_params)
        self.conv_4 = CBR(128, (3, 3), **conv_params)
        self.conv_5 = CBR(256, (3, 3), **conv_params)
        self.conv_6 = CBR(256, (3, 3), **conv_params)
        self.conv_7 = CBR(512, (3, 3), **conv_params)
        self.conv_8 = CBR(512, (3, 3), **conv_params)
        self.conv_9 = CBR(512, (3, 3), **conv_params)
        self.conv_10 = CBR(512, (3, 3), **conv_params)

        self.conv_11 = CBR(48, (1, 1), **conv_params)
        self.conv_12 = CBR(256, (3, 3), **conv_params)
        self.conv_13 = CBR(256, (3, 3), **conv_params)
        self.conv_14 = tf.layers.Conv2D(1, (1, 1), activation=None, padding='same')

        self.maxpool = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')

        self.aspp = ASPP(**conv_params)

    def __call__(self, inp, is_training=False):

        out = self.conv_1(inp, is_training)
        out = self.conv_2(out, is_training)
        out = self.maxpool(out)
        out = self.conv_3(out, is_training)
        out = self.conv_4(out, is_training)
        out = self.maxpool(out)
        out = self.conv_5(out, is_training)
        out = self.conv_6(out, is_training)
        out_enc_mid = out
        out = self.maxpool(out)
        out = self.conv_7(out, is_training)
        out = self.conv_8(out, is_training)
        out = self.maxpool(out)
        out = self.conv_9(out, is_training)
        out = self.conv_10(out, is_training)

        out = self.aspp(out, is_training)

        out = tf.image.resize_bilinear(out, tf.shape(out_enc_mid)[1:3])

        out_enc_mid = self.conv_11(out_enc_mid, is_training)

        out = tf.concat([out, out_enc_mid], axis=3)

        out = self.conv_12(out, is_training)
        out = self.conv_13(out, is_training)
        out = self.conv_14(out)

        out = tf.image.resize_bilinear(out, tf.shape(inp)[1:3])

        return out
