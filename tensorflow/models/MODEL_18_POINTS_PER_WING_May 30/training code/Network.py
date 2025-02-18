from constants import *
import vitPose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, MaxPooling2D, Concatenate, Lambda, Reshape, \
    Activation, Dropout, LeakyReLU, BatchNormalization, Resizing
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

ALPHA = 0.01


class Network:
    def __init__(self, config, image_size, number_of_output_channels):
        self.model_type = config['model type']
        self.image_size = image_size
        self.number_of_output_channels = number_of_output_channels
        self.num_base_filters = config["number of base filters"]
        self.num_blocks = config["number of encoder decoder blocks"]
        self.kernel_size = config["convolution kernel size"]
        self.learning_rate = config["learning rate"]
        self.optimizer = config["optimizer"]
        self.loss_function = config["loss_function"]
        self.dilation_rate = config["dilation rate"]
        self.dropout = config["dropout ratio"]
        self.batches_per_epoch = config["batches per epoch"]
        # self.do_attention = config["do_attention"]
        if "VIT" in self.model_type:
            self.patch_size = config["patch_size"],
            if type(self.patch_size) == tuple:
                self.patch_size = self.patch_size[0]
            self.projection_dim = config["projection_dim"]
            self.num_heads = config["num_heads"]
            self.transformer_layers = config["transformer_layers"]
            self.fully_connected_expand = config["fully_connected_expand"]
        self.model = self.config_model()

    def get_model(self):
        return self.model

    def config_model(self):
        if self.model_type == ALL_CAMS or self.model_type == ALL_CAMS_18_POINTS or self.model_type == ALL_CAMS_ALL_POINTS:
            model = self.all_4_cams()
        elif self.model_type == ALL_CAMS_AND_3_GOOD_CAMS:
            model = self.all_3_cams()
        elif self.model_type == TWO_WINGS_TOGATHER:
            model = self.two_wings_net()
        elif self.model_type == HEAD_TAIL_ALL_CAMS:
            model = self.head_tail_all_cams()
        elif self.model_type == C2F_PER_WING:
            model = self.C2F_per_wing()
        elif self.model_type == COARSE_PER_WING:
            model = self.coarse_per_wing()
        elif self.model_type == MODEL_18_POINTS_PER_WING_VIT or self.model_type == ALL_POINTS_MODEL_VIT:
            model = self.get_transformer()
        elif self.model_type == RESNET_18_POINTS_PER_WING:
            model = self.resnet50_encoder_shallow_decoder()
        else:
            model = self.basic_nn()
        return model

    def get_transformer(self):
        model_vit = vitPose.vision_transformer(image_size=self.image_size[0],
                                               patch_size=self.patch_size,
                                               projection_dim=self.projection_dim,
                                               num_heads=self.num_heads,
                                               transformer_layers=self.transformer_layers,
                                               num_input_channels=self.image_size[-1],
                                               num_output_channels=self.number_of_output_channels,
                                               fully_connected_expand=self.fully_connected_expand)
        return model_vit

    def head_tail_all_cams(self):
        x_in = Input(self.image_size, name="x_in")  # image size should be (M, M, num_channels * num cameras)
        num_cameras = 4
        # encoder encodes 1 image at a time
        shared_encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1],
                                                self.image_size[2] // num_cameras),
                                               self.num_base_filters, self.num_blocks, self.kernel_size,
                                               self.dilation_rate, self.dropout)
        shared_encoder.summary()

        # decoder accepts 1 encoder output concatenated with all other cameras encoders output so 1 + num cams
        shared_decoder = self.decoder2d(
            (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
             (1 + num_cameras) * shared_encoder.output_shape[3]),
            self.number_of_output_channels // num_cameras, self.num_base_filters, self.num_blocks, self.kernel_size)
        shared_decoder.summary()

        # spliting input of 12 channels to 3 different cameras
        # x_in_split_1 = Lambda(lambda x: x[..., 0:3], name="lambda_1")(x_in)
        # x_in_split_2 = Lambda(lambda x: x[..., 3:6], name="lambda_2")(x_in)
        # x_in_split_3 = Lambda(lambda x: x[..., 6:9], name="lambda_3")(x_in)
        # x_in_split_4 = Lambda(lambda x: x[..., 9:12], name="lambda_4")(x_in)

        # spliting input of 20 channels to 3 different cameras
        x_in_split_1 = Lambda(lambda x: x[..., 0:5], name="lambda_1")(x_in)
        x_in_split_2 = Lambda(lambda x: x[..., 5:10], name="lambda_2")(x_in)
        x_in_split_3 = Lambda(lambda x: x[..., 10:15], name="lambda_3")(x_in)
        x_in_split_4 = Lambda(lambda x: x[..., 15:20], name="lambda_4")(x_in)

        # different outputs of encoder
        code_out_1 = shared_encoder(x_in_split_1)
        code_out_2 = shared_encoder(x_in_split_2)
        code_out_3 = shared_encoder(x_in_split_3)
        code_out_4 = shared_encoder(x_in_split_4)

        # concatenated output of the 4 different encoders
        x_code_merge = Concatenate()([code_out_1, code_out_2, code_out_3, code_out_4])

        # prepare encoder's input as camera + concatenated latent vector of all cameras
        map_out_1 = shared_decoder(Concatenate()([code_out_1, x_code_merge]))
        map_out_2 = shared_decoder(Concatenate()([code_out_2, x_code_merge]))
        map_out_3 = shared_decoder(Concatenate()([code_out_3, x_code_merge]))
        map_out_4 = shared_decoder(Concatenate()([code_out_4, x_code_merge]))

        # merging all the encoders outputs, meaning we get a (M, M, num_pnts_per_wing * num_cams) confmaps
        x_maps_merge = Concatenate()([map_out_1, map_out_2, map_out_3, map_out_4])

        net = Model(inputs=x_in, outputs=x_maps_merge, name="head_tail_all_cams")
        net.summary()
        net.compile(optimizer=Adam(amsgrad=False),
                    loss=self.loss_function)
        return net

    def basic_nn(self):
        x_in = Input(self.image_size, name="x_in")
        encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1], self.image_size[2]),
                                        self.num_base_filters,
                                        self.num_blocks, self.kernel_size,
                                        self.dilation_rate,
                                        self.dropout)
        encoder.summary()
        decoder = self.decoder2d((encoder.output_shape[1], encoder.output_shape[2], encoder.output_shape[3]),
                                 self.number_of_output_channels, self.num_base_filters, self.num_blocks,
                                 self.kernel_size)
        decoder.summary()
        x_out = decoder(encoder(x_in))

        net = Model(inputs=x_in, outputs=x_out, name="basic_nn")
        net.summary()
        net.compile(optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_function)
        return net

    def coarse_per_wing(self):
        self.num_blocks = 3  # important!
        x_in = Input(self.image_size, name="x_in")
        encoder_p1 = self.encoder2d_atrous((self.image_size[0], self.image_size[1], self.image_size[2]),
                                           self.num_base_filters,
                                           self.num_blocks, self.kernel_size,
                                           self.dilation_rate,
                                           self.dropout, add_name="1")
        encoder_p1.summary()
        decoder_p1 = self.decoder2d(
            (encoder_p1.output_shape[1], encoder_p1.output_shape[2], encoder_p1.output_shape[3]),
            self.number_of_output_channels, self.num_base_filters, self.num_blocks,
            self.kernel_size, add_name="1")

        decoder_p1.summary()
        x_out = decoder_p1(encoder_p1(x_in))
        net = Model(inputs=x_in, outputs=x_out, name="coarse_per_wing")
        net.summary()
        net.compile(optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_function)
        return net

    def C2F_per_wing(self):
        x_in = Input(self.image_size, name="x_in")

        # part 1 load a trained fixed model for the coarse estimation:
        coarse_trained_model = r"coarse per wing sigma 6 model.h5"
        trained_model = models.load_model(coarse_trained_model)
        trained_model.trainable = False
        x_out_p1 = trained_model(x_in)  # x_out_p1 is (192, 192, 7) confmaps of sigma 6

        # part 2: run network on concatenated input (192, 192, 4) + (192, 192, 7) = (192, 192, 11)
        x_in_p2 = Concatenate()([x_in, x_out_p1])  # (192, 192, 4) cat (192, 192, 7) = (192, 192, 11)
        # continue the same
        encoder_p2 = self.encoder2d_atrous((x_in_p2.shape[1], x_in_p2.shape[2], x_in_p2.shape[3]),
                                           self.num_base_filters,
                                           self.num_blocks, self.kernel_size,
                                           self.dilation_rate,
                                           self.dropout, add_name="2")
        encoder_p2.summary()
        decoder_p2 = self.decoder2d(
            (encoder_p2.output_shape[1], encoder_p2.output_shape[2], encoder_p2.output_shape[3]),
            self.number_of_output_channels, self.num_base_filters, self.num_blocks,
            self.kernel_size, add_name="2")
        decoder_p2.summary()
        x_out = decoder_p2(encoder_p2(x_in_p2))

        net = Model(inputs=x_in, outputs=x_out, name="C2F_per_wing")
        net.summary()
        net.compile(optimizer=Adam(learning_rate=self.learning_rate),
                    loss=self.loss_function)
        return net

    def two_wings_net(self):
        x_in = Input(self.image_size, name="x_in")
        num_wings = 2
        num_time_channels = self.image_size[2] - 2
        # decoder gets an image of shape (M, N, num_time_channels + 1 wing)
        shared_encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1], num_time_channels + 1),
                                               self.num_base_filters, self.num_blocks, self.kernel_size,
                                               self.dilation_rate, self.dropout)
        shared_encoder.summary()

        # decoder gets 2 encoder outputs, [wing 1, wing 2] and outputs the points for wing 1 (7 points)
        shared_decoder = self.decoder2d(
            (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
             (num_wings + 0) * shared_encoder.output_shape[3]),
            self.number_of_output_channels // num_wings, self.num_base_filters, self.num_blocks, self.kernel_size)
        shared_decoder.summary()

        # todo a more general input
        x_in_wing_1 = Lambda(lambda x: tf.gather(x, [0, 1, 2, 3], axis=-1), name="lambda_1")(x_in)
        x_in_wing_2 = Lambda(lambda x: tf.gather(x, [0, 1, 2, 4], axis=-1), name="lambda_2")(x_in)

        # get inputs through the encoder and get a latents space representation
        code_out_1 = shared_encoder(x_in_wing_1)
        code_out_2 = shared_encoder(x_in_wing_2)

        merged_1 = Concatenate()([code_out_1, code_out_2])
        merged_2 = Concatenate()([code_out_2, code_out_1])

        # send to encoder: encoder_i = [wing_i, wing_j]
        map_out_1 = shared_decoder(Concatenate()([code_out_1, code_out_2]))
        map_out_2 = shared_decoder(Concatenate()([code_out_2, code_out_1]))
        #
        # map_out_1 = shared_decoder(Concatenate()([code_out_1, merged_1]))
        # map_out_2 = shared_decoder(Concatenate()([code_out_2, merged_2]))

        # arrange output
        x_maps_merge = Concatenate()([map_out_1, map_out_2])

        # get the model
        net = Model(inputs=x_in, outputs=x_maps_merge, name="two_wings_net")
        net.summary()
        net.compile(optimizer=Adam(amsgrad=False),
                    loss=self.loss_function)
        return net

    def all_3_cams(self):
        x_in = Input(self.image_size, name="x_in")  # image size should be (M, M, num_channels * num cameras)
        num_cameras = 3
        # encoder encodes 1 image at a time
        shared_encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1],
                                                self.image_size[2] // num_cameras),
                                               self.num_base_filters, self.num_blocks, self.kernel_size,
                                               self.dilation_rate, self.dropout)
        shared_encoder.summary()

        # for average
        # shared_decoder = decoder2d(
        #    (shared_encoder.output_shape[1], shared_encoder.output_shape[2], 2 * shared_encoder.output_shape[3])
        #    , num_output_channels, filters, num_blocks, kernel_size)
        # for cat

        # decoder accepts 1 encoder output concatenated with all other cameras encoders output so 1 + num cams
        shared_decoder = self.decoder2d(
            (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
             (1 + num_cameras) * shared_encoder.output_shape[3]),
            self.number_of_output_channels // num_cameras, self.num_base_filters, self.num_blocks, self.kernel_size)
        shared_decoder.summary()

        # spliting input of 12 channels to 3 different cameras
        num_input_channels = self.image_size[-1]
        if num_input_channels == 12:
            x_in_split_1 = Lambda(lambda x: x[..., 0:4], name="lambda_1")(x_in)
            x_in_split_2 = Lambda(lambda x: x[..., 4:8], name="lambda_2")(x_in)
            x_in_split_3 = Lambda(lambda x: x[..., 8:12], name="lambda_3")(x_in)
        elif num_input_channels == 28:
            x_in_split_1 = Lambda(lambda x: x[..., 0:6], name="lambda_1")(x_in)
            x_in_split_2 = Lambda(lambda x: x[..., 6:12], name="lambda_2")(x_in)
            x_in_split_3 = Lambda(lambda x: x[..., 12:18], name="lambda_3")(x_in)
        # different outputs of encoder
        code_out_1 = shared_encoder(x_in_split_1)
        code_out_2 = shared_encoder(x_in_split_2)
        code_out_3 = shared_encoder(x_in_split_3)

        # concatenated output of the 3 different encoders
        x_code_merge = Concatenate()([code_out_1, code_out_2, code_out_3])

        # shorter latent vector : each map_out gets only the other encoded vectors
        # map_out_1 = shared_decoder(Concatenate()([code_out_1, code_out_2, code_out_3, code_out_4]))
        # map_out_2 = shared_decoder(Concatenate()([code_out_2, code_out_1, code_out_3, code_out_4]))
        # map_out_3 = shared_decoder(Concatenate()([code_out_3, code_out_1, code_out_2, code_out_4]))
        # map_out_4 = shared_decoder(Concatenate()([code_out_4, code_out_1, code_out_2, code_out_3]))

        # prepare encoder's input as camera + concatenated latent vector of all cameras
        map_out_1 = shared_decoder(Concatenate()([code_out_1, x_code_merge]))
        map_out_2 = shared_decoder(Concatenate()([code_out_2, x_code_merge]))
        map_out_3 = shared_decoder(Concatenate()([code_out_3, x_code_merge]))

        # merging all the encoders outputs, meaning we get a (M, M, num_pnts_per_wing * num_cams) confmaps
        x_maps_merge = Concatenate()([map_out_1, map_out_2, map_out_3])

        net = Model(inputs=x_in, outputs=x_maps_merge, name="ed3d")
        net.summary()
        net.compile(optimizer=Adam(amsgrad=False),
                    loss=self.loss_function)
        # loss="categorical_crossentropy")
        return net

    @staticmethod
    def self_attention_layer(x, num_heads=8, key_dim=64):
        # Reshape to (batch_size, height * width, channels)
        batch_size, height, width, channels = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        x_reshaped = Reshape((height * width, channels))(x)

        # Apply Multi-Head Attention
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_reshaped, x_reshaped)

        # Reshape back to (batch_size, height, width, channels)
        attn_output_reshaped = Reshape((height, width, channels))(attn_output)

        return Add()([x, attn_output_reshaped])

    def all_4_cams(self):
        x_in = Input(self.image_size, name="x_in")  # image size should be (M, M, num_channels * num cameras)
        num_cameras = 4
        # encoder encodes 1 image at a time
        shared_encoder = self.encoder2d_atrous((self.image_size[0], self.image_size[1],
                                                self.image_size[2] // num_cameras),
                                               self.num_base_filters, self.num_blocks, self.kernel_size,
                                               self.dilation_rate, self.dropout)
        shared_encoder.summary()

        # decoder accepts 1 encoder output concatenated with all other cameras encoders output so 1 + num cams
        shared_decoder = self.decoder2d(
            (shared_encoder.output_shape[1], shared_encoder.output_shape[2],
             (1 + num_cameras) * shared_encoder.output_shape[3]),
            self.number_of_output_channels // num_cameras, self.num_base_filters, self.num_blocks, self.kernel_size)
        shared_decoder.summary()

        # spliting input of 12 channels to 3 different cameras
        num_input_channels = self.image_size[-1]
        if num_input_channels == 16:
            x_in_split_1 = Lambda(lambda x: x[..., 0:4])(x_in)
            x_in_split_2 = Lambda(lambda x: x[..., 4:8])(x_in)
            x_in_split_3 = Lambda(lambda x: x[..., 8:12])(x_in)
            x_in_split_4 = Lambda(lambda x: x[..., 12:16])(x_in)

        elif num_input_channels == 20:
            x_in_split_1 = Lambda(lambda x: x[..., 0:5])(x_in)
            x_in_split_2 = Lambda(lambda x: x[..., 5:10])(x_in)
            x_in_split_3 = Lambda(lambda x: x[..., 10:15])(x_in)
            x_in_split_4 = Lambda(lambda x: x[..., 15:20])(x_in)
        # different outputs of encoder
        code_out_1 = shared_encoder(x_in_split_1)
        code_out_2 = shared_encoder(x_in_split_2)
        code_out_3 = shared_encoder(x_in_split_3)
        code_out_4 = shared_encoder(x_in_split_4)

        # concatenated output of the 3 different encoders
        x_code_merge = Concatenate()([code_out_1, code_out_2, code_out_3, code_out_4])
        # if self.do_attention:
        #     x_code_merge = Network.self_attention_layer(x_code_merge)

        # prepare encoder's input as camera + concatenated latent vector of all cameras
        map_out_1 = shared_decoder(Concatenate()([code_out_1, x_code_merge]))
        map_out_2 = shared_decoder(Concatenate()([code_out_2, x_code_merge]))
        map_out_3 = shared_decoder(Concatenate()([code_out_3, x_code_merge]))
        map_out_4 = shared_decoder(Concatenate()([code_out_4, x_code_merge]))

        # merging all the encoders outputs, meaning we get a (M, M, num_pnts_per_wing * num_cams) confmaps
        x_maps_merge = Concatenate()([map_out_1, map_out_2, map_out_3, map_out_4])

        net = Model(inputs=x_in, outputs=x_maps_merge, name="ed3d")
        net.summary()
        net.compile(optimizer=Adam(amsgrad=False),
                    loss=self.loss_function)
        return net

    def resnet50_encoder_shallow_decoder(self):
        x_in = Input(self.image_size, name="x_in")

        # x_in = Resizing((self.image_size[0] * 2, self.image_size[0] * 2))(x_in)

        encoder = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=self.image_size)(x_in)

        decoder = Conv2DTranspose(encoder.shape[-1] // 2,
                                  kernel_size=self.kernel_size, strides=2, padding="same",
                                  activation=LeakyReLU(alpha=ALPHA),
                                  kernel_initializer="glorot_normal")(encoder)

        decoder = Conv2DTranspose(decoder.shape[-1] // 2,
                                  kernel_size=self.kernel_size, strides=2, padding="same",
                                  activation=LeakyReLU(alpha=ALPHA),
                                  kernel_initializer="glorot_normal")(decoder)

        decoder = Conv2DTranspose(decoder.shape[-1] // 2,
                                  kernel_size=self.kernel_size, strides=2, padding="same",
                                  activation=LeakyReLU(alpha=ALPHA),
                                  kernel_initializer="glorot_normal")(decoder)

        decoder = Conv2DTranspose(decoder.shape[-1] // 2,
                                  kernel_size=self.kernel_size, strides=2, padding="same",
                                  activation=LeakyReLU(alpha=ALPHA),
                                  kernel_initializer="glorot_normal")(decoder)

        x_out = Conv2DTranspose(self.number_of_output_channels,
                                kernel_size=self.kernel_size, strides=2, padding="same",
                                activation=LeakyReLU(alpha=ALPHA),
                                kernel_initializer="glorot_normal")(decoder)

        net = Model(inputs=x_in, outputs=x_out)
        net.summary()
        net.compile(optimizer=Adam(amsgrad=False),
                    loss=self.loss_function)
        return net

    @staticmethod
    def encoder2d_atrous(img_size, filters, num_blocks, kernel_size, dilation_rate, dropout, add_name=""):
        x_in = Input(img_size)
        dilation_rate = (dilation_rate, dilation_rate)
        for block_ind in range(num_blocks):
            if block_ind == 0:
                x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                               padding="same", activation=LeakyReLU(alpha=ALPHA))(x_in)
            else:
                x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                               padding="same", activation=LeakyReLU(alpha=ALPHA))(x_out)

            x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                           padding="same", activation=LeakyReLU(alpha=ALPHA))(x_out)

            x_out = Conv2D(filters * (2 ** block_ind), kernel_size, dilation_rate=dilation_rate,
                           padding="same", activation="linear")(x_out)

            x_out = MaxPooling2D(pool_size=2, strides=2, padding="same")(x_out)
            x_out = Activation('relu')(x_out)
            x_out = Dropout(dropout)(x_out)

        x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                       padding="same", activation=LeakyReLU(alpha=ALPHA))(x_out)

        x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                       padding="same", activation=LeakyReLU(alpha=ALPHA))(x_out)

        x_out = Conv2D(filters * (2 ** num_blocks), kernel_size, dilation_rate=dilation_rate,
                       padding="same", activation=LeakyReLU(alpha=ALPHA))(x_out)
        x_out = Dropout(dropout)(x_out)
        return Model(inputs=x_in, outputs=x_out, name=f"Encoder2DAtrous{add_name}")

    @staticmethod
    def decoder2d(input_shape, num_output_channels, filters, num_blocks, kernel_size, add_name=""):
        x_in = Input(input_shape)
        for block_ind in range(num_blocks - 1, 0, -1):
            if block_ind == (num_blocks - 1):
                x_out = Conv2DTranspose(filters * (2 ** (block_ind)), kernel_size=kernel_size, strides=2,
                                        padding="same", activation=LeakyReLU(alpha=ALPHA),
                                        kernel_initializer="glorot_normal")(x_in)
            else:
                x_out = Conv2DTranspose(filters * (2 ** (block_ind)), kernel_size=kernel_size, strides=2,
                                        padding="same", activation=LeakyReLU(alpha=ALPHA),
                                        kernel_initializer="glorot_normal")(x_out)

            x_out = Conv2D(filters * (2 ** (block_ind)), kernel_size=kernel_size, padding="same",
                           activation=LeakyReLU(alpha=ALPHA))(
                x_out)

            x_out = Conv2D(filters * (2 ** (block_ind)), kernel_size=kernel_size, padding="same",
                           activation=LeakyReLU(alpha=ALPHA))(
                x_out)

        x_out = Conv2DTranspose(num_output_channels, kernel_size=kernel_size, strides=2, padding="same",
                                activation="linear",
                                kernel_initializer="glorot_normal")(x_out)

        return Model(inputs=x_in, outputs=x_out, name=f"Decoder2D{add_name}")


class PointWiseLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.pointwize_loss = PointWiseLoss.pointwize_loss

    def call(self, y_true, y_pred):
        return self.pointwize_loss

    @staticmethod
    def tf_find_peaks(x):
        """ Finds the maximum value in each channel and returns the location and value.
        Args:
            x: rank-4 tensor (samples, height, width, channels)

        Returns:
            peaks: rank-3 tensor (samples, [x, y, val], channels)
        """

        # Store input shape
        in_shape = tf.shape(x)

        # Flatten height/width dims
        flattened = tf.reshape(x, [in_shape[0], -1, in_shape[-1]])

        # Find peaks in linear indices
        idx = tf.argmax(flattened, axis=1)

        # Convert linear indices to subscripts
        rows = tf.math.floordiv(tf.cast(idx, tf.int32), in_shape[1])
        cols = tf.math.floormod(tf.cast(idx, tf.int32), in_shape[1])

        # Dumb way to get actual values without indexing
        vals = tf.math.reduce_max(flattened, axis=1)

        # Return N x 3 x C tensor
        pred = tf.stack([
            tf.cast(cols, tf.float32),
            tf.cast(rows, tf.float32),
            vals],
            axis=1)
        return pred

    @staticmethod
    def _calculate_heatmap_keypoints(linspace, heatmaps):
        h_y = tf.reduce_sum(linspace * tf.reduce_sum(heatmaps, axis=-2), axis=-2) / tf.reduce_sum(heatmaps,
                                                                                                  axis=[-3, -2])
        h_x = tf.reduce_sum(linspace * tf.reduce_sum(heatmaps, axis=-3), axis=-2) / tf.reduce_sum(heatmaps,
                                                                                                  axis=[-3, -2])

        return tf.transpose(tf.stack([h_x - 1, h_y - 1], axis=1), [0, 2, 1])

    @staticmethod
    def find_peaks(heatmaps):
        # height = heatmaps.shape[1]
        height = 192
        linspace = tf.range(1, height + 1, dtype='float32')
        linspace = tf.reshape(linspace, (height, 1))
        return PointWiseLoss._calculate_heatmap_keypoints(linspace, heatmaps)

    @staticmethod
    def pointwize_loss(y_true, y_pred):
        """
        Args:
            y_true: ground truth confmaps
            y_pred: predicted confmaps
        Returns: loss (float)

        """
        true_peaks = PointWiseLoss.find_peaks(y_true)
        pred_peaks = PointWiseLoss.find_peaks(y_pred)
        return tf.losses.mean_squared_error(true_peaks, pred_peaks)
