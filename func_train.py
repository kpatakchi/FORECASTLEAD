from py_env_train import *
from keras_unet_collection import models

def UNET(n_lat, n_lon, n_channels, ifn, dropout_rate, type_):
    
    import tensorflow as tf

    n_lat = n_lat
    n_lon = n_lon
    n_channels = n_channels  # t-1, t, t+1
    ifn = ifn  # initial feature number (number of initial filters)
    leakyrelu = tf.keras.layers.LeakyReLU()
    dropout_rate=dropout_rate

    if type_ == "unet-att-s":
        
        model = models.att_unet_2d((n_lat, n_lon, n_channels), filter_num=[ifn, ifn*2, ifn*4, ifn*8], n_labels=1,
                                   stack_num_down=2, stack_num_up=2,
                                   activation='ReLU', atten_activation='ReLU', attention='add', output_activation=None, 
                                   batch_norm=True, pool=True, unpool='bilinear', name='attunet')
        return model

    if type_ == "unet-att-l":
        
        model = models.att_unet_2d((n_lat, n_lon, n_channels), filter_num=[ifn, ifn*2, ifn*4, ifn*8], n_labels=1,
                                   stack_num_down=4, stack_num_up=4,
                                   activation='ReLU', atten_activation='ReLU', attention='add', output_activation=None, 
                                   batch_norm=True, pool=True, unpool='bilinear', name='attunet')
        return model

    if type_ == "unet-trans-s":
    
        model = models.transunet_2d((n_lat, n_lon, n_channels), filter_num=[ifn, ifn*2, ifn*4, ifn*8], n_labels=1, stack_num_down=2,
                                    stack_num_up=2,embed_dim=1024, num_mlp=1024, num_heads=6, num_transformer=6,
                                    activation='ReLU', mlp_activation='GELU', output_activation=None, 
                                    batch_norm=True, pool=True, unpool='bilinear', name='transunet')
        return model
        
    if type_ == "unet-trans-l":
    
        model = models.transunet_2d((n_lat, n_lon, n_channels), filter_num=[ifn, ifn*2, ifn*4, ifn*8], n_labels=1, stack_num_down=4,
                                    stack_num_up=4,embed_dim=1024, num_mlp=1024, num_heads=6, num_transformer=6,
                                    activation='ReLU', mlp_activation='GELU', output_activation=None, 
                                    batch_norm=True, pool=True, unpool='bilinear', name='transunet')
        return model
        
    if type_ == "unet-se":

        # squeeze and excitation block
        def se_block(input_tensor, filters):
            se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
            se = tf.keras.layers.Reshape((1, 1, filters))(se)
            se = tf.keras.layers.Conv2D(filters // 8, (1, 1), activation='relu', padding='same')(se)
            se = tf.keras.layers.Conv2D(filters, (1, 1), activation='sigmoid', padding='same')(se)
            return tf.keras.layers.Multiply()([input_tensor, se])
            
        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(inputs)
        c1 = se_block(c1, ifn)  # Add attention block here
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = se_block(c2, ifn * 2)  # Add attention block here
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
        c3 = se_block(c3, ifn * 4)  # Add attention block here
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.BatchNormalization()(p3)
    
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(p3)
        c4 = se_block(c4, ifn * 8)  # Add attention block here
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = tf.keras.layers.BatchNormalization()(p4)
    
        c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(p4)
        c5 = se_block(c5, ifn * 16)  # Add attention block here
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
        c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(ifn * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(u6)
        c6 = se_block(c6, ifn * 8)  # Add attention block here
        c6 = tf.keras.layers.Dropout(dropout_rate)(c6)  # Add dropout layer here
        c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
    
        u7 = tf.keras.layers.Conv2DTranspose(ifn * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(u7)
        c7 = se_block(c7, ifn * 4)  # Add attention block here
        c7 = tf.keras.layers.Dropout(dropout_rate)(c7)  # Add dropout layer here
        c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
    
        u8 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u8)
        c8 = se_block(c8, ifn * 2)  # Add attention block here
        c8 = tf.keras.layers.Dropout(dropout_rate)(c8)  # Add dropout layer here
        c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c8)
        c8 = tf.keras.layers.BatchNormalization()(c8)
    
        u9 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u9)
        c9 = se_block(c9, ifn)  # Add attention block here
        c9 = tf.keras.layers.Dropout(dropout_rate)(c9)  # Add dropout layer here
        c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-l":

        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.BatchNormalization()(p3)
    
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(p3)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = tf.keras.layers.BatchNormalization()(p4)
    
        c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(p4)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
        c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(ifn * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(u6)
        c6 = tf.keras.layers.Dropout(dropout_rate)(c6)  # Add dropout layer here
        c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
    
        u7 = tf.keras.layers.Conv2DTranspose(ifn * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(u7)
        c7 = tf.keras.layers.Dropout(dropout_rate)(c7)  # Add dropout layer here
        c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
    
        u8 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u8)
        c8 = tf.keras.layers.Dropout(dropout_rate)(c8)  # Add dropout layer here
        c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c8)
        c8 = tf.keras.layers.BatchNormalization()(c8)
    
        u9 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u9)
        c9 = tf.keras.layers.Dropout(dropout_rate)(c9)  # Add dropout layer here
        c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-l-dw":

        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.BatchNormalization()(p3)
    
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(p3)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = tf.keras.layers.BatchNormalization()(p4)
    
        c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(p4)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
        c5 = tf.keras.layers.Conv2D(ifn * 16, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        # Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(ifn * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(u6)
        c6 = tf.keras.layers.Dropout(dropout_rate)(c6)  # Add dropout layer here
        c6 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
    
        u7 = tf.keras.layers.Conv2DTranspose(ifn * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(u7)
        c7 = tf.keras.layers.Dropout(dropout_rate)(c7)  # Add dropout layer here
        c7 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
    
        u8 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u8)
        c8 = tf.keras.layers.Dropout(dropout_rate)(c8)  # Add dropout layer here
        c8 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c8)
        c8 = tf.keras.layers.BatchNormalization()(c8)
    
        u9 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u9)
        c9 = tf.keras.layers.Dropout(dropout_rate)(c9)  # Add dropout layer here
        c9 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-m":
        
        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.BatchNormalization()(p3)
    
        # Bottleneck
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(p3)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
    
        # Expansive path
        u5 = tf.keras.layers.Conv2DTranspose(ifn * 4, (3, 3), strides=(2, 2), padding='same')(c4)
        u5 = tf.keras.layers.concatenate([u5, c3])
        c5 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(u5)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
        c5 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        u6 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c2])
        c6 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u6)
        c6 = tf.keras.layers.Dropout(dropout_rate)(c6)  # Add dropout layer here
        c6 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
    
        u7 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c1], axis=3)
        c7 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u7)
        c7 = tf.keras.layers.Dropout(dropout_rate)(c7)  # Add dropout layer here
        c7 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c7)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-m-dw":
        
        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        p3 = tf.keras.layers.BatchNormalization()(p3)
    
        # Bottleneck
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(p3)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
        c4 = tf.keras.layers.Conv2D(ifn * 8, (3, 3), activation=leakyrelu, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
    
        # Expansive path
        u5 = tf.keras.layers.Conv2DTranspose(ifn * 4, (3, 3), strides=(2, 2), padding='same')(c4)
        u5 = tf.keras.layers.concatenate([u5, c3])
        c5 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(u5)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
        c5 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        u6 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c2])
        c6 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u6)
        c6 = tf.keras.layers.Dropout(dropout_rate)(c6)  # Add dropout layer here
        c6 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
    
        u7 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c1], axis=3)
        c7 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u7)
        c7 = tf.keras.layers.Dropout(dropout_rate)(c7)  # Add dropout layer here
        c7 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c7)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-s":

        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        # Bottleneck (removed one level of downsampling)
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)
    
        # Expansive path (adjusted to match the new architecture)
        u4 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c3)
        u4 = tf.keras.layers.concatenate([u4, c2])
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u4)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
    
        u5 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c4)
        u5 = tf.keras.layers.concatenate([u5, c1], axis=3)
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u5)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c5)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-s-dw":

        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)  # Add dropout layer here
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)  # Add dropout layer here
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        # Bottleneck (removed one level of downsampling)
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(p2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)  # Add dropout layer here
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)
    
        # Expansive path (adjusted to match the new architecture)
        u4 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c3)
        u4 = tf.keras.layers.concatenate([u4, c2])
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u4)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)  # Add dropout layer here
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
    
        u5 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c4)
        u5 = tf.keras.layers.concatenate([u5, c1], axis=3)
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(u5)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)  # Add dropout layer here
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c5)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-xs":

        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path (only one level)
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)
        c1 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
    
        # Bottleneck
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)
    
        # Expansive path (only one level)
        u4 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c3)
        u4 = tf.keras.layers.concatenate([u4, c1])
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u4)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c4)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c5)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-xs-dw":

        # Inputs
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path (only one level)
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(dropout_rate)(c1)
        c1 = tf.keras.layers.DepthwiseConv2D((3, 3), depth_multiplier=int(ifn/16), activation=leakyrelu, padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(p1)
        c2 = tf.keras.layers.Dropout(dropout_rate)(c2)
        c2 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c2)
    
        # Bottleneck
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c2)
        c3 = tf.keras.layers.Dropout(dropout_rate)(c3)
        c3 = tf.keras.layers.Conv2D(ifn * 4, (3, 3), activation=leakyrelu, padding='same')(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)
    
        # Expansive path (only one level)
        u4 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c3)
        u4 = tf.keras.layers.concatenate([u4, c1])
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(u4)
        c4 = tf.keras.layers.Dropout(dropout_rate)(c4)
        c4 = tf.keras.layers.Conv2D(ifn * 2, (3, 3), activation=leakyrelu, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c4)
        c5 = tf.keras.layers.Dropout(dropout_rate)(c5)
        c5 = tf.keras.layers.Conv2D(ifn, (3, 3), activation=leakyrelu, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c5)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model

    if type_ == "unet-att-res":

        # Convolutional block
        def conv_block(x, filters, kernel_size, dropout_rate):
            conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.LeakyReLU()(conv)
            conv = tf.keras.layers.Dropout(dropout_rate)(conv)
            conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(conv)
            conv = tf.keras.layers.BatchNormalization()(conv)
            conv = tf.keras.layers.LeakyReLU()(conv)
            return conv
    
        # Residual block
        def res_conv_block(x, filters, kernel_size, dropout_rate):
            conv = conv_block(x, filters, kernel_size, dropout_rate)
            shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            res_path = tf.keras.layers.add([shortcut, conv])
            return res_path
    
        # Gating signal
        def gating_signal(input, filters):
            x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return x
    
        # Attention block
        def attention_block(x, gating, filters):
            theta_x = tf.keras.layers.Conv2D(filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
            phi_g = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(gating)
            upsample_g = tf.keras.layers.Conv2DTranspose(filters, (3, 3), strides=(theta_x.shape[1] // phi_g.shape[1], theta_x.shape[2] // phi_g.shape[2]), padding='same', kernel_initializer='he_normal')(phi_g)
            concat_xg = tf.keras.layers.add([upsample_g, theta_x])
            act_xg = tf.keras.layers.LeakyReLU()(concat_xg)
            psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(act_xg)
            sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
            upsample_psi = tf.keras.layers.UpSampling2D(size=(x.shape[1] // sigmoid_xg.shape[1], x.shape[2] // sigmoid_xg.shape[2]))(sigmoid_xg)
            y = tf.keras.layers.multiply([upsample_psi, x])
            result = tf.keras.layers.Conv2D(x.shape[3], (1, 1), padding='same', kernel_initializer='he_normal')(y)
            result = tf.keras.layers.BatchNormalization()(result)
            return result
    
        inputs = tf.keras.layers.Input((n_lat, n_lon, n_channels))
        #inputs_bn = tf.keras.layers.BatchNormalization()(inputs)
    
        # Contraction path
        c1 = res_conv_block(inputs, ifn, (3, 3), dropout_rate)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.BatchNormalization()(p1)
    
        c2 = res_conv_block(p1, ifn * 2, (3, 3), dropout_rate)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.BatchNormalization()(p2)
    
        # Bottleneck
        c3 = res_conv_block(p2, ifn * 4, (3, 3), dropout_rate)
        c3 = tf.keras.layers.BatchNormalization()(c3)
    
        # Expansive path with attention
        g2 = gating_signal(c3, ifn * 2)
        a2 = attention_block(c2, g2, ifn * 2)
        u2 = tf.keras.layers.Conv2DTranspose(ifn * 2, (3, 3), strides=(2, 2), padding='same')(c3)
        u2 = tf.keras.layers.concatenate([u2, a2])
        c4 = res_conv_block(u2, ifn * 2, (3, 3), dropout_rate)
        c4 = tf.keras.layers.BatchNormalization()(c4)
    
        g1 = gating_signal(c4, ifn)
        a1 = attention_block(c1, g1, ifn)
        u1 = tf.keras.layers.Conv2DTranspose(ifn, (3, 3), strides=(2, 2), padding='same')(c4)
        u1 = tf.keras.layers.concatenate([u1, a1], axis=3)
        c5 = res_conv_block(u1, ifn, (3, 3), dropout_rate)
        c5 = tf.keras.layers.BatchNormalization()(c5)
    
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c5)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
        return model


def make_canvas(data, canvas_shape, trim=True):
    """
    Pads the input data with zeros to create a canvas of the specified shape
    while keeping the original data centered or trimmed to fit the canvas. 

    Args:
    - data (numpy.ndarray): Input data with shape (num_samples, original_dim1, original_dim2, channels)
    - canvas_shape (tuple): Desired shape of the canvas in the format (canvas_dim1, canvas_dim2)
    - trim (bool): If True, trims the original data to fit the canvas instead of making a bigger canvas.

    Returns:
    - numpy.ndarray: Canvas with shape (num_samples, canvas_dim1, canvas_dim2, channels)
    """
    num_samples, original_dim1, original_dim2, channels = data.shape
    canvas_dim1, canvas_dim2 = canvas_shape

    # If trim is True and the original data is larger than the canvas, trim the data
    if trim and (original_dim1 > canvas_dim1 or original_dim2 > canvas_dim2):
        start_idx1 = (original_dim1 - canvas_dim1) // 2
        start_idx2 = (original_dim2 - canvas_dim2) // 2
        data = data[:, start_idx1:start_idx1+canvas_dim1, start_idx2:start_idx2+canvas_dim2, :]

    # If trim is False or the original data is smaller than the canvas, pad the data
    else:
        # Calculate the difference between the original dimensions and the canvas dimensions
        diff_dim1 = canvas_dim1 - original_dim1
        diff_dim2 = canvas_dim2 - original_dim2

        # Calculate the padding for the top and left sides
        top_pad = diff_dim1 // 2
        left_pad = diff_dim2 // 2

        # Calculate the padding for the bottom and right sides
        bottom_pad = diff_dim1 - top_pad
        right_pad = diff_dim2 - left_pad

        # Create a new array of zeros with the shape of the canvas
        canvas = np.zeros((num_samples, canvas_dim1, canvas_dim2, channels))

        # Insert the original data in the center of the canvas
        canvas[:, top_pad:top_pad+original_dim1, left_pad:left_pad+original_dim2, :] = data
        return canvas

def create_calendar (date_start, date_end, MODEL_shape, daily=True):
    
    """
    Creates a numpy array in the same shape as MODEL (model-based data) representing a calendar.
    CAL[..., 0] contains the day of the year for each day inside a year.
    CAL[..., 1] contains the year number.
    
    Args:
    - date_start (str): Start date of the calendar in YYYY-MM-DD format.
    - date_end (str): End date of the calendar in YYYY-MM-DD format.
    - daily (bool): Whether the data is daily or hourly. If hourly, each day will contain 24 hours.
    
    Returns:
    - CAL (np.ndarray): Numpy array of shape (n_days, LAT, LON, 2) representing the calendar.
    """

    CAL = np.ones((MODEL_shape[0], MODEL_shape[1], MODEL_shape[2], 2), dtype="int32")    
    y_start = int(date_start[:4])
    y_end = int(date_end[:4])
    days_in_year = 365
    hours_in_day = 24

    if daily:
        for year in range(y_start, y_end+1):
            for day in range(days_in_year):
                dayloc=(year-y_start)*days_in_year+day-1
                CAL[dayloc, ..., 0] = year
                CAL[dayloc, ..., 1] = day + 1
    else: 
        for year in range(y_start, y_end+1):
            for day in range(days_in_year):
                for hour in range (hours_in_day):
                    hourlock=(year-y_start)*days_in_year*hours_in_day+day*hours_in_day-1
                    CAL[hourlock, ..., 0] = year
                    CAL[hourlock, ..., 1] = day + 1
                    
    return CAL

def spatiodataloader (directory, MODEL_shape):
    
    """
    Loads spatial data from a NumPy file and reshapes it to the specified dimensions.

    Parameters:
        directory (str): Path and filename of the NumPy binary file containing the spatial data.
        MODEL_shape (tuple): Desired dimensions of the output array. A 3-tuple specifying the height, width, and number of channels (e.g., color channels) of the output array.

    Returns:
        A NumPy array of shape MODEL_shape that contains the loaded spatial data.

    """
    SP = np.load(directory)["arr_0"]
    SP_reshaped=np.empty((MODEL_shape[0], MODEL_shape[1], MODEL_shape[2], SP.shape[2]))
    
    for x in range (MODEL_shape[1]):
        for y in range (MODEL_shape[2]):
            SP_reshaped[:, x, y, 0] = SP[x, y, 0]
            SP_reshaped[:, x, y, 1] = SP[x, y, 1]
            SP_reshaped[:, x, y, 2] = SP[x, y, 2]

    return SP_reshaped


def calculate_channels(n_ensembles, task_name, laginensemble=False):
    """
    Calculate the number of channels based on the given task name and number of ensembles.

    Args:
    - n_ensembles: int, number of ensembles.
    - task_name: str, name of the task.
    - laginensemble: bool, optional, whether the ensembles are lagged or not. Default is False.

    Returns:
    - n_channels: int, number of channels.

    Raises:
    - ValueError: if an invalid task name is given.

    Task Name	Explanation	Name	N_channels
    A1	Model-only for t=t	“model-only”	N_ensembles
    A2	Model-only for t=t and t=t-1	“model-lag”	N_ensembles*2
    A3	Same as A2, including the day of the year and the year information. 	“temporal”	N_ensembles+2 (if laginensemble=False), N_ensembles*2 (if laginensemble=True)
    A4	Same as A3, but also including the lat/lon/altitude information.	“spatiotemporal”	N_ensembles+5 (if laginensemble=False), N_ensembles*2+5 (if laginensemble=True)
    A5	Same as A2, but also including the lat/lon/altitude iformation	“spatial”	N_ensembles+3 (if laginensemble=False), N_ensembles*2+3 (if laginensemble=True)
    """
    if task_name == "model_only":
        n_channels = n_ensembles
    elif task_name == "model-lag":
        n_channels = n_ensembles * 2
    elif task_name == "temporal":
        n_channels = n_ensembles + 2 if not laginensemble else n_ensembles * 2
    elif task_name == "spatiotemporal":
        n_channels = n_ensembles + 5 if not laginensemble else n_ensembles * 2 + 5
    elif task_name == "spatial":
        n_channels = n_ensembles + 3 if not laginensemble else n_ensembles * 2 + 3
    else:
        raise ValueError("Invalid task name: " + task_name)
        
    return n_channels

def data_unique_name_generator(model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble):
    # The following is defined automatically:
    n_ensembles = len(model_data)
    n_channels = calculate_channels(n_ensembles, task_name, laginensemble=laginensemble)
    
    if reference_data == ["COSMO_REA6"]: #publication with TSMP-G2A & COSMO-REA6
        canvas_size = (400, 400) 
        topo_dir = PPROJECT_DIR + '/IO/03-TOPOGRAPHY/EU-11-TOPO.npz'
        trim = True
        daily = True
        
    elif reference_data == ["HSAF"]: #publication with H-SAF & H-RES
        topo_dir = PPROJECT_DIR + '/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
        canvas_size = (128, 256)
        trim = False
        daily = False
        
    elif reference_data == ["ADAPTER_DE05.day01.merged.nc"]:  #publication with HRES lead times
        topo_dir = PPROJECT_DIR + '/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
        canvas_size = (128, 256)
        trim = False
        daily = False
        
    else:
        topo_dir = ''  # Set the default value if reference_data is not recognized
        canvas_size = (0, 0)
        trim = False
        daily = False
        
    
    data_unique_name = f"train_data{'_daily' if daily else '_hourly'}_{variable}_{model_data}_{reference_data}_{mm}_{n_channels}_{'laginensemble' if laginensemble else ''}_{task_name}_{'.'.join(map(str, canvas_size))}_{date_start}_{date_end}_{mask_type}"
    filename = f"{data_unique_name}.npz"
    return filename

def generate_training_unique_name(loss, Filters, LR, min_LR, lr_factor, lr_patience, BS, patience, val_split, epochs, dropout, unet_type, leadtime):
    training_unique_name = loss + "_" + str(Filters) + "_" + str(LR) + "_" + str(min_LR) + "_" + str(lr_factor) + "_" + str(lr_patience) + "_" + str(BS) + "_" + str(patience) + "_" + str(val_split) + "_" + str(epochs) + "_" + dropout + "_" + str(unet_type) +  "_" + leadtime 
    return training_unique_name

def HRES_NETCDF_LEADTIME_TRAIN_PREPROCESS(dataset, variable, leadtime):
    """
    Process the dataset based on the leadtime requirements.
    
    Parameters:
    dataset (xr.Dataset): The input dataset.
    variable (str): The variable name in the dataset to process.
    leadtime (str): The leadtime value ('day02', 'day03', 'day04', 'day05', 'day06', 'day07', 'day08', 'day09', 'day10').

    Day02 and Day03: No changes are made to the dataset.
    Day04: Applies the specific preprocessing by zeroing out hours 07, 08, 10, and 11, and summing the values for hours 09 and 12.
    Day05 and Day06: Resamples the data to 3-hourly intervals.
    Day07 to Day10: Resamples the data to 6-hourly intervals
    """
    if variable != "novar":
        var_data = dataset[variable]
    else:
        var_data = dataset*1
        
    if leadtime == "day04":

        # Assuming the time steps are hourly and formatted as 'YYYY-MM-DDTHH:00:00.000000000'
        unique_days = np.unique(var_data.time.dt.floor('D').values)
        
        for day in unique_days:

            # Mask for the specific day
            mask_day = (var_data.time.dt.floor('D') == day)
            
            # Select the indices for the specified hours
            indices_T07 = mask_day & (var_data.time.dt.hour == 7)
            indices_T08 = mask_day & (var_data.time.dt.hour == 8)
            indices_T09 = mask_day & (var_data.time.dt.hour == 9)
            indices_T10 = mask_day & (var_data.time.dt.hour == 10)
            indices_T11 = mask_day & (var_data.time.dt.hour == 11)
            indices_T12 = mask_day & (var_data.time.dt.hour == 12)
                
            if np.sum(indices_T07.values) != 0:
                # Replace T09 with the sum of values in T07, T08, and T09 for the specified variable
                sum_T09 = var_data[indices_T07].values + var_data[indices_T08].values + var_data[indices_T09].values
                var_data[indices_T09] = sum_T09
    
                # Replace T12 with the sum of values in T10, T11, and T12 for the specified variable
                sum_T12 = var_data[indices_T10].values + var_data[indices_T11].values + var_data[indices_T12].values
                var_data[indices_T12] = sum_T12

                # Drop the time steps T07, T08, T10, and T11
                drop_times = var_data.time[indices_T07 | indices_T08 | indices_T10 | indices_T11]
                var_data = var_data.drop_sel(time=drop_times)
            else:
                print("indices_T07 doesn't exist for" + str(day))

    elif leadtime in ["day05", "day06"]:
        # Resample to 3-hourly data
        var_data = var_data.resample(time='3H').sum()

    elif leadtime in ["day07", "day08", "day09", "day10"]:
        # Resample to 6-hourly data
        var_data = var_data.resample(time='6H').sum()

    return var_data

def prepare_train(PPROJECT_DIR, TRAIN_FILES, ATMOS_DATA, filename, model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble, val_split, leadtime):
    
    """
    This function prepares the training data for UNET model.
    
    Args:
        PPROJECT_DIR (str): The project directory path.
        TRAIN_FILES (str): The directory where the training files will be saved.
        ATMOS_DATA (str): The directory containing the atmospheric data.
        filename (str): The name of the file to be saved.
        model_data (list): A list of model names.
        reference_data (list): A list of reference data names.
        task_name (str): The type of task for the model.
        mm (str): The type of target (mismatch or direct).
        date_start (str): The start date for selecting the data.
        date_end (str): The end date for selecting the data.
        variable (str): The variable to be used in the data.
        mask_type (str): The type of mask to be applied.
        laginensemble (int): The lag in the ensemble dimension.
        val_split (float): The validation data split ratio.
    """
        
    import os
    import xarray as xr
    
    data_unique_name=filename[:-4]
    
    if filename not in os.listdir(TRAIN_FILES):
        print("Training data isn't already available; creating it ...")
        print("Opening Netcdf files ...")

        # 1) Open the datasets:
        datasets = []
        
        for model in model_data:
            
            dataset = xr.open_dataset(f"{ATMOS_DATA}/{model}")
            dataset = dataset[variable].sel(time=slice(date_start, date_end))
            if variable == "pr":
                dataset = dataset.where(dataset > 0, 0)
            datasets.append(dataset)

        REFERENCE = xr.open_dataset(f"{ATMOS_DATA}/{reference_data[0]}")
        REFERENCE = REFERENCE[variable].sel(time=slice(date_start, date_end))
        if variable == "pr":
            REFERENCE = REFERENCE.where(REFERENCE > 0, 0)
        
        REFERENCE = HRES_NETCDF_LEADTIME_TRAIN_PREPROCESS(REFERENCE, "novar", leadtime)
                                   
        # Align all datasets with the reference dataset
        datasets_aligned = []
        for dataset in datasets:
            dataset_aligned, REFERENCE_aligned = xr.align(dataset, REFERENCE, join='inner')
            datasets_aligned.append(dataset_aligned)
        
        # Update datasets to use aligned datasets
        datasets = datasets_aligned
        REFERENCE = REFERENCE_aligned
                                    
        # 2) Calculate the calendar data according to REFERENCE (starting the calendar one day later)
        dayofyear = REFERENCE[1:, ...].time.dt.dayofyear.values
        dayofyear_resh = np.tile(dayofyear[:, np.newaxis, np.newaxis], (1, REFERENCE[1:, ...].shape[1], REFERENCE[1:, ...].shape[2]))
        yeardate = REFERENCE[1:, ...].time.dt.year.values
        yeardate_resh = np.tile(yeardate[:, np.newaxis, np.newaxis], (1, REFERENCE[1:, ...].shape[1], REFERENCE[1:, ...].shape[2]))
        CAL = np.stack((dayofyear_resh, yeardate_resh), axis=3).astype(float)

        REFERENCE = REFERENCE.values[:, :, :, np.newaxis]  # add new axis along ensemble dimension
        datasets = [dataset.values for dataset in datasets]
        MODEL = np.stack(datasets, axis=-1)

        # 2) Define the Target (mismatch or direct):
        print("Calculating the target (mismatch) ...")
        if len(datasets) > 1:
            TARGET = (MODEL[0] - REFERENCE) if (mm == "MM") else REFERENCE
        else:
            TARGET = (MODEL - REFERENCE) if (mm == "MM") else REFERENCE
        if MODEL.shape[0] < 1:
            print("The selected dates don't exist in the netcdf files!")

        if reference_data == ["COSMO_REA6"]:
            canvas_size = (400, 400) 
            topo_dir=PPROJECT_DIR+'/IO/03-TOPOGRAPHY/EU-11-TOPO.npz'
            trim=True
            daily=True
        if reference_data == ["HSAF"]:
            topo_dir=PPROJECT_DIR+'/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
            canvas_size = (128, 256)
            trim=False
            daily=False                  
        if reference_data == ["ADAPTER_DE05.day01.merged.nc"]:  #publication with HRES lead times
            topo_dir = PPROJECT_DIR + '/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
            canvas_size = (128, 256)
            trim = False
            daily = False

        # Close the netCDF files and release memory
        datasets = None
        REFERENCE = None
        dayofyear = None
        yeardate = None
        yeardate_resh = None

        # 3) Define X_Train and Y_Train
        print("Defining X_Train and Y_Train...")
                                    
        Y_TRAIN = TARGET[1:, ...]  # t
        X_TRAIN = MODEL[1:, ...]  # t
        X_TRAIN_tminus = MODEL[:-1, ...]
        
        canvas_y = make_canvas(Y_TRAIN, canvas_size, trim)
        canvas_y = np.nan_to_num(canvas_y, nan=-999)  # fill values
        SPP = spatiodataloader(topo_dir, X_TRAIN.shape)
        
        if mask_type == "no_na":
            canvas_m = np.ones_like(canvas_y)  
            canvas_m[canvas_y == -999] = 0
            
        if mask_type == "no_na_land":
            canvas_m = np.ones_like(canvas_y)  
            canvas_m[canvas_y == -999] = 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land
            # to remove the zeros out of the boundaries:
            #outbound = np.nanmean(canvas_x[:, ..., 0], axis=0) < 0.00001
            #for i in range(canvas_m.shape[0]):
            #    canvas_m[i, outbound, 0] = 0
                
        if mask_type == "no_na_intensity":
            
            TRUTH = Y_TRAIN - X_TRAIN #reference(to be used in intensity weights)
            canvas_t = make_canvas(TRUTH, canvas_size, trim)
            greater_zero = canvas_t[..., 0]>=0
            less_pointone = canvas_t[..., 0]<0.1
            greater_pointone = canvas_t[..., 0]>=0.1
            less_twohalf = canvas_t[..., 0]<2.5
            greater_twohalf = canvas_t[..., 0]>=2.5
            dry=greater_zero*less_pointone
            light=greater_pointone*less_twohalf
            heavy=greater_twohalf

            canvas_m = np.ones_like(canvas_y) 
            canvas_m[canvas_y == -999] = 0 #replace nan with 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land #only on land
            
            # to remove the ones out of the boundaries:
            outbound = np.nanmean(canvas_m[:, ..., 0], axis=0) > 0.999
            for i in range(canvas_m.shape[0]):
                canvas_m[i, outbound, 0] = 0.
                
            canvas_m[dry] *= 0.01  # Multiply by 0.01 in dry conditions
            canvas_m[light] *= 0.04  # Multiply by 0.04 in light conditions
            canvas_m[heavy] *= 0.95  # Multiply by 0.95 in heavy conditions

        # rescaling (standardization):
        x_min = np.nanmin(X_TRAIN)
        x_max = np.nanmax(X_TRAIN)

        X_TRAIN = (X_TRAIN - x_min) / (x_max - x_min)
        X_TRAIN_tminus = (X_TRAIN_tminus - x_min) / (x_max - x_min)

        # Rescale day of the year (CAL[..., 0])
        cal_min = np.nanmin(CAL[..., 0])
        cal_max = np.nanmax(CAL[..., 0])
        CAL[..., 0] = (CAL[..., 0] - cal_min) / (cal_max - cal_min)
        # Set yeardate (CAL[..., 1]) to 0 if you don't want to consider it
        CAL[..., 1] = 0
            
        for ch in range (0, 3):
            spp_min = np.nanmin(SPP[..., ch])
            spp_max = np.nanmax(SPP[..., ch])
            SPP[..., ch] = (SPP[..., ch] - spp_min) / (spp_max - spp_min) 
        
        if task_name == "model_only":
            X_TRAIN = X_TRAIN

        if task_name == "model-lag":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN), axis=3)

        if task_name == "temporal":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, CAL), axis=3)

        if task_name == "spatial":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, SPP), axis=3)

        if task_name == "spatiotemporal":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, CAL, SPP), axis=3)

        canvas_x = make_canvas(X_TRAIN, canvas_size, trim)
            
        X_TRAIN_tminus = None
        CAL = None
        SPP = None

        # Save train and validation data
        print("Saving train and validation data...")
                                    
        np.random.seed(hash(data_unique_name) % 2**32 - 1)
        num_samples = canvas_x.shape[0]
        indices = np.arange(num_samples)  # Create an array of indices

        train_prop = 1 - val_split
        num_train_samples = int(np.round(num_samples * train_prop))

        # Create clusters of 10 consecutive numbers for validation
        cluster_size = 10
        num_clusters = num_samples // cluster_size
        num_val_clusters = int(np.ceil(num_clusters * val_split))
        val_clusters = np.random.choice(num_clusters, size=num_val_clusters, replace=False)

        val_indices = []
        for cluster in val_clusters:
            start_index = cluster * cluster_size
            end_index = start_index + cluster_size
            val_indices.extend(list(range(start_index, end_index)))

        val_indices = np.sort(np.array(val_indices))
        train_indices = np.setdiff1d(indices, val_indices)

        train_x = canvas_x[train_indices].astype(np.float16)
        train_y = canvas_y[train_indices].astype(np.float16)
        train_m = canvas_m[train_indices].astype(np.float16)
        val_x = canvas_x[val_indices].astype(np.float16)
        val_y = canvas_y[val_indices].astype(np.float16)
        val_m = canvas_m[val_indices].astype(np.float16)

        canvas_y = None
        canvas_x = None
        canvas_m = None

        # rescaling the output (standardization)
        y_min = np.nanmin(train_y)
        y_max = np.nanmax(train_y)

        train_y = 2 * (train_y - y_min) / (y_max - y_min) - 1
        val_y =  2 * (val_y - y_min) / (y_max - y_min) - 1
        
        # Save min and max values to a CSV file
        import csv
        
        csv_file = f"{PPROJECT_DIR2}/CODES-MS3/FORECASTLEAD/scaling_info.csv"
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:  # If the file doesn't exist, write the header
                writer.writerow(["leadtime", "y_min", "y_max", "x_min", "x_max"])
            writer.writerow([leadtime, y_min, y_max, x_min, x_max])

        # Save as float16 files
        np.savez(TRAIN_FILES + "/" + filename,
                 train_x=train_x,
                 train_y=train_y,
                 train_m=train_m,
                 val_x=val_x,
                 val_y=val_y,
                 val_m=val_m)
        
        train_x = None
        train_y = None
        train_m = None
        val_x = None
        val_y = None
        val_m = None

        #np.save(PPROJECT_DIR+'/AI MODELS/00-UNET/'+data_unique_name+"_train_indices.npy", train_indices)
        #np.save(PPROJECT_DIR+'/AI MODELS/00-UNET/'+data_unique_name+"_val_indices.npy", val_indices)

        print("Data generated")
    else:  
        print("Data is available already")
        

def generate_produce_unique_name(loss, Filters, LR, min_LR, lr_factor, lr_patience, BS, patience, val_split, epochs):
    training_unique_name = loss + "_" + str(Filters) + "_" + str(LR) + "_" + str(min_LR) + "_" + str(lr_factor) + "_" + str(lr_patience) + "_" + str(BS) + "_" + str(patience) + "_" + str(val_split) + "_" + str(epochs)
    return training_unique_name

def prepare_produce(PPROJECT_DIR, PRODUCE_FILES, ATMOS_DATA, filename, model_data, reference_data, task_name, mm, date_start, date_end, variable, mask_type, laginensemble, leadtime, y_min, y_max, x_min, x_max):
    
    """
    This function prepares the production data for UNET model.
    
    Args:
        PPROJECT_DIR (str): The project directory path.
        PRODUCE_FILES (str): The directory where the production files will be saved.
        ATMOS_DATA (str): The directory containing the atmospheric data.
        filename (str): The name of the file to be saved.
        model_data (list): A list of model names.
        reference_data (list): A list of reference data names.
        task_name (str): The type of task for the model.
        mm (str): The type of target (mismatch or direct).
        date_start (str): The start date for selecting the data.
        date_end (str): The end date for selecting the data.
        variable (str): The variable to be used in the data.
        mask_type (str): The type of mask to be applied.
        laginensemble (int): The lag in the ensemble dimension.
        val_split (float): The validation data split ratio.
    """
        
    import os
    import xarray as xr
    
    data_unique_name=filename[:-4]
    file_name_pro = "produce_for_" + filename
    
    if file_name_pro not in os.listdir(PRODUCE_FILES):
        
        # 1) Open the datasets:
        datasets = []
        for model in model_data:
            dataset = xr.open_dataset(f"{ATMOS_DATA}/{model}")
            dataset = dataset[variable].sel(time=slice(date_start, date_end))
            if variable == "pr":
                dataset = dataset.where(dataset > 0, 0)
            datasets.append(dataset)

        REFERENCE = xr.open_dataset(f"{ATMOS_DATA}/{reference_data[0]}")
        REFERENCE = REFERENCE[variable].sel(time=slice(date_start, date_end))
        
        if variable == "pr":
            REFERENCE = REFERENCE.where(REFERENCE > 0, 0)
            
        REFERENCE = HRES_NETCDF_LEADTIME_TRAIN_PREPROCESS(REFERENCE, "novar", leadtime)

        # Align all datasets with the reference dataset
        datasets_aligned = []
        for dataset in datasets:
            dataset_aligned, REFERENCE_aligned = xr.align(dataset, REFERENCE, join='inner')
            datasets_aligned.append(dataset_aligned)
            
        # Update datasets to use aligned datasets
        datasets = datasets_aligned
        REFERENCE = REFERENCE_aligned

        # 2) Calculate the calendar data according to REFERENCE (starting the calendar one day later)
        dayofyear = REFERENCE[1:, ...].time.dt.dayofyear.values
        dayofyear_resh = np.tile(dayofyear[:, np.newaxis, np.newaxis], (1, REFERENCE[1:, ...].shape[1], REFERENCE[1:, ...].shape[2]))
        yeardate = REFERENCE[1:, ...].time.dt.year.values
        yeardate = yeardate*0 # add if you do not want to consider year information in inputs
        yeardate_resh = np.tile(yeardate[:, np.newaxis, np.newaxis], (1, REFERENCE[1:, ...].shape[1], REFERENCE[1:, ...].shape[2]))
        CAL = np.stack((dayofyear_resh, yeardate_resh), axis=3).astype(float)

        REFERENCE = REFERENCE.values[:, :, :, np.newaxis]  # add new axis along ensemble dimension
        datasets = [dataset.values for dataset in datasets]
        MODEL = np.stack(datasets, axis=-1)

        # 2) Define the Target (mismatch or direct):
        print("Defining the target...")
        if len(datasets) > 1:
            TARGET = (MODEL[0] - REFERENCE) if (mm == "MM") else REFERENCE
        else:
            TARGET = (MODEL - REFERENCE) if (mm == "MM") else REFERENCE
        if MODEL.shape[0] < 1:
            print("The selected dates don't exist in the netcdf files!")

        if reference_data == ["COSMO_REA6"]:
            canvas_size = (400, 400) 
            topo_dir=PPROJECT_DIR+'/IO/03-TOPOGRAPHY/EU-11-TOPO.npz'
            trim=True
            daily=True
        if reference_data == ["HSAF"]:
            topo_dir=PPROJECT_DIR+'/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
            canvas_size = (128, 256)
            trim=False
            daily=False
        if reference_data == ["ADAPTER_DE05.day01.merged.nc"]:  #publication with HRES lead times
            topo_dir = PPROJECT_DIR + '/IO/03-TOPOGRAPHY/HSAF-TOPO.npz'
            canvas_size = (128, 256)
            trim = False
            daily = False

        # Close the netCDF files and release memory
        datasets = None
        REFERENCE = None
        dayofyear = None
        yeardate = None
        yeardate_resh = None
        
        # 3) Define X_Train and Y_Train
        print("Defining X_Train and Y_Train...")

        Y_TRAIN = TARGET[1:, ...]  # t
        X_TRAIN = MODEL[1:, ...]  # t
        X_TRAIN_tminus = MODEL[:-1, ...]
        canvas_y = make_canvas(Y_TRAIN, canvas_size, trim)
        canvas_y = np.nan_to_num(canvas_y, nan=-999)  # fill values
        SPP = spatiodataloader(topo_dir, X_TRAIN.shape)
        
        if mask_type == "no_na":
            canvas_m = np.ones_like(canvas_y)  
            canvas_m[canvas_y == -999] = 0
            

        if mask_type == "no_na_land":
            canvas_m = np.ones_like(canvas_y)  # mask for na values (-9999)
            canvas_m[canvas_y == -999] = 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land
            # to remove the zeros out of the boundaries:
            #outbound = np.nanmean(canvas_x[:, ..., 0], axis=0) < 0.00001
            #for i in range(canvas_m.shape[0]):
            #    canvas_m[i, outbound, 0] = 0
                
        if mask_type == "no_na_intensity":
            TRUTH = Y_TRAIN - X_TRAIN #reference(to be used in intensity weights)
            canvas_t = make_canvas(TRUTH, canvas_size, trim)
            greater_zero = canvas_t[..., 0]>=0
            less_pointone = canvas_t[..., 0]<0.1
            greater_pointone = canvas_t[..., 0]>=0.1
            less_twohalf = canvas_t[..., 0]<2.5
            greater_twohalf = canvas_t[..., 0]>=2.5
            dry=greater_zero*less_pointone
            light=greater_pointone*less_twohalf
            heavy=greater_twohalf

            canvas_m = np.ones_like(canvas_y) 
            canvas_m[canvas_y == -999] = 0 #replace nan with 0
            SPP_canvas = make_canvas(SPP, canvas_size, trim)
            land=SPP_canvas[..., 2]>0*1
            canvas_m[..., 0] = canvas_m[..., 0]*land #only on land
            
            # to remove the ones out of the boundaries:
            outbound = np.nanmean(canvas_m[:, ..., 0], axis=0) > 0.999
            for i in range(canvas_m.shape[0]):
                canvas_m[i, outbound, 0] = 0.
                
            canvas_m[dry] *= 0.01  # Multiply by 0.01 in dry conditions
            canvas_m[light] *= 0.04  # Multiply by 0.04 in light conditions
            canvas_m[heavy] *= 0.95  # Multiply by 0.95 in heavy conditions

        X_TRAIN = (X_TRAIN - x_min) / (x_max - x_min)
        X_TRAIN_tminus = (X_TRAIN_tminus - x_min) / (x_max - x_min)

        # Rescale day of the year (CAL[..., 0])
        cal_min = np.nanmin(CAL[..., 0])
        cal_max = np.nanmax(CAL[..., 0])
        CAL[..., 0] = (CAL[..., 0] - cal_min) / (cal_max - cal_min)
        
        # Set yeardate (CAL[..., 1]) to 0 if you don't want to consider it
        CAL[..., 1] = 0

        for ch in range (0, 3):
            spp_min = np.nanmin(SPP[..., ch])
            spp_max = np.nanmax(SPP[..., ch])
            SPP[..., ch] = (SPP[..., ch] - spp_min) / (spp_max - spp_min)

        if task_name == "model_only":
            X_TRAIN = X_TRAIN

        if task_name == "model-lag":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN), axis=3)

        if task_name == "temporal":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, CAL), axis=3)

        if task_name == "spatial":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, SPP), axis=3)

        if task_name == "spatiotemporal":
            X_TRAIN = np.concatenate((X_TRAIN_tminus, X_TRAIN, CAL, SPP), axis=3)

        canvas_x = make_canvas(X_TRAIN, canvas_size, trim)
        
        canvas_x = canvas_x.astype(np.float16)
        canvas_y = canvas_y.astype(np.float16)
        canvas_m = canvas_m.astype(np.float16)

        canvas_y = 2 * (canvas_y - y_min) / (y_max - y_min) - 1
        
        X_TRAIN_tminus = None
        CAL = None
        SPP = None
        
        np.savez(PRODUCE_FILES + "/produce_for_" + filename, canvas_x=canvas_x, canvas_y=canvas_y, canvas_m=canvas_m)

        print("Data generated")
    else:
        print("The data with the same unique name is already available")
        

def unmake_canvas(canvas, original_shape):
    """
    Restores the original data from a canvas by removing zero-padding and optionally trimming.

    Args:
    - canvas (numpy.ndarray): Input canvas data with shape (num_samples, canvas_dim1, canvas_dim2, channels)
    - original_shape (tuple): Original shape of the data in the format (original_dim1, original_dim2)
    Returns:
    - numpy.ndarray: Restored data with shape (num_samples, original_dim1, original_dim2)
    """
    num_samples, canvas_dim1, canvas_dim2 = canvas.shape
    original_dim1, original_dim2 = original_shape
    
    diff_dim1 = canvas_dim1 - original_dim1
    diff_dim2 = canvas_dim2 - original_dim2
    
    # Calculate the padding for the top and left sides
    top_pad = diff_dim1 // 2
    left_pad = diff_dim2 // 2

    # Calculate the padding for the bottom and right sides
    bottom_pad = diff_dim1 - top_pad
    right_pad = diff_dim2 - left_pad
    
    # Remove zero-padding from the canvas to restore the original data
    data = canvas[:, top_pad:top_pad+original_dim1, left_pad:left_pad+original_dim2]
    return data

def de_prepare_produce(Y_PRED, PREDICT_FILES, ATMOS_DATA, filename, model_data, date_start, date_end, variable, training_unique_name, reference_data, leadtime, y_min, y_max):
        
    import xarray as xr
    import pandas as pd

    REFERENCE = xr.open_dataset(f"{ATMOS_DATA}/{reference_data}")
    REFERENCE = REFERENCE[variable].sel(time=slice(date_start, date_end))

    if variable == "pr":
        REFERENCE = REFERENCE.where(REFERENCE > 0, 0)
                
    REFERENCE = HRES_NETCDF_LEADTIME_TRAIN_PREPROCESS(REFERENCE, "novar", leadtime)
    
    model = xr.open_dataset(f"{ATMOS_DATA}/{model_data}")
    model = model[variable].sel(time=slice(date_start, date_end))

    if variable == "pr":
        model = model.where(model > 0, 0)

    model_aligned, REFERENCE_aligned = xr.align(model, REFERENCE, join='inner')

    # Retrieve lat and lon shape from the model
    lat_shape = model_aligned.latitude.shape[0]
    lon_shape = model_aligned.longitude.shape[0]
    
    # Restore the original shape of Y_PRED using unmake_canvas function
    Y_PRED = unmake_canvas(Y_PRED, (lat_shape, lon_shape))
    Y_PRED = (Y_PRED + 1) / 2 * (y_max - y_min) + y_min # rescale back to original format

    # Subtract Y_PRED from model
    diff = model_aligned[1:, ...] - Y_PRED
    diff_clipped = np.clip(diff, 0, None) # so that there is no less than zero precip generated!
    
    # Save the result in a NETCDF file
    data_unique_name = filename[:-4]
    output_filename = f"{PREDICT_FILES}/{model_data}.corrected.nc"
    diff_clipped.to_netcdf(output_filename)


def get_scaling_params(scaling_file, PPROJECT_DIR2, leadtime):
    import csv
    with open(scaling_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['leadtime'] == leadtime:
                y_min = float(row['y_min'])
                y_max = float(row['y_max'])
                x_min = float(row['x_min'])
                x_max = float(row['x_max'])
                
                return y_min, y_max, x_min, x_max
