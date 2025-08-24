# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tempfile
import os

import tensorflow as tf
import numpy as np
import datetime

from tensorflow import keras

def float_to_ap_fixed_16_6(float_val):
    """
    Simulates the conversion of a floating-point number (or array of numbers)
    to its ap_fixed<16,6> fixed-point representation.

    ap_fixed<16,6> format:
    - 16 total bits
    - 6 integer bits (including the sign bit)
    - 10 fractional bits (16 - 6)
    This corresponds to a signed Q5.10 format (1 sign bit, 5 integer magnitude bits, 10 fractional bits).

    Scaling factor = 2^10 = 1024.

    The range for the floating-point input that can be represented is approx [-32.0, 31.999...].
    Values outside this range will be saturated (clamped) to the min/max representable values.

    Args:
        float_val (float or np.ndarray): The floating-point number(s) to convert.

    Returns:
        np.int16 or np.ndarray of np.int16: The fixed-point representation as a 16-bit
                                            signed integer. This integer represents the
                                            scaled value.
    """
    total_bits = 16
    integer_bits = 6
    fractional_bits = total_bits - integer_bits # 10

    scale = 2**fractional_bits # 2^10 = 1024

    # The maximum and minimum representable integer values for a signed 16-bit number
    # This corresponds to the range of the scaled value.
    max_scaled_val = (2**(total_bits - 1)) - 1 # 32767
    min_scaled_val = -(2**(total_bits - 1))     # -32768

    # Ensure input is a NumPy array for consistent handling of scalars or lists
    float_array = np.array(float_val, dtype=np.float64) # Use float64 for intermediate precision

    # 1. Multiply by the scale factor to shift the fractional part into integer range
    scaled_value = float_array * scale

    # 2. Round to the nearest integer.
    # HLS fixed-point usually uses round-half-to-even or round-to-nearest-positive/negative.
    # np.round defaults to round-half-to-even, which is a common HLS rounding mode.
    rounded_value = np.round(scaled_value)

    # 3. Saturate (clamp) the rounded integer value to the valid range for a 16-bit signed integer.
    # This correctly handles values that exceed the ap_fixed type's range.
    saturated_value = np.clip(rounded_value, min_scaled_val, max_scaled_val)

    # 4. Cast to a 16-bit signed integer type
    # np.int16 corresponds to a signed 16-bit integer, exactly what ap_fixed<16,6>
    # would store internally (the scaled value).
    int16_value = saturated_value.astype(np.int16)

    # If the input was a scalar, return a scalar, otherwise return an array
    if isinstance(float_val, (float, int)):
        return int16_value.item() # .item() converts 0-d array to scalar
    else:
        return int16_value

def float_to_q2_5_int8(float_val):
    """
    Converts a floating-point number (or array of numbers) to its Q2.5 fixed-point
    representation, returned as an 8-bit signed integer (np.int8).

    Q2.5 format: 1 sign bit, 2 integer bits, 5 fractional bits = 8 bits total.
    Range for the floating-point input: approximately [-4.0, 3.96875].
    The scaling factor used is 2^5 = 32.

    Args:
        float_val (float or np.ndarray): The floating-point number(s) to convert.
                                        If an array, elements outside the Q2.5
                                        float range [-4, 3.96875] will be clamped.

    Returns:
        np.int8 or np.ndarray of np.int8: The Q2.5 fixed-point representation.
    """
    n_integer_bits = 2 # m
    n_fractional_bits = 5 # n

    scale = 2**n_fractional_bits  # 2^5 = 32

    # Ensure input is a NumPy array for consistent handling of scalars or lists
    float_array = np.array(float_val, dtype=np.float32)

    # 1. Multiply by the scale factor to shift the fractional part into integer range
    scaled_value = float_array * scale

    # 2. Round to the nearest integer
    # np.round handles rounding correctly for positive and negative numbers
    rounded_value = np.round(scaled_value)

    # 3. Clamp the rounded integer value to the valid range for an 8-bit signed integer [-128, 127]
    # This also implicitly handles clamping for values outside the Q2.5 float range.
    # For example, a float 5.0 (outside Q2.5 max 3.96875) would be scaled to 5*32=160,
    # then clamped to 127.
    int8_value = np.clip(rounded_value, -128, 127).astype(np.int8)

    return int8_value

if (0):
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = tf.expand_dims(train_images, -1)
    test_images = tf.expand_dims(test_images, -1)

    from tensorflow_model_optimization.quantization.keras.vitis.layers import vitis_activation

    # Define the model architecture.
    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), use_bias=True, activation='linear')(
            inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3), use_bias=True, activation='linear')(
            x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('gelu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(rate=0.1)(x)
    x = keras.layers.Dense(10)(x)
    predictions = x
    model = keras.Model(inputs=inputs, outputs=predictions, name="mnist_model")

    #  Train the float model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])

    log_dir = "logs/float_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    model.fit(
        train_images,
        train_labels,
        epochs=1,
        validation_data=(test_images, test_labels))

    model.save('float.h5')
    model.evaluate(test_images, test_labels, batch_size=500)

elif(1) :
    from tensorflow import keras
    from tensorflow.keras.utils import to_categorical
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import numpy as np

    seed = 0
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)
    ## Fetch the jet tagging dataset from Open ML ##
    data = fetch_openml('hls4ml_lhc_jets_hlf')
    X, y = data['data'], data['target']

    print(data['feature_names'])
    print(X.shape, y.shape)
    print(X[:5])
    print(y[:5])

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, 5)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(y[:5])

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    np.save('X_train_val.npy', X_train_val)
    np.save('X_test.npy', X_test)
    np.save('y_train_val.npy', y_train_val)
    np.save('y_test.npy', y_test)
    np.save('classes.npy', le.classes_)

    ### int8 npy part ###
    x_int8_value = float_to_ap_fixed_16_6(X_train_val)
    x_int8_value_c = x_int8_value.copy(order='C')
    np.save('X_train_int8_val.npy', x_int8_value_c)
    print(x_int8_value[0:3])
    print(x_int8_value_c.flags['C_CONTIGUOUS'], x_int8_value_c.flags['F_CONTIGUOUS'])

    y_int8_value = float_to_ap_fixed_16_6(y_train_val)
    np.save('y_train_int8_val.npy', y_int8_value)
    print(y_int8_value[0:3])

    x_int8_value = float_to_ap_fixed_16_6(X_test)
    x_int8_value_c = x_int8_value.copy(order='C')
    np.save('X_test_int8_val.npy', x_int8_value_c)

    y_int8_value = float_to_ap_fixed_16_6(y_test)
    np.save('y_test_int8_val.npy', y_int8_value)
    ### int8 npy end ###
    
    ## Now construct a model ##
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1
    from callbacks import all_callbacks

    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu1'))
    model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu2'))
    model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu3'))
    model.add(Dense(5, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))
    model.summary()

    # Define the model architecture.
    inputs = keras.layers.Input(shape=(16,), name='input')
    x = keras.layers.Dense(64, name='fc1')(inputs)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(32)(inputs)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(32)(inputs)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(5, name='fc4')(x)
    x = keras.layers.Activation('softmax', name='softmax')(x)
    predictions = x
    model = keras.Model(inputs=inputs, outputs=predictions, name="mnist_model")
    model.summary()

    #  Train the float model
    # model.compile(
    #     optimizer='adam',
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=['sparse_categorical_accuracy'])

    # log_dir = "logs/float_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir, histogram_freq=1)
    # model.fit(
    #     X_train_val,
    #     y_train_val,
    #     epochs=10,
    #     validation_data=(X_test, y_test))

    ## Train the model ##
    if 1:
        adam = Adam(lr=0.0001)
        model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        callbacks = all_callbacks(
            stop_patience=1000,
            lr_factor=0.5,
            lr_patience=10,
            lr_epsilon=0.000001,
            lr_cooldown=2,
            lr_minimum=0.0000001,
            outputDir='model_1',
        )
        model.fit(
            X_train_val,
            y_train_val,
            batch_size=1024,
            epochs=10,
            validation_split=0.25,
            shuffle=True,
            callbacks=callbacks.callbacks,
        )
        print(y_train_val[0:5])

else:
    model = keras.models.load_model('../KERAS_check_best_model.h5')

# Inspect the float model
from tensorflow_model_optimization.quantization.keras import vitis_inspect

# `target` is the target DPU to deploy this model, it can be a name(e.g. "DPUCZDX8G_ISA1_B4096"),
# a json(e.g. "./U50/arch.json") or a fingerprint.
inspector = vitis_inspect.VitisInspector(target='DPUCZDX8G_ISA1_B4096')

# In this model only `gelu` layer is not supported by DPU target.
# Inspect results will be shown on screen, and more detailed results will be saved in
# 'inspect_results.txt'. We can also visualize the results in 'model.svg'.
inspector.inspect_model(
    model,
    input_shape=None,
    dump_model=True,
    dump_model_file='inspect_model.h5',
    plot=True,
    plot_file='model.svg',
    dump_results=True,
    dump_results_file='inspect_results.txt',
    verbose=0)

### come from quantize with xcompiler ###

# Post-Training Quantize
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# `target` is the target DPU to deploy this model, it can be a name(e.g. "DPUCZDX8G_ISA1_B4096"),
# a json(e.g. "./U50/arch.json") or a fingerprint.
quantizer = vitis_quantize.VitisQuantizer(model, target='DPUCZDX8G_ISA1_B4096')
# quantized_model = quantizer.quantize_model(
#     calib_dataset=X_train_val[0:10],
#     include_cle=True,
#     cle_steps=10,
#     quantize_with_xcompiler=True,
#     include_fast_ft=True)
quantized_model = quantizer.quantize_model(
    calib_dataset=X_train_val[0:10],
    )

quantized_model.save('quantized.h5')

print(X_train_val[0:5])
