from tensorflow.keras import layers, models

def build_rnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape[0], input_shape[1]*input_shape[2]), input_shape=input_shape),
        layers.SimpleRNN(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
