import numpy as np
import tensorflow as tf


def evaluate_accuracy(model, X_test, y_test):
  logits = model(X_test)
  prediction = tf.argmax(logits, axis=1)
  accuracy = tf.keras.metrics.Accuracy()(prediction, y_test)
  return accuracy

def train_and_evaluate_from_scratch(train_data, validation_data, test_data, supernet,
                                    n_filters, n_dense, n_classes, lr, batch_size,
                                    patience=2, n_loops=5):
  """
    Function initializes each SubNet with random weights train it on `train_data`,
    with early stopping for accuracy evaluated on `validation_data`.
    Trained model evaluates on `test_data`.

    Function repeats train `n_loops` times to calculate mean and std of model accuracy
    evaluated on `test_data`.
  """

  X_train, y_train = train_data
  X_val, y_val = validation_data
  X_test, y_test = test_data

  for model_idx in [1, 2, 3, 4]:
    accuracy = []
    for _ in range(n_loops):
      model = supernet(n_filters, n_dense, n_classes)
      subnet = model.sample_subnet(model_idx)
      subnet.compile(optimizer=tf.keras.optimizers.Adam(lr),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
      subnet.fit(X_train, y_train,
                 epochs=50,
                 batch_size=batch_size,
                 validation_data=(X_val, y_val),
                 verbose=0,
                 callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            verbose=0,
                                                            restore_best_weights=True))
      eval_acc = subnet.evaluate(X_test, y_test)
      accuracy.append(eval_acc[1])

    accuracy = np.array(accuracy)
    print(f'SubNet {model_idx} test accuracy: {accuracy.mean():.2%}+-{accuracy.std():.2%}')