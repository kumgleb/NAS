import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L


class SuperNet(tf.Module):

    def __init__(self, n_filters, n_dense, n_classes):
        self.conv_3x3_1 = L.Conv2D(n_filters, (3, 3), padding='same', activation='relu', use_bias=False,
                                   name='3x3_1_conv_subnets_1_and_2')
        self.conv_3x3_2 = L.Conv2D(n_filters, (3, 3), padding='same', activation='relu', use_bias=False,
                                   name='3x3_2_conv_subnets_1_and_3')
        self.conv_5x5_1 = L.Conv2D(n_filters, (5, 5), padding='same', activation='relu', use_bias=False,
                                   name='5x5_1_conv_subnets_3_and_4')
        self.conv_5x5_2 = L.Conv2D(n_filters, (5, 5), padding='same', activation='relu', use_bias=False,
                                   name='5x5_2_conv_subnets_2_and_4')
        self.max_pool = L.MaxPooling2D(pool_size=(2, 2), name='MaxPooling')
        self.dense_1 = L.Dense(n_dense, activation='relu', name='dense_1')
        self.dense_output = L.Dense(n_classes, activation=None, name='dense_output')

    def sample_subnet(self, subnet_idx, path_dropout_prob=0):

        """
            Assemble one of 4 available sub networks from shared layers.
            `subnet_idx` - index of specific sub network (int from 0 to 4).
                           0 - corresponds to SuperNet;
                           1-4 - corresponds to SubNet's;
            `path_dropout_prob` - regulates probability of path dropout,
                                  utilized with `path_dropout` training strategy.
            returns:
              model with architecture corresponds to `subnet_idx`.
        """
        input = tf.keras.Input(shape=(28, 28, 1))

        # Select one of convolution layers w.r.t. subnet_idx
        if subnet_idx == 0:
            hidden_1 = self.conv_3x3_1(input) * np.random.binomial(1, 1 - path_dropout_prob)
            hidden_2 = self.conv_5x5_1(input) * np.random.binomial(1, 1 - path_dropout_prob)
        elif subnet_idx in [1, 2]:
            hidden_1 = self.conv_3x3_1(input)
            hidden_2 = tf.zeros_like(hidden_1)
        elif subnet_idx in [3, 4]:
            hidden_2 = self.conv_5x5_1(input)
            hidden_1 = tf.zeros_like(hidden_2)
        else:
            raise ValueError(f'Valid `subnet_idx` is in {0, 1, 2, 3, 4}, provided: {subnet_idx}')

        hidden = L.Concatenate()([hidden_1, hidden_2])
        hidden = self.max_pool(hidden)

        # Select one of convolution layers w.r.t. subnet_idx
        if subnet_idx == 0:
            hidden_1 = self.conv_3x3_2(hidden) * np.random.binomial(1, 1 - path_dropout_prob)
            hidden_2 = self.conv_5x5_2(hidden) * np.random.binomial(1, 1 - path_dropout_prob)
        if subnet_idx in [1, 3]:
            hidden_1 = self.conv_3x3_2(hidden)
            hidden_2 = tf.zeros_like(hidden_1)
        elif subnet_idx in [2, 4]:
            hidden_2 = self.conv_5x5_2(hidden)
            hidden_1 = tf.zeros_like(hidden_2)

        hidden = L.Concatenate()([hidden_1, hidden_2])
        hidden = self.max_pool(hidden)
        hidden = L.Flatten()(hidden)
        hidden = self.dense_1(hidden)
        output = self.dense_output(hidden)
        model = tf.keras.Model(input, output, name=f'subnet_{subnet_idx}')
        return model

    def _batch_gen(self, X, y, batch_size):
        n_batches = X.shape[0] // batch_size
        if X.shape[0] % batch_size == 0:
            for i in range(n_batches):
                yield X[i * batch_size: (i + 1) * batch_size, :, :, :], y[i * batch_size: (i + 1) * batch_size]
        else:
            for i in range(n_batches + 1):
                yield X[i * batch_size: (i + 1) * batch_size, :, :, :], y[i * batch_size: (i + 1) * batch_size]

    def loss(self, model, X, y):
        y_predicted = model(X)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = loss_object(y, y_predicted)
        return loss

    def grads(self, model, X, y):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, X, y)
            gradients = tape.gradient(loss_value, model.trainable_variables)
        return loss_value, gradients

    def _train_random_subnet(self, X, y, optimizer, model_idx):
        model = self.sample_subnet(model_idx)
        loss_value, grads = self.grads(model, X, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    def _train_with_path_dropout(self, X, y, optimizer, path_dropout_prob):
        model = self.sample_subnet(0, path_dropout_prob)
        loss_value, grads = self.grads(model, X, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    def eval_supernet(self, validation_data):
        X_val, y_val = validation_data
        supernet = self.sample_subnet(0)
        logits = supernet(X_val)
        prediction = tf.argmax(logits, axis=1)
        test_accuracy = tf.keras.metrics.Accuracy()(prediction, y_val)
        return test_accuracy

    def fit(self, X_train, y_train, n_epoches, batch_size, learning_rate,
            validation_data=None, n_btch_to_switch_subnet=None,
            path_dropout_prob=0, train_mode='path_dropout', verbosity=4):

        """
         Train SuperNet with one of defined strategies:
                'random_subnet' - randomly choose one of SubNets (including SuperNet)
                                  train on it and choose another SubNet after every
                                  `n_btch_to_switch_subnet` batches for every epoch.
                'path_dropout' - randomly zeroing convolution layers with probability
                                 `path_dropout_prob` on every iteration.

            Evaluates accuracy of SuperNet on validation data after every epoch.
        """

        if n_btch_to_switch_subnet and n_btch_to_switch_subnet > X_test.shape[0] // batch_size:
            raise ValueError(f'`n_btch_to_switch_subnet` should be less or equal to number of batches.')

        optimizer = tf.keras.optimizers.Adam(learning_rate)

        for epoch in range(n_epoches):
            epoch_loss = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            for i, (X, y) in enumerate(self._batch_gen(X_train, y_train, batch_size)):

                if train_mode == 'random_subnet':
                    if i % n_btch_to_switch_subnet == 0:
                        model_idx = np.random.randint(0, 5)
                    loss_value = self._train_random_subnet(X, y, optimizer, model_idx)

                elif train_mode == 'path_dropout':
                    loss_value = self._train_with_path_dropout(X, y, optimizer, path_dropout_prob)

                epoch_loss.update_state(loss_value)
                supernet = self.sample_subnet(0)
                epoch_accuracy.update_state(y, supernet(X))

            if epoch % verbosity == 0:
                print(f'Epoch: {epoch}, SuperNet train loss: {epoch_loss.result():.4f}, SuperNet train accuracy: {epoch_accuracy.result():.2%}')
                if validation_data:
                    supernet_validation_accuracy = self.eval_supernet(validation_data)
                    print(f'SuperNet validation accuracy: {supernet_validation_accuracy:.2%}')