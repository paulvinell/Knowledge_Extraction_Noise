import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

optimizer = keras.optimizers.Adam(1e-4)
optimizer_noise = keras.optimizers.Adam(1e-2)
cross_entropy = keras.losses.CategoricalCrossentropy(from_logits=True)

@tf.function
def train_random_noise_step(input_shape, student, teacher):
    noise = tf.random.normal([32, input_shape[0], input_shape[1], input_shape[2]])
    noise = tf.clip_by_value(noise, -1, 1)
    student_loss = student_loss_func(student, teacher, noise)
    optimizer.minimize(student_loss, student.trainable_variables)

def train_adversarial_noise_step(input_shape, student, teacher):
    noise = tf.random.normal([32, input_shape[0], input_shape[1], input_shape[2]])
    noise = tf.clip_by_value(noise, -1, 1)
    noise = tf.Variable(noise)

    goal_label = np.random.randint(10, size=32)
    goal_label = keras.utils.to_categorical(goal_label, 10)

    last_loss = 1e10
    for i in range(100):
        noise_loss = adversarial_noise_loss(teacher, noise, goal_label)
        optimizer_noise.minimize(noise_loss, [noise])
        new_loss = noise_loss()
        if last_loss - new_loss <= 1e-5:
            break
        else:
            last_loss = new_loss

    student_loss = student_loss_func(student, teacher, noise)
    optimizer.minimize(student_loss, student.trainable_variables)

def adversarial_noise_loss(teacher, noise, goal_label):
    def loss():
        prediction = teacher(noise, training=True)
        ce = cross_entropy(goal_label, prediction)
        return ce
    return loss

def student_loss_func(student, teacher, noise):
    def loss():
        teacher_output = teacher(noise, training=True)
        student_output = student(noise, training=True)
        ce = cross_entropy(teacher_output, student_output)
        return ce
    return loss
