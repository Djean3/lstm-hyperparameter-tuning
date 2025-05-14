import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU is available:")
    for gpu in gpus:
        print("  -", gpu)
else:
    print("❌ No GPU detected.")

# Basic computation test
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.matmul(a, b)

print("Matrix multiplication result:\n", c.numpy())
