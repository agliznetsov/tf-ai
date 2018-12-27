import tensorflow as tf

columns = [tf.int32] * 10

dataset = tf.data.experimental.CsvDataset("./data/t.csv", columns)

print(dataset)
print(dataset.output_shapes)

next_element = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    while True:
        try:
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            break

