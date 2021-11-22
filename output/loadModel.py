import numpy as np
import tensorflow as tf
import tflearn

meta_path = 'models/mstar_targets/mstarnet/9641.meta' # Your .meta file
output_node_names = ['output:0']    # Output nodes

with tf.compat.v1.Session() as sess:
    # Restore the graph
    saver = tf.compat.v1.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('path/of/your/.meta/file'))

    # Freeze the graph
    output_node_names = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())