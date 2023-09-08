#!/usr/bin/env python3
import os, sys


def load_tf_model(model_iof, inputs, outputs, mem_pct = 0.45):
    #import tensorflow as tf
    
    from tensorflow.compat.v1 import GraphDef, import_graph_def, Session, ConfigProto

    from tensorflow import dtypes
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

    #infer_config = tf.compat.v1.ConfigProto()
    infer_config = ConfigProto()
    infer_config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_OP_PARALLELISM_THREADS']) \
        if 'TF_INTRA_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
    infer_config.inter_op_parallelism_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS']) \
        if 'TF_INTER_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
    infer_config.use_per_session_threads = 1

    infer_config.gpu_options.allow_growth = True
    infer_config.gpu_options.per_process_gpu_memory_fraction = mem_pct

    print("q", file=sys.stderr, flush=True)

    #graph_def = tf.compat.v1.GraphDef()
    graph_def = GraphDef()
    #with tf.compat.v1.gfile.FastGFile(model_path, "rb") as f:
    #    graph_def.ParseFromString(f.read())
    graph_def.ParseFromString(model_iof.getbuffer())
    
    print("w", file=sys.stderr, flush=True)

    as_datatype_enum = dtypes.float32.as_datatype_enum
    optimized_graph_def = optimize_for_inference(graph_def, [item.split(':')[0] for item in inputs],
                                                 [item.split(':')[0] for item in outputs], as_datatype_enum, False)
    print("e", file=sys.stderr, flush=True)
    graph_def = optimized_graph_def

    #g = tf.compat.v1.import_graph_def(graph_def, name='')
    g =import_graph_def(graph_def, name='')
    print("r", file=sys.stderr, flush=True)
    #sess = tf.compat.v1.Session(graph=g, config=infer_config)
    sess = Session(graph=g, config=infer_config)
    print("t", file=sys.stderr, flush=True)
    return sess

