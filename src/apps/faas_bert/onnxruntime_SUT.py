# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

import array
import os
import sys
from timeit import default_timer as timer
sys.path.insert(0, os.getcwd())

import numpy as np
import onnxruntime


class BERT_ONNXRuntime_SUT():
    def __init__(self, model_path, quantized, profile, batchsize):
        self.profile = profile
        self.options = onnxruntime.SessionOptions()
        self.options.enable_profiling = profile
        self.options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.options.intra_op_num_threads = int(os.environ["TF_INTRA_OP_PARALLELISM_THREADS"])
        self.options.inter_op_num_threads = int(os.environ["TF_INTER_OP_PARALLELISM_THREADS"])
        self.batchsize = batchsize

        self.quantized = quantized
        if "RUN_CPU" in os.environ:
            providers = [ ('CPUExecutionProvider')]
        else:
            providers = [ ('CUDAExecutionProvider', {'gpu_mem_limit': int(4 * 1024 * 1024 * 1024), 
                'do_copy_in_default_stream': True, 'arena_extend_strategy': 'kSameAsRequested' } )]
        self.sess = onnxruntime.InferenceSession(model_path.read(), self.options, providers=providers)


    def run_with_each_sample(self, qsl, query_samples):
        responses = []
        for i in range(len(query_samples)):
            eval_features = qsl.get_features(query_samples[i]['idx'])
            if self.quantized:
                fd = {
                    "input_ids": np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :],
                    "attention_mask": np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :],
                    "token_type_ids": np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :]
                }
            else:
                fd = {
                    "input_ids": np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :],
                    "input_mask": np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :],
                    "segment_ids": np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :]
                }
            scores = self.sess.run([o.name for o in self.sess.get_outputs()], fd)
            output = np.stack(scores, axis=-1)[0]

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            responses.append((query_samples[i]['id'], bi[0], bi[1]))
        return responses

    def run_one_batch(self, qsl, query_samples, responses, cur_batch_size=1, base_index=0):
        input_ids_list = []
        attention_mask_list = []
        segment_ids_list = []
        for i in range(cur_batch_size):
            idx = base_index + i
            eval_features = qsl.get_features(query_samples[idx]['idx'])
            input_ids_list.append(eval_features.input_ids)
            attention_mask_list.append(eval_features.input_mask)
            segment_ids_list.append(eval_features.segment_ids)
        if self.quantized:
            fd = {
                "input_ids": np.array(input_ids_list).astype(np.int64),
                "attention_mask": np.array(attention_mask_list).astype(np.int64),
                "token_type_ids": np.array(segment_ids_list).astype(np.int64)
            }
        else:
            fd = {
                "input_ids": np.array(input_ids_list).astype(np.int64),
                "input_mask": np.array(attention_mask_list).astype(np.int64),
                "segment_ids": np.array(segment_ids_list).astype(np.int64)
            }
        scores = self.sess.run([o.name for o in self.sess.get_outputs()], fd)
        output = np.stack(scores, axis=-1)
        out_list = np.split(output, cur_batch_size, axis = 0)
        for i, o in enumerate(out_list):
            idx = base_index + i
            response_array = array.array("B", np.array(o).tobytes())
            bi = response_array.buffer_info()
            responses.append((query_samples[idx]['id'], bi[0], bi[1]))

    def issue_queries(self, qsl, query_samples):
        if self.batchsize > 1:
            num_samples = len(query_samples)
            num_batch = num_samples // self.batchsize
            remaining_batch = num_samples % self.batchsize
            responses = []
            for b in range(num_batch):
                base_index = b * self.batchsize
                self.run_one_batch(qsl, query_samples, responses,
                                   self.batchsize, base_index)
            if remaining_batch > 0:
                base_index = num_batch * self.batchsize
                self.run_one_batch(qsl, query_samples, responses,
                                   remaining_batch, base_index)
            return responses
        else:
            return self.run_with_each_sample(qsl, query_samples)

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        if self.profile:
            print("ONNX runtime profile dumped to: '{}'".format(self.sess.end_profiling()))


def get_onnxruntime_sut(model_path, quantized, profile, batchsize):
    return BERT_ONNXRuntime_SUT(model_path, quantized, profile, batchsize)
