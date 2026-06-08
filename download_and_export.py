import os
import torch
from transformers import AutoTokenizer, AutoModel
import onnx
from onnx import helper, TensorProto

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "model")
os.makedirs(model_dir, exist_ok=True)

print("1. Downloading model from Hugging Face...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

print("2. Saving tokenizer to JSON...")
tokenizer.backend_tokenizer.save(os.path.join(model_dir, "tokenizer.json"))

print("3. Exporting base model to ONNX...")
inputs = tokenizer("Dummy text for ONNX export.", return_tensors="pt")
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    os.path.join(model_dir, "onnx_model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=18
)

print("4. Generating valid onnx_classifier.onnx...")
axes_initializer = helper.make_tensor(
    name='axes_tensor',
    data_type=TensorProto.INT64,
    dims=[1],
    vals=[1]
)

node_reduce = helper.make_node(
    'ReduceMax',
    inputs=['input_tensor', 'axes_tensor'],
    outputs=['reduced_tensor'],
    keepdims=0
)

node_argmax = helper.make_node(
    'ArgMax',
    inputs=['reduced_tensor'],
    outputs=['output_tensor'],
    axis=1,
    keepdims=0
)

graph = helper.make_graph(
    [node_reduce, node_argmax],
    'classifier-graph',
    [helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [None, None, 768])],
    [helper.make_tensor_value_info('output_tensor', TensorProto.INT64, [None])],
    initializer=[axes_initializer]
)

onnx_model = helper.make_model(graph, producer_name='mlops-classifier')
onnx_model.opset_import[0].version = 18

onnx.save(onnx_model, os.path.join(model_dir, "onnx_classifier.onnx"))
print("Success! Models are ready in: ", model_dir)
