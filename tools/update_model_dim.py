import argparse
import onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    args = parser.parse_args()
    model = onnx.load(args.input_model)
    inputs = model.graph.input
    outputs = model.graph.output
    print(inputs)
    print(outputs)
    for i in range(len(inputs)):
        input_dim1 = inputs[i].type.tensor_type.shape.dim[0]
        input_dim1.dim_param = "N"
    for i in range(len(outputs)):
        output_dim1 = outputs[i].type.tensor_type.shape.dim[0]
        output_dim1.dim_param = "N"
    onnx.save(model, args.output_model)
    onnx.checker.check_model(args.output_model)


if __name__ == '__main__':
    main()
