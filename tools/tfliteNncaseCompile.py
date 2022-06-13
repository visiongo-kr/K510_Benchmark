# -*- coding: utf-8 -*-
"""
@File    :   tfliteNncaseCompile.py
@Time    :   2022/04/28 22:04:33
@Author  :   lijunyu
@Version :   0.0.2
@Desc    :   None
"""

from turtle import st
import nncase
import argparse

from numpy import uint8

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', type=str, required=True, help='path to tflite model')
    parser.add_argument('--output_path', type=str, default='test.kmodel', help='output kmodel')
    parser.add_argument('--input_type', type=str, default='float32', help='model input type')
    parser.add_argument('--output_type', type=str, default='float32', help='model output type')
    parser.add_argument('--quant_type', type=str, default=None, help='quant type')
    parser.add_argument('--w_quant_type', type=str, default=None, help='w_quant_type')
    args = parser.parse_args()
    model=args.input_model
    target = 'k510'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'
    if args.quant_type != None:
        compile_options.quant_type = args.quant_type
    if args.w_quant_type != None:
        compile_options.w_quant_type = args.w_quant_type
    compile_options.input_type = args.input_type
    compile_options.output_type = args.output_type

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    # import
    model_content = read_model_file(model)
    compiler.import_tflite(model_content, import_options)

    # compile
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open(args.output_path, 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()