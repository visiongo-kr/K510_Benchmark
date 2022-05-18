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
    args = parser.parse_args()
    model=args.input_model
    target = 'k510'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'

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