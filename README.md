# K510_Benchmark

Benchmark program for k510.

## Test Model

| Network                                                      | DataSet   | Resolution | Time(ms) |
| ------------------------------------------------------------ | --------- | ---------- | -------- |
| [YoloFastV2](https://github.com/dog-qiuqiu/Yolo-FastestV2) | COCO 2017 | 352x352    | 98       |
| [SSD mobilenet_V1](https://tfhub.dev/iree/lite-model/ssd_mobilenet_v1_100_320/fp32/default/1) | COCO 2017 | 320x320    | 7        |
| ...                                                          | ...       | ...        |          |

## Model Convert

​	Model converted by [nncase](https://github.com/kendryte/nncase).

​	nncase version v1.5.0

​	python version 3.9

​	[onnx-simplifier](https://github.com/daquexian/onnx-simplifier) 0.3.8