/* Copyright 2022 Visiongo Ltd.
 * 
 * @author lijunyu
 * @date 2022/5/6
 * @file inference_kmodel.h
 */

#ifndef _INFERENCE_KMODEL
#define _INFERENCE_KMODEL

#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>

#include "utils.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;
using namespace std;

namespace inferencekmodel
{

class InferenceKmodel
{
private:
    /* data */
    uint32_t *inputSize;
    uint32_t *outputSize;
    uint32_t inputAllSize;
    uint32_t outputAllSize;
    interpreter interpKmodel;
    vector<unsigned char> kmodel;

    vector<struct data_shape> inputShapes;
    vector<struct data_shape> outputShapes;
    datatype_t inputDataType;
    datatype_t outputDataType;
    size_t inputDataTypeSize;
    size_t outputDataTypeSize;

    struct share_memory_alloc_align_args allocAlignMemNetOutput;
    struct share_memory_alloc_align_args *allocAlignMemNetInput;

    float **outputs;

    size_t getTypeSize(datatype_t dataType);
public:
    char **virtualAddrKmodelOutputs;
    char **virtualAddrKmodelInput;
    uint32_t *outputPaAddr;

    int shareMemory;
    int memMap;

    InferenceKmodel(vector<struct data_shape> inputShapes, vector<struct data_shape> outputShapes,
                    datatype_t inputDataType, datatype_t outputDataType);
    void prepareMemory();
    void setInput(uint32_t index);
    void setOutput();
    void loadKmodel(char *path);
    void run();
    void getOutput();
    ~InferenceKmodel();
};

} // namespace inferencekmodel

#endif // !_INFERENCE_KMODEL