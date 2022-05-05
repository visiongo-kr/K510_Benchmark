/* Copyright 2022 Visiongo Ltd.
 * 
 * @author lijunyu
 * @date 2022/5/6
 * @file ssdmobilenetv1.h
 */

#ifndef _SSDMOBILENETV1
#define _SSDMOBILENETV1

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

#define GNNE_BUFFERS_COUNT          1

class SSDMobileNetV1
{
private:
    /* data */
    uint32_t input_size;
    uint32_t output_size[2];
    uint32_t output_all_size;
    interpreter interp_ssdmobilenetv1;
    vector<unsigned char> ssdmobilenetv1_model;

    struct data_shape input_shape;
    struct data_shape output_shape0;
    struct data_shape output_shape1;

    float *output_0;
    float *output_1;
public:
    SSDMobileNetV1(struct data_shape input_shape,
                struct data_shape output_shape0,
                struct data_shape output_shape1);
    void prepare_memory();
    void set_input(uint32_t index);
    void set_output();
    void load_model(char *path);
    void run();
    void get_output();
    ~SSDMobileNetV1();

    char *virtualAddrSSDMobileNetV1Output[2];
    uint32_t output_pa_addr[2];

    char *virtual_addr_output;
    char *virtual_addr_input[GNNE_BUFFERS_COUNT];
    struct share_memory_alloc_align_args allocAlignMemSSDMobileNetV1Output;
    struct share_memory_alloc_align_args allocAlignMemSSDMobileNetV1Input[GNNE_BUFFERS_COUNT];

    int share_memory;
    int mem_map;
};


#endif // !_SSDMOBILENETV1