/* Copyright 2022 Visiongo Ltd.
 * 
 * @author lijunyu
 * @date 2022/5/6
 * @file ssdmobilenetv1.cc
 */

#include "ssdmobilenetv1.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

template <class T>
std::vector<T> read_binary_file(const char *file_name)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    std::vector<T> vec(len / sizeof(T), 0);
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char *>(vec.data()), len);
    ifs.close();
    return vec;
}

SSDMobileNetV1::SSDMobileNetV1(struct data_shape input_shape,
                struct data_shape output_shape0,
                struct data_shape output_shape1)
                :input_shape(input_shape), output_shape0(output_shape0), output_shape1(output_shape1)
{
    share_memory = open(SHARE_MEMORY_DEV, O_RDWR);
    if(share_memory < 0) 
    {
        std::cerr << "open /dev/k510-share-memory error" << std::endl;
        std::abort();
    }
    mem_map = open(MAP_MEMORY_DEV, O_RDWR | O_SYNC);
    if (mem_map < 0) 
    {
        std::cerr << "open /dev/mem error" << std::endl;
        std::abort();
    }
}

void SSDMobileNetV1::prepare_memory()
{
    input_size = ((input_shape.weight * input_shape.height * (input_shape.channel + 1) * sizeof(float) + 4095) & (~4095));
    output_size[0] = (output_shape0.weight * output_shape0.height  * output_shape0.channel  * sizeof(float));
    output_size[1] = (output_shape1.weight * output_shape1.height  * output_shape1.channel  * sizeof(float));
    output_all_size = ((output_size[0] + output_size[1] + 4095) & (~4095));

    allocAlignMemSSDMobileNetV1Output.size = output_all_size;
    allocAlignMemSSDMobileNetV1Output.alignment = MEMORY_TEST_BLOCK_ALIGN;
    allocAlignMemSSDMobileNetV1Output.phyAddr = 0;

    if(ioctl(share_memory, SHARE_MEMORY_ALIGN_ALLOC, &allocAlignMemSSDMobileNetV1Output) < 0) {
        std::cerr << "alloc allocAlignMemOdOutput error" << std::endl;
        std::abort();
    }

    virtual_addr_output = (char *)mmap(NULL, allocAlignMemSSDMobileNetV1Output.size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_map, allocAlignMemSSDMobileNetV1Output.phyAddr);
    if(virtual_addr_output == MAP_FAILED) {
        std::cerr << "map allocAlignMemOdOutput error" << std::endl;
        std::abort();
    }
    
    virtualAddrSSDMobileNetV1Output[0] = virtual_addr_output;
    output_pa_addr[0] = allocAlignMemSSDMobileNetV1Output.phyAddr;
    for(uint32_t i = 1; i < 2; i++) {
        virtualAddrSSDMobileNetV1Output[i] = virtualAddrSSDMobileNetV1Output[i - 1] + output_size[i - 1];
        output_pa_addr[i] = output_pa_addr[i - 1] + output_size[i - 1];
    }

    for(uint32_t i = 0; i < GNNE_BUFFERS_COUNT; i++) {
        allocAlignMemSSDMobileNetV1Input[i].size = input_size;
        allocAlignMemSSDMobileNetV1Input[i].alignment = MEMORY_TEST_BLOCK_ALIGN;
        allocAlignMemSSDMobileNetV1Input[i].phyAddr = 0;

        if(ioctl(share_memory, SHARE_MEMORY_ALIGN_ALLOC, &allocAlignMemSSDMobileNetV1Input[i]) < 0) {
            std::cerr << "alloc allocAlignMemOdInput error" << std::endl;
            std::abort();
        }
        virtual_addr_input[i] = (char *)mmap(NULL, allocAlignMemSSDMobileNetV1Input[i].size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_map, allocAlignMemSSDMobileNetV1Input[i].phyAddr);
        if(virtual_addr_input[i] == MAP_FAILED) {
            std::cerr << "map allocAlignMemOdInput error" << std::endl;
            std::abort();
        }
    }
}

void SSDMobileNetV1::set_input(uint32_t index)
{
    auto in_shape = interp_ssdmobilenetv1.input_shape(index);

    auto input_tensor = host_runtime_tensor::create(dt_float32, in_shape,
        { (gsl::byte *)virtual_addr_input[index], input_shape.weight * input_shape.height * input_shape.channel * sizeof(float)},
        false, hrt::pool_shared, allocAlignMemSSDMobileNetV1Input[index].phyAddr)
                            .expect("cannot create input tensor");
    interp_ssdmobilenetv1.input_tensor(index, input_tensor).expect("cannot set input tensor");
}

void SSDMobileNetV1::set_output()
{
    for (size_t i = 0; i < interp_ssdmobilenetv1.outputs_size(); i++) {
        auto out_shape = interp_ssdmobilenetv1.output_shape(i);
        auto output_tensor = host_runtime_tensor::create(dt_float32, out_shape,
        { (gsl::byte *)virtualAddrSSDMobileNetV1Output[i], output_size[i]},
        false, hrt::pool_shared, output_pa_addr[i])
                            .expect("cannot create output tensor");

        interp_ssdmobilenetv1.output_tensor(i, output_tensor).expect("cannot set output tensor");
    }
}

void SSDMobileNetV1::load_model(char *path)
{
    ssdmobilenetv1_model = read_binary_file<unsigned char>(path);
    interp_ssdmobilenetv1.load_model({ (const gsl::byte *)ssdmobilenetv1_model.data(), ssdmobilenetv1_model.size() }).expect("cannot load model.");
    std::cout << "============> interp_ssdmobilenetv1.load_model finished!" << std::endl;
}

void SSDMobileNetV1::run()
{
    interp_ssdmobilenetv1.run().expect("error occurred in running model");
}

void SSDMobileNetV1::get_output()
{
    output_0 = reinterpret_cast<float *>(virtualAddrSSDMobileNetV1Output[0]);
    output_1 = reinterpret_cast<float *>(virtualAddrSSDMobileNetV1Output[1]);
}

SSDMobileNetV1::~SSDMobileNetV1()
{
    for(uint32_t i = 0; i < GNNE_BUFFERS_COUNT; i++) {
        if(virtual_addr_input[i])
            munmap(virtual_addr_input[i], allocAlignMemSSDMobileNetV1Input[i].size);

        if(allocAlignMemSSDMobileNetV1Input[i].phyAddr != 0) {
            if(ioctl(share_memory, SHARE_MEMORY_FREE, &allocAlignMemSSDMobileNetV1Input[i].phyAddr) < 0) {
                std::cerr << "free allocAlignMemOdInput error" << std::endl;
                std::abort();
            }
        }
    }
    if(virtual_addr_output)
        munmap(virtual_addr_output, allocAlignMemSSDMobileNetV1Output.size);

    if(allocAlignMemSSDMobileNetV1Output.phyAddr != 0) {
        if(ioctl(share_memory, SHARE_MEMORY_FREE, &allocAlignMemSSDMobileNetV1Output.phyAddr) < 0) {
            std::cerr << "free allocAlignMemOdOutput error" << std::endl;
            std::abort();
        }
    }
    close(share_memory);
    close(mem_map);
}

