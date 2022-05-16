/* Copyright 2022 Visiongo Ltd.
 * 
 * @author lijunyu
 * @date 2022/5/6
 * @file inference_kmodel.cc
 */

#include "inference_kmodel.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

namespace inferencekmodel
{

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

InferenceKmodel::InferenceKmodel(vector<struct data_shape> inputShapes, vector<struct data_shape> outputShapes,
                                datatype_t inputDataType, datatype_t outputDataType)
:inputShapes(inputShapes), outputShapes(outputShapes), inputDataType(inputDataType), outputDataType(outputDataType)
{
    inputSize = new uint32_t[inputShapes.size()];
    outputSize = new uint32_t[outputShapes.size()];
    outputs = new float *[outputShapes.size()];
    virtualAddrKmodelOutputs = new char *[outputShapes.size()];
    virtualAddrKmodelInput = new char *[inputShapes.size()];
    outputPaAddr = new uint32_t[outputShapes.size()];
    allocAlignMemNetInput = new struct share_memory_alloc_align_args[inputShapes.size()];

    shareMemory = open(SHARE_MEMORY_DEV, O_RDWR);
    if(shareMemory < 0) 
    {
        std::cerr << "open /dev/k510-share-memory error" << std::endl;
        std::abort();
    }
    memMap = open(MAP_MEMORY_DEV, O_RDWR | O_SYNC);
    if (memMap < 0) 
    {
        std::cerr << "open /dev/mem error" << std::endl;
        std::abort();
    }
}

void InferenceKmodel::prepareMemory()
{
    inputAllSize = 0;
    for (int i = 0; i < inputShapes.size(); i++) {
        inputSize[i] = inputShapes[i].weight * inputShapes[i].height * (inputShapes[i].channel + 1);
        inputAllSize += inputSize[i];
    }
    outputAllSize = 0;
    for (int i = 0; i < outputShapes.size(); i++) {
        outputSize[i] = outputShapes[i].weight * outputShapes[i].height * outputShapes[i].channel;
        outputAllSize += outputSize[i];
    }

    allocAlignMemNetOutput.size = outputAllSize;
    allocAlignMemNetOutput.alignment = MEMORY_TEST_BLOCK_ALIGN;
    allocAlignMemNetOutput.phyAddr = 0;
    if (ioctl(shareMemory, SHARE_MEMORY_ALIGN_ALLOC, &allocAlignMemNetOutput) < 0) {
        std::cerr << "alloc allocAlignMemOdOutput error" << std::endl;
        std::abort();
    }
    virtualAddrKmodelOutputs[0] = (char *)mmap(NULL, allocAlignMemNetOutput.size, PROT_READ | PROT_WRITE, MAP_SHARED, memMap, allocAlignMemNetOutput.phyAddr);
    if(virtualAddrKmodelOutputs[0] == MAP_FAILED) {
        std::cerr << "map allocAlignMemOdOutput error" << std::endl;
        std::abort();
    }
    outputPaAddr[0] = allocAlignMemNetOutput.phyAddr;
    for(uint32_t i = 1; i < 2; i++) {
        virtualAddrKmodelOutputs[i] = virtualAddrKmodelOutputs[i - 1] + outputSize[i - 1];
        outputPaAddr[i] = outputPaAddr[i - 1] + outputSize[i - 1];
    }

    for(uint32_t i = 0; i < inputShapes.size(); i++) {
        allocAlignMemNetInput[i].size = inputSize[i];
        allocAlignMemNetInput[i].alignment = MEMORY_TEST_BLOCK_ALIGN;
        allocAlignMemNetInput[i].phyAddr = 0;

        if(ioctl(shareMemory, SHARE_MEMORY_ALIGN_ALLOC, &allocAlignMemNetInput[i]) < 0) {
            std::cerr << "alloc allocAlignMemOdInput error" << std::endl;
            std::abort();
        }
        virtualAddrKmodelInput[i] = (char *)mmap(NULL, allocAlignMemNetInput[i].size, PROT_READ | PROT_WRITE, MAP_SHARED, memMap, allocAlignMemNetInput[i].phyAddr);
        if(virtualAddrKmodelInput[i] == MAP_FAILED) {
            std::cerr << "map allocAlignMemOdInput error" << std::endl;
            std::abort();
        }
    }
}

void InferenceKmodel::setInput(uint32_t index)
{
    auto in_shape = interpKmodel.input_shape(index);

    auto input_tensor = host_runtime_tensor::create(inputDataType, in_shape,
        { (gsl::byte *)virtualAddrKmodelInput[index], 
        inputShapes[index].weight * inputShapes[index].height * inputShapes[index].channel * sizeof(float)},
        false, hrt::pool_shared, allocAlignMemNetInput[index].phyAddr)
                            .expect("cannot create input tensor");
    interpKmodel.input_tensor(index, input_tensor).expect("cannot set input tensor");
}

void InferenceKmodel::setOutput()
{
    for (size_t i = 0; i < interpKmodel.outputs_size(); i++) {
        auto out_shape = interpKmodel.output_shape(i);
        auto output_tensor = host_runtime_tensor::create(outputDataType, out_shape,
        { (gsl::byte *)virtualAddrKmodelOutputs[i], outputSize[i]},
        false, hrt::pool_shared, outputPaAddr[i])
                            .expect("cannot create output tensor");

        interpKmodel.output_tensor(i, output_tensor).expect("cannot set output tensor");
    }
}

void InferenceKmodel::loadKmodel(char *path)
{
    kmodel = read_binary_file<unsigned char>(path);
    interpKmodel.load_model({ (const gsl::byte *)kmodel.data(), kmodel.size() }).expect("cannot load model.");
    std::cout << "============> interp_od.load_model finished!" << std::endl;
}

void InferenceKmodel::run()
{
    interpKmodel.run().expect("error occurred in running model");
}

void InferenceKmodel::getOutput()
{
    
}

InferenceKmodel::~InferenceKmodel()
{
    for(uint32_t i = 0; i < inputShapes.size(); i++) {
        if(virtualAddrKmodelInput[i])
            munmap(virtualAddrKmodelInput[i], allocAlignMemNetInput[i].size);

        if(allocAlignMemNetInput[i].phyAddr != 0) {
            if(ioctl(shareMemory, SHARE_MEMORY_FREE, &allocAlignMemNetInput[i].phyAddr) < 0) {
                std::cerr << "free allocAlignMemOdInput error" << std::endl;
                std::abort();
            }
        }
    }
    if(virtualAddrKmodelOutputs[0])
        munmap(virtualAddrKmodelOutputs[0], allocAlignMemNetOutput.size);

    if(allocAlignMemNetOutput.phyAddr != 0) {
        if(ioctl(shareMemory, SHARE_MEMORY_FREE, &allocAlignMemNetOutput.phyAddr) < 0) {
            std::cerr << "free allocAlignMemOdOutput error" << std::endl;
            std::abort();
        }
    }
    close(shareMemory);
    close(memMap);

    delete inputSize;
    delete outputSize;
    delete outputs;
    delete virtualAddrKmodelOutputs;
    delete virtualAddrKmodelInput;
    delete outputPaAddr;
    delete allocAlignMemNetInput;
}

} // namespace inferencekmodel