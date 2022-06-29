#include <nncase/runtime/host_runtime_tensor.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <signal.h>

/*  进程优先级  */
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <thread>
#include <mutex>
/* 申请物理内存 */
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <atomic>
#include <vector>

#include "k510_drm.h"
#include "media_ctl.h"
#include <linux/videodev2.h>
#include <ctime>
#include "yolofastv2.h"
#include "ssdmobilenetv1.h"
#include "inference_kmodel.h"

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;
    /*
     * 1, model path
     * 2, input type
     * 3, output type
     * 4, input shape
     * 5, output shape
     * 6, run count
     */
    std::cout << argc << std::endl;
    if (argc != 7) {
        std::cerr << "1, model path\n2, input type\n3, output type\n4, input shape\n5, output shape\n6, run count" << std::endl;
        return -1;
    }
    char *kmodel_name = argv[1];
    printf("%s\n", kmodel_name);
    // get input type
    datatype_t inputDataType;
    switch (argv[2][0]) {
        case '0':
            printf("float32\n");
            inputDataType = dt_float32;
            break;
        case '1':
            printf("uint8\n");
            inputDataType = dt_uint8;
            break;
    }
    // get output type
    datatype_t outputDataType;
    switch (argv[3][0]) {
        case '0':
            printf("float32\n");
            outputDataType = dt_float32;
            break;
        case '1':
            printf("uint8\n");
            outputDataType = dt_uint8;
            break;
    }
    uint32_t read_shape[3];
    // get input shape
    vector<struct data_shape> inputShapes;
    for (int i = 0, j = 0; argv[4][i] != 0; i++) {
        if (argv[4][i] >= '0' && argv[4][i] <= '9') {
            uint32_t a = 0;
            sscanf(&argv[4][i], "%d", &a);
            while (argv[4][i + 1] >= '0' && argv[4][i + 1] <= '9') i++;
            read_shape[j] = a;
            j++;
            if (j == 3) {
                inputShapes.push_back({read_shape[0], read_shape[1], read_shape[2]});
                j = 0;
            }
            printf("%d-", a);
        }
    }
    printf("\n");
    // get output shape
    vector<struct data_shape> outputShapes;
    for (int i = 0, j = 0; argv[5][i] != 0; i++) {
        if (argv[5][i] >= '0' && argv[5][i] <= '9') {
            int a = 0;
            sscanf(&argv[5][i], "%d", &a);
            while (argv[5][i + 1] >= '0' && argv[5][i + 1] <= '9') i++;
            read_shape[j] = a;
            j++;
            if (j == 3) {
                outputShapes.push_back({read_shape[0], read_shape[1], read_shape[2]});
                j = 0;
            }
            printf("%d-", a);
        }
    }
    printf("\n");
    int run_count = 0;
    sscanf(argv[6], "%d", &run_count);
    printf("%d\n", run_count);

    clock_t start, end;
    start = clock();

    inferencekmodel::InferenceKmodel *inf = new inferencekmodel::InferenceKmodel(inputShapes, outputShapes, inputDataType, outputDataType);
    inf->loadKmodel(kmodel_name);
    inf->prepareMemory();
    printf("prepare successed!\n");

    // preheat
    for (int i = 0; i < 100; i++) {
        inf->setInput(0);
        inf->setOutput();
        inf->run();
        inf->getOutput();
    }

    start = clock();

    for (int i = 0; i < run_count; i++) {
        inf->setInput(0);
        inf->setOutput();
        inf->run();
        inf->getOutput();
    }

    end = clock();
    {
        double ms = (double)(end - start)/1000;
        printf("[%s]Use time is: %f\n", kmodel_name, ms / run_count);
    }
    delete inf;
    
    return 0;
}