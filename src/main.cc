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
#include<vector>

#include "k510_drm.h"
#include "media_ctl.h"
#include <linux/videodev2.h>
#include <ctime>
#include "yolofastv2.h"
#include "ssdmobilenetv1.h"

struct video_info dev_info[2];
std::mutex mtx;
uint8_t drm_bufs_index = 0;
uint8_t drm_bufs_argb_index = 0;
struct drm_buffer *fbuf_yuv, *fbuf_argb;
int obj_cnt;

std::atomic<bool> quit(true);

void fun_sig(int sig)
{
    if(sig == SIGINT)
    {
        quit.store(false);
    }
}

int main(int argc, char *argv[])
{
    // std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;
    // if (argc != 2)
    // {
    //     std::cerr << "Usage: " << argv[0] << " <.kmodel> <image_file>" << std::endl;
    //     return -1;
    // }

    // kmodel_name = argv[1];

    struct sigaction sa;
    memset( &sa, 0, sizeof(sa) );
    sa.sa_handler = fun_sig;
    sigfillset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    drm_init();

    YoloFastV2 *yolofast = nullptr;
    yolofast = new YoloFastV2({352, 352, 3}, {22, 22, 95}, {11, 11, 95});
    yolofast->load_model("./yolofast.kmodel");
    yolofast->prepare_memory();

    // preheat
    for (int i = 0; i < 100; i++) {
        yolofast->set_input(0);
        yolofast->set_output();
        yolofast->run();
        yolofast->get_output();
    }

    clock_t start, end;
    start = clock();

    for (int i = 0; i < 1000; i++) {
        yolofast->set_input(0);
        yolofast->set_output();
        yolofast->run();
        yolofast->get_output();
    }

    end = clock();
    {
        double ms = (double)(end - start)/1000;
        printf("Use time is: %f\n", ms / 1000);
    }
    delete yolofast;

    SSDMobileNetV1 *ssdmobile = nullptr;
    ssdmobile = new SSDMobileNetV1({320, 320, 3}, {1, 2034, 4}, {1, 2034, 91});
    ssdmobile->load_model("./mobilenetssd.kmodel");
    ssdmobile->prepare_memory();

    // preheat
    for (int i = 0; i < 100; i++) {
        ssdmobile->set_input(0);
        ssdmobile->set_output();
        ssdmobile->run();
        ssdmobile->get_output();
    }

    start = clock();

    for (int i = 0; i < 1000; i++) {
        ssdmobile->set_input(0);
        ssdmobile->set_output();
        ssdmobile->run();
        ssdmobile->get_output();
    }

    end = clock();
    {
        double ms = (double)(end - start)/1000;
        printf("Use time is: %f\n", ms / 1000);
    }
    delete ssdmobile;

    for(int i = 0; i < DRM_BUFFERS_COUNT; i++) {
        drm_destory_dumb(&drm_dev.drm_bufs[i]);
    }
    for(int i = 0; i < DRM_BUFFERS_COUNT; i++) {
        drm_destory_dumb(&drm_dev.drm_bufs_argb[i]);
    }
    
    return 0;
}