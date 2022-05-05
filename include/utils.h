/* Copyright 2022 Visiongo Ltd.
 * 
 * @author lijunyu
 * @date 2022/5/4
 * @file utils.h
 */

#ifndef _UTILS
#define _UTILS

#include <cstdint>
#include <string>

#define SHARE_MEMORY_ALLOC          _IOWR('m', 1, unsigned long)
#define SHARE_MEMORY_ALIGN_ALLOC    _IOWR('m', 2, unsigned long)
#define SHARE_MEMORY_FREE           _IOWR('m', 3, unsigned long)
#define SHARE_MEMORY_SHOW           _IOWR('m', 4, unsigned long)
#define SHARE_MEMORY_INVAL_RANGE    _IOWR('m', 5, unsigned long)
#define SHARE_MEMORY_WB_RANGE       _IOWR('m', 6, unsigned long)
#define MEMORY_TEST_BLOCK_SIZE      4096        /* 测试申请的内存空间大小 */
#define MEMORY_TEST_BLOCK_ALIGN     4096        /* 如果需要mmap映射,对齐需要4K的整数倍 */
#define SHARE_MEMORY_DEV            "/dev/k510-share-memory"
#define MAP_MEMORY_DEV              "/dev/mem"

struct share_memory_alloc_align_args {
    uint32_t size;
    uint32_t alignment;
    uint32_t phyAddr;
};

struct data_shape {
    uint32_t weight;
    uint32_t height;
    uint32_t channel;
};

typedef struct ai_worker_args
{
    char* kmodel_path;
    int net_len;
    int valid_width;
    int valid_height;
    float obj_thresh;
    float nms_thresh;
    int is_rgb;
    int enable_profile;
    std::string dump_img_dir;
}ai_worker_args;


#endif // !_UTILS
