cmake_minimum_required(VERSION 3.2)
project(benchmark C CXX)

# set(DEMO_ROOT "${PROJECT_SOURCE_DIR}/../crb_demo")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-O2 -s")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(SRC ./src/main.cc ./src/yolofastv2.cc ./src/ssdmobilenetv1.cc ./src/inference_kmodel.cc ./drm/k510_drm.c)
# set(bin benchmark)
set(script ./benchmark.sh)
set(input model/Yolo-Fast-V2/yolofast.kmodel model/MobilenetSSD/mobilenetssd.kmodel)

include_directories("./include")
include_directories("./drm")
include_directories(${DEMO_ROOT}/riscv64/include)
include_directories(${DEMO_ROOT}/riscv64/opencv_4.5.1/include/opencv4)
include_directories(${DEMO_ROOT}/riscv64/video_zt)

link_directories(${DEMO_ROOT}/riscv64/lib/)
link_directories(${DEMO_ROOT}/riscv64/opencv_4.5.1/lib)
link_directories(${DEMO_ROOT}/riscv64/video_zt)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} mediactl drm pthread)
target_link_libraries(${PROJECT_NAME} nncase.runtime nncase.rt_modules.k510)
target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio)

install(TARGETS ${PROJECT_NAME} DESTINATION exe)
install(PROGRAMS ${script} DESTINATION shell)
install(PROGRAMS ${input} DESTINATION exe)
