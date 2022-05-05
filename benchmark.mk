################################################################################
#
# benchmark
#
################################################################################
ifeq ($(BR2_PACKAGE_BENCHMARK), y)
	BENCHMARK_LOCAL_PATH:= $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

	BENCHMARK_SITE = $(BENCHMARK_LOCAL_PATH)
	BENCHMARK_SITE_METHOD = local

	BENCHMARK_CXXFLAGS = $(TARGET_CXXFLAGS)
	BENCHMARK_CFLAGS = $(TARGET_CFLAGS)

	BENCHMARK_CONF_OPTS += \
			-DCMAKE_CXX_FLAGS="$(BENCHMARK_CXXFLAGS) -I$(STAGING_DIR)/usr/include/opencv4 -I$(STAGING_DIR)/usr/include/libdrm" \
			-DCMAKE_C_FLAGS="$(BENCHMARK_CFLAGS) -I$(STAGING_DIR)/usr/include/opencv4 -I$(STAGING_DIR)/usr/include/libdrm -I$(STAGING_DIR)/usr/include" \
		-DCMAKE_INSTALL_PREFIX="/app/benchmark"

	BENCHMARK_DEPENDENCIES += mediactl_lib nncase_linux_runtime opencv4 libdrm
$(eval $(cmake-package))
endif
