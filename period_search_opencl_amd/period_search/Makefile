# This should work on Linux.  Modify as needed for other platforms.

# READ ME FIRST:
#
# To keep constinstance of the repository and not be relying to a hardcoded paths please create symbolic links that will be outside the path under
# versioning controll and this way will not brake anyones environment setups.
# From your /home/[YOUR_USERNAME] direcory run:
#	ln -s /[OpenSL_SDK_PATH] AMDSKD
#	ln -s /[your_bpinc_path] boinc_src
# or change those paths acordingly to your setup.
#
# Some materials
# https://stackoverflow.com/questions/57361083/gcc-equivalent-of-gs-gl-gy-oi-md-in-msvc

# DO-NOT-CHANGE!
BUILD_DIR = build
BOINC_DIR =../../../boinc_src
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib
OPENCL_PATH =../../../AMDSDK
# OPENCL_PATH = /usr
# OPENCL_LIB_PATH = /opt/amdgpu-pro/lib/x86_64-linux-gnu
PS_COMMON_DIR = ../../period_search_common
PS_COMMON_LIB_DIR = $(PS_COMMON_DIR)/build

ifeq ($(OS),Windows_NT)
    OPENCL_LIB := libOpenCL.dll.a
else
    OPENCL_LIB := libOpenCL.so
endif

# Define compiler and flags
CXX = g++
CXXFLAGS = -march=x86-64 -O3 -s -std=c++2a -w -fstack-protector-strong -fstack-clash-protection -ffunction-sections \
	-I$(BOINC_DIR) \
	-I$(BOINC_LIB_DIR) \
	-I$(BOINC_API_DIR) \
	-I$(PS_COMMON_DIR) \
	-L$(PS_COMMON_LIB_DIR) \
	-I$(OPENCL_PATH)/include \
	-L$(OPENCL_PATH) \
	-L/usr/X11R6/lib \
	-DAMD \
	-DNDEBUG \
	-DCL_HPP_ENABLE_EXCEPTIONS \
	-L.

# Define the target executable
TARGET = $(BUILD_DIR)/period_search_BOINC_linux_amd_opencl_110_x64_Release

# Define the source files
SRC = HelperFunctions.cpp trifac.cpp areanorm.cpp sphfunc.cpp ellfit.cpp ludcmp.cpp lubksb.cpp covsrt.cpp\
	compareResult.cpp Start_OpenCl.cpp VersionInfo.cpp period_search_BOINC.cpp LcHelpersCl.cpp

# Define the object files
OBJ = $(SRC:%.cpp=$(BUILD_DIR)/%.o)

# Default target to build the executable
all: $(BUILD_DIR) $(TARGET)

# Rule to create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to build the static library
$(TARGET): $(OBJ) $(BUILD_DIR)/libstdc++.a $(BOINC_LIB_DIR)/libboinc.a $(BOINC_API_DIR)/libboinc_api.a $(PS_COMMON_LIB_DIR)/libps_common.a $(BUILD_DIR)/$(OPENCL_LIB)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(BUILD_DIR)/libstdc++.a -pthread \
	$(BUILD_DIR)/$(OPENCL_LIB) \
	$(PS_COMMON_LIB_DIR)/libps_common.a \
	$(BOINC_API_DIR)/libboinc_api.a \
	$(BOINC_LIB_DIR)/libboinc.a \
	-lm

# -lm -Xlinker -rpath -m64 .

# Rule to build object files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to create symbolic link for libstdc++.a
$(BUILD_DIR)/libstdc++.a:
	@mkdir -p $(BUILD_DIR)
	ln -s `g++ -print-file-name=libstdc++.a` $(BUILD_DIR)/libstdc++.a

$(BUILD_DIR)/$(OPENCL_LIB):
	@mkdir -p $(BUILD_DIR)
	ln -s $(OPENCL_PATH)/$(OPENCL_LIB) $(BUILD_DIR)/$(OPENCL_LIB)

# Clean rule to remove compiled files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
