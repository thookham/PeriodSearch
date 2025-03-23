//#if !defined INTEL

#if !defined _WIN32
#define CL_TARGET_OPENCL_VERSION 110
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_CL_1_1_DEFAULT_BUILD
// #define CL_API_SUFFIX__VERSION_1_0 CL_API_SUFFIX_COMMON
#define CL_BLOCKING 	CL_TRUE
#else // WIN32
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
// #define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS
typedef unsigned int uint;
#endif



// #include <CL/opencl.hpp>
#include <CL/cl.h>
#include "opencl_helper.h"

#define MINI_CASE_SENSITIVE
#include "ini.h"

// https://stackoverflow.com/questions/18056677/opencl-double-precision-different-from-cpu-double-precision

// TODO:
// <kernel>:2589 : 10 : warning : incompatible pointer types initializing '__generic double *' with an expression of type '__global float *'
// double* dytemp = &CUDA_Dytemp[blockIdx.x];
// ~~~~~~~~~~~~~~~~~~~~~~~~

//#include <vector>
#include <cmath>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <algorithm>
#include <ctime>
#include "boinc_api.h"
#include "mfile.h"

#include "globals.h"
#include "constants.h"
#include "declarations.hpp"
#include "Start_OpenCl.h"
#include "kernels.cpp"


#ifdef _WIN32
#include "boinc_win.h"
//#include <Shlwapi.h>
#else
#endif

#include "Globals_OpenCl.h"
#include <cstddef>
#include <numeric>
#include <arrayHelpers.hpp>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

// NOTE: global to all freq

// cl_platform_id *platforms;
//cl_device_id* devices;
//cl_context contextCpu;
//cl_kernel kernel, kernelDave, kernelSig2wght;

cl_context context;
cl_program binProgram, program;
cl_command_queue queue;

// cl_mem bufCg, bufArea, bufDarea, bufDg, bufFc, bufFs, bufDsph,  bufPleg, bufMmax, bufLmax,  bufX,  bufY, bufZ;
// cl_mem bufSig2iwght, bufDy, bufYmod;
// cl_mem bufDave, bufDyda;
// cl_mem bufD;

cl_mem bufSig;
cl_mem bufTim;
cl_mem bufBrightness;
cl_mem bufEe;
cl_mem bufEe0;
cl_mem bufWeight;
cl_mem bufIa;

cl_kernel kernelClCheckEnd;
cl_kernel kernelCalculatePrepare;
cl_kernel kernelCalculatePreparePole;
cl_kernel kernelCalculateIter1Begin;
cl_kernel kernelCalculateIter1Mrqcof1Start;
cl_kernel kernelCalculateIter1Mrqcof1Matrix;
cl_kernel kernelCalculateIter1Mrqcof1Curve1;
cl_kernel kernelCalculateIter1Mrqcof1Curve2;
cl_kernel kernelCalculateIter1Mrqcof1Curve1Last;
cl_kernel kernelCalculateIter1Mrqcof1End;
cl_kernel kernelCalculateIter1Mrqmin1End;
cl_kernel kernelCalculateIter1Mrqcof2Start;
cl_kernel kernelCalculateIter1Mrqcof2Matrix;
cl_kernel kernelCalculateIter1Mrqcof2Curve1;
cl_kernel kernelCalculateIter1Mrqcof2Curve2;
cl_kernel kernelCalculateIter1Mrqcof2Curve1Last;
cl_kernel kernelCalculateIter1Mrqcof2End;
cl_kernel kernelCalculateIter1Mrqmin2End;
cl_kernel kernelCalculateIter2;
cl_kernel kernelCalculateFinishPole;

size_t CUDA_grid_dim;
//int CUDA_grid_dim_precalc;

// NOTE: global to one thread
#if !defined _WIN32
// TODO: Check compiler version. If  GCC 4.8 or later is used switch to 'alignas(n)'.
#if defined (INTEL)
cl_uint faOptimizedSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)aligned_alloc(4096, faOptimizedSize);
#else
// freq_context* Fa; // __attribute__((aligned(8)));
cl_uint faSize = (sizeof(freq_context) / 128 + 1) * 128;
auto Fa = (freq_context*)aligned_alloc(128, faSize);
// freq_context* Fa __attribute__((aligned(8))) = static_cast<freq_context*>(malloc(sizeof(freq_context)));
#endif
#else // WIN32

#if defined INTEL
cl_uint faOptimizedSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)_aligned_malloc(faOptimizedSize, 4096);
#elif defined AMD
//cl_uint faSize = sizeof(freq_context);
//alignas(8) freq_context* Fa;
cl_uint faSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
auto Fa = (freq_context*)_aligned_malloc(faSize, 128);
#elif defined NVIDIA
#endif

#endif

unsigned char* GetKernelBinaries(cl_program binProgram, const size_t binary_size)
{
    auto binary = new unsigned char[binary_size];
    cl_int err = clGetProgramInfo(binProgram, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, NULL);

    return binary;
}

static cl_int SaveKernelsToBinary(const char *kernelFileName)
{
    size_t binary_size;
    clGetProgramInfo(binProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
    auto binary = GetKernelBinaries(binProgram, binary_size);

    FILE* fp = fopen(kernelFileName, "wb+");
    if (!fp) {
        cerr << "Error while saving kernels binary file." << endl;
        delete[] binary;

        return 1;
    }

    fwrite(binary, binary_size, 1, fp);
    fclose(fp);
    delete[] binary;

    return 0;
}

cl_int ClBuildProgramWithSource(char *name, cl_device_id device, char deviceName[1024], const char *kernelFileName)
{
    cl_int err_num;
    binProgram = clCreateProgramWithSource(context, 1, &ocl_src_kernelSource, NULL, &err_num);
    if (!binProgram || err_num != CL_SUCCESS)
    {
        cerr << "Error: Failed to create compute program! " << cl_error_to_str(err_num) << " (" << err_num << ")" << endl;
        return EXIT_FAILURE;
    }

    char options[]{ "-w" }; // char options[]{ "-Werror" };
#if defined (AMD)
    err_num = clBuildProgram(binProgram, 1, &device, options, NULL, NULL); // "-Werror -cl-std=CL1.1"
    if (err_num != CL_SUCCESS)
    {
        GetProgramBuildInfo(binProgram, device, name, deviceName, err_num);
        return err_num;
    }
#elif defined (NVIDIA)
        binProgram.build(devices, "-D NVIDIA -w -cl-std=CL1.2"); // "-w" "-Werror"
#elif defined (INTEL)
        binProgram.build(devices, "-D INTEL -cl-std=CL1.2");
#endif

    //#if defined (NDEBUG)
    //        std::ifstream fs(kernelFileName);
    //        bool kernelExist = fs.good();
    //        if (kernelExist) {
    //            std::remove(kernelSourceFile.c_str());
    //        }
    //#endif

    size_t len;
    err_num = clGetProgramBuildInfo(binProgram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    auto buildlog = new char[len];
    err_num = clGetProgramBuildInfo(binProgram, device, CL_PROGRAM_BUILD_LOG, len, buildlog, NULL);

    cl_build_status buildStatus;
    err_num =
        clGetProgramBuildInfo(binProgram, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);

    std::string buildlogStr = buildlog;
    if (buildStatus == 0)
    {
        buildlogStr.append("OK");
    }

    cerr << "Binary build log for " << deviceName << ":" << std::endl << buildlogStr << " (" << buildStatus << ")" << endl;
    delete[] buildlog;
    err_num = SaveKernelsToBinary(kernelFileName);

    return err_num;
}

cl_int ClBuildProgramWithBinary(const char *name, const cl_device_id device, const uint strBufSize, char deviceName[1024], const char *kernelFileName)
{
    cl_int err_num;
    try
    {
        std::ifstream file(kernelFileName, std::ios::binary | std::ios::in | std::ios::ate);
        size_t binary_size = file.tellg();
        file.seekg(0, std::ios::beg);
        auto binary = new char[binary_size];
        file.read(binary, static_cast<std::streamsize>(binary_size));
        file.close();

        cl_int binary_status;
        const auto binaryConstPtr = const_cast<const unsigned char **>(reinterpret_cast<unsigned char**>(&binary));
        program = clCreateProgramWithBinary(context, 1, &device, &binary_size, binaryConstPtr, &binary_status, &err_num);

        //char options[]{ "-Werror" };
        char options[]{ "-w" };
#if defined (AMD)
        err_num = clBuildProgram(program, 1, &device, options, NULL, NULL); // "-Werror -cl-std=CL1.1" "-g -x cl -cl-std=CL1.2 -Werror"
#elif defined (NVIDIA)
        program.build(devices); //, "-D NVIDIA -w -cl-std=CL1.2"); // "-Werror" "-w"
#elif defined (INTEL)
        program.build(devices, "-D INTEL -cl-std=CL1.2");
#endif
        if (err_num != CL_SUCCESS)
        {
            GetProgramBuildInfo(program, device, name, deviceName, err_num);
            return err_num;
        }

        queue = clCreateCommandQueue(context, device, 0, &err_num);
        if (err_num != CL_SUCCESS) {
            std::cerr << " Error creating queue: " << cl_error_to_str(err_num) << "(" << err_num << ")\n";
            return err_num;
        }

        delete[] binary;

        char* buildlog = new char[strBufSize];
        err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, strBufSize, buildlog, NULL);
        cl_build_status buildStatus;
        err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, NULL);

#if _DEBUG
#if CL_TARGET_OPENCL_VERSION > 110
        size_t bufSize;
        size_t numKernels;
        err_num = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(numKernels), &numKernels, NULL);
        err_num = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &bufSize);
        auto kernelNames = new char[bufSize];
        err_num = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, bufSize, kernelNames, NULL);
        cerr << "Kernels: " << numKernels << endl;
        cerr << "Kernel names: " << endl << kernelNames << endl;
        delete[] kernelNames;
#endif
        char buildOptions[1024];
        err_num = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, sizeof(buildOptions), buildOptions, NULL);
        std::cerr << "Build options: " << buildOptions << std::endl;
#endif
        std::string buildlogStr = buildlog;
        delete[] buildlog;
        if (buildStatus == 0)
        {
            buildlogStr.append("OK");
        }

        err_num = clGetDeviceInfo(device, CL_DEVICE_BOARD_NAME_AMD, sizeof(deviceName), &deviceName, NULL);
        cerr << "Program build log for " << deviceName << ":" << std::endl << buildlogStr << " (" << buildStatus << ")" << endl;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Caught runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return  -1;
    }

    return 0;
}

cl_int ClPrepare(cl_int deviceId, cl_double* beta_pole, cl_double* lambda_pole, cl_double* par, cl_double lcoef, cl_double a_lamda_start, cl_double a_lamda_incr,
    std::vector<std::vector<double>>& ee, std::vector<std::vector<double>>& ee0, std::vector<double>& tim,
    cl_double Phi_0, cl_int checkex, cl_int ndata, struct globals& gl)
{
#if !defined _WIN32

#else
#ifndef INTEL
    //Fa = static_cast<freq_context*>(malloc(sizeof(freq_context)));
#else

#endif
#endif
    bool rebuildBinaries = false;
    mINI::INIFile file("settings.ini");
    mINI::INIStructure ini;

    if(!file.read(ini))
    {
        file.generate(ini, true);
    }

    std::string &savedKernelsHashString = ini["kernels"]["hash"];
    std::string savedKernelsHash = savedKernelsHashString.empty() ? "0" : savedKernelsHashString;

    if (savedKernelsHash != std::string(kernel_hash))
    {
        ini["kernels"]["hash"] = std::string(kernel_hash);
        rebuildBinaries = true;
        file.write(ini, true);
    }

    //try {
    cl_int err_num;
    cl_uint num_platforms_available;
    err_num = clGetPlatformIDs(0, NULL, &num_platforms_available);

    auto platforms = new cl_platform_id[num_platforms_available];
    err_num = clGetPlatformIDs(num_platforms_available, platforms, NULL);
    cl_platform_id platform = nullptr;

#if defined AMD
    auto name = new char[1024];
    auto vendor = new char[1024];
#else
    cl::STRING_CLASS name;
    cl::STRING_CLASS vendor;
#endif

    //for (iter = platforms.begin(); iter != platforms.end(); ++iter)
    for (uint i = 0; i < num_platforms_available; i++)
    {
        platform = platforms[i];
        err_num = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 1024, name, NULL);
        err_num = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 1024, vendor, NULL);
        if (!strcmp(name, "Clover")) {
            continue;
        }
#if defined (AMD)
        if (!strcmp(vendor, "Advanced Micro Devices, Inc."))
        {
            break;
        }
#elif defined (NVIDIA)
        if (!strcmp(vendor, "NVIDIA Corporation"))
        {
            break;
        }
#elif defined (INTEL)
        if (!strcmp(vendor, "Intel(R) Corporation"))
        {
            break;
        }
#endif
        if (!strcmp(name, "rusticl"))
        {
            break;
        }
    }

    delete[] platforms;

    std::cerr << "Platform name: " << name << endl;
    std::cerr << "Platform vendor: " << vendor << endl;

    // Detect OpenCL devices
    cl_uint numDevices;
    err_num = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

    cl_device_id* devices = new cl_device_id[numDevices];
    err_num = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);


    if (numDevices < 1)
    {
        cerr << "No GPU device found for platform " << vendor << "(" << name << ")" << endl;
        return (1);
    }

    delete[] vendor;

    cl_device_id device = devices[deviceId];
    delete[] devices;

    // Create an OpenCL context for the chosen device
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    context = clCreateContext(properties, 1, &device, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS) {
        cerr << "Error: Failed to create a device group! " << cl_error_to_str(err_num) << " (" << err_num << ")" << endl;
        return EXIT_FAILURE;
    }

    const uint strBufSize = 1024;
    char deviceVendor[strBufSize];
    char driverVersion[strBufSize];

#if !defined _WIN32
#if defined INTEL
#else
    char deviceName[strBufSize]; // Another AMD thing... Don't ask
    err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), &deviceName, NULL);
#endif
#else
#if defined INTEL
    size_t nameSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &nameSize);
    //auto deviceNameChars = new char[nameSize];
    auto deviceName = (char*)malloc(nameSize);
    //char deviceName[strBufSize];
    err_num = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char) * nameSize, &deviceName, NULL);
#else
    char deviceName[strBufSize]; // Another AMD thing... Don't ask
    err_num = clGetDeviceInfo(device, CL_DEVICE_BOARD_NAME_AMD, sizeof(deviceName), &deviceName, NULL);
#endif

#endif

    std::string &savedDeviceName = ini["device"]["name"];
    if(strcmp(savedDeviceName.c_str(), deviceName) != 0)
    {
        ini["device"]["name"] = deviceName;
        rebuildBinaries = true;
        file.write(ini, true);
    }

    err_num = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor), &deviceVendor, NULL);
    err_num = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driverVersion), &driverVersion, NULL);

    char openClVersion[strBufSize];
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, strBufSize, openClVersion, NULL);

    char clDeviceVersion[strBufSize];
    clGetDeviceInfo(device, CL_DEVICE_VERSION, strBufSize, clDeviceVersion, NULL);

    // cl_device_exec_capabilities
    char clDeviceExtensionCapabilities[strBufSize];
    err_num = clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES, strBufSize, &clDeviceExtensionCapabilities, NULL);

    cl_device_fp_config deviceDoubleFpConfig;
    err_num = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &deviceDoubleFpConfig, NULL);

    cl_ulong clDeviceGlobalMemSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &clDeviceGlobalMemSize, NULL);

    cl_ulong clDeviceLocalMemSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &clDeviceLocalMemSize, NULL);

    uint clDeviceMaxConstantArgs;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(uint), &clDeviceMaxConstantArgs, NULL);

    unsigned long long clDeviceMaxConstantBufferSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(unsigned long long), &clDeviceMaxConstantBufferSize, NULL);

    size_t clDeviceMaxParameterSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &clDeviceMaxParameterSize, NULL);

    unsigned long long clDeviceMaxMemAllocSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(unsigned long long), &clDeviceMaxMemAllocSize, NULL);

    cl_ulong clGlobalMemory;
    err_num = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &clGlobalMemory, NULL);
    cl_ulong globalMemory = clGlobalMemory / 1048576;

    cl_uint msCount;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &msCount, NULL);

    uint block;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_SAMPLERS, sizeof(uint), &block, NULL);

    cl_uint baseAddrAlign;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &baseAddrAlign, NULL);

    cl_uint minDataTypeAlignSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(cl_uint), &minDataTypeAlignSize, NULL);

    size_t extSize;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &extSize);
    auto deviceExtensions = new char[extSize];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extSize, deviceExtensions, NULL);

    size_t devMaxWorkGroupSize;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &devMaxWorkGroupSize, NULL);

    cl_uint devMaxWorkItemDims;
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &devMaxWorkItemDims, NULL);

    size_t devWorkItemSizes[3];
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(devWorkItemSizes), &devWorkItemSizes, NULL);

    cerr << "OpenCL device C version: " << openClVersion << " | " << clDeviceVersion << endl;
    cerr << "OpenCL device Id: " << deviceId << endl;
    string sufix = "MB";
    if (globalMemory > 1024) {
        globalMemory /= 1024;
        sufix = "GB";
    }
    cerr << "OpenCL device name: " << deviceName << " " << globalMemory << sufix << endl;
    cerr << "Device driver version: " << driverVersion << endl;
    cerr << "Multiprocessors: " << msCount << endl;
    cerr << "Max work item dimensions: " << devMaxWorkItemDims << endl;
#ifdef _DEBUG
    cerr << "Debug info:" << endl;
    cerr << "CL_DEVICE_EXTENSIONS: " << deviceExtensions << endl;
    cerr << "CL_DEVICE_GLOBAL_MEM_SIZE: " << globalMemory << sufix << endl;
    cerr << "CL_DEVICE_LOCAL_MEM_SIZE: " << clDeviceLocalMemSize << " B" << endl;
    cerr << "CL_DEVICE_MAX_CONSTANT_ARGS: " << clDeviceMaxConstantArgs << endl;
    cerr << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << clDeviceMaxConstantBufferSize << " B" << endl;
    cerr << "CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << baseAddrAlign << endl;
    cerr << "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: " << minDataTypeAlignSize << endl;
    cerr << "CL_DEVICE_MAX_PARAMETER_SIZE: " << clDeviceMaxParameterSize << " B" << endl;
    cerr << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << clDeviceMaxMemAllocSize << " B" << endl;
    cerr << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << devMaxWorkGroupSize << endl;
    cerr << "CL_DEVICE_MAX_WORK_ITEM_SIZES: {";

    for (size_t work_item_dim = 0; work_item_dim < devMaxWorkItemDims; work_item_dim++) {
        if (work_item_dim > 0) cerr << ", ";
        cerr << (long int)devWorkItemSizes[work_item_dim];
    }
    cerr << "}" << endl;

#endif
    // cl_khr_fp64 || cl_amd_fp64
    bool isFp64 = string(deviceExtensions).find("cl_khr_fp64") != std::string::npos
        || string(deviceExtensions).find("cl_amd_fp64") != std::string::npos;

    delete[] deviceExtensions;

    bool doesNotSupportsFp64 = !isFp64;
    if (doesNotSupportsFp64)
    {
        fprintf(stderr, "Double precision floating point not supported by OpenCL implementation on current device. Exiting...\n");
        return (1);
    }

    auto SMXBlock = 32;
    //CUDA_grid_dim = msCount * SMXBlock; //  24 * 32
    //CUDA_grid_dim = 8 * 32 = 256; 6 * 32 = 192
    CUDA_grid_dim = msCount * SMXBlock; // 256 (RX 550), 384 (1050Ti), 1536 (Nvidia GTX1660Ti), 768 (Intel Graphics HD)
    std::cerr << "Resident blocks per multiprocessor: " << SMXBlock << endl;
    std::cerr << "Grid dim: " << CUDA_grid_dim << " = " << msCount << " * " << SMXBlock << endl;
    std::cerr << "Block dim: " << BLOCK_DIM << endl;

    //Global parameters
    bufTim = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tim.size() * sizeof(double), tim.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufTim, CL_BLOCKING, 0, tim.size() * sizeof(double), tim.data(), 0, NULL, NULL);

    auto flattened_ee = flatten2Dvector<double>(ee);
    bufEe = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, flattened_ee.size() * sizeof(double), flattened_ee.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufEe, CL_BLOCKING, 0, flattened_ee.size() * sizeof(double), flattened_ee.data(), 0, NULL, NULL);

    auto flattened_ee0 = flatten2Dvector<double>(ee0);
    bufEe0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, flattened_ee0.size() * sizeof(double), flattened_ee0.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufEe0, CL_BLOCKING, 0, flattened_ee0.size() * sizeof(double), flattened_ee0.data(), 0, NULL, NULL);

    bufWeight = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gl.Weight.size() * sizeof(double), gl.Weight.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufWeight, CL_BLOCKING, 0, gl.Weight.size() * sizeof(double), gl.Weight.data(), 0, NULL, NULL);

    memcpy((*Fa).beta_pole, beta_pole, sizeof(cl_double) * (N_POLES + 1));
    memcpy((*Fa).lambda_pole, lambda_pole, sizeof(cl_double) * (N_POLES + 1));
    memcpy((*Fa).par, par, sizeof(cl_double) * 4);
    //memcpy((*Fa).tim, tim, sizeof(double) * (MAX_N_OBS + 1));
    //memcpy((*Fa).ee, ee, (ndata + 1) * 3 * sizeof(cl_double));
    //memcpy((*Fa).ee0, ee0, (ndata + 1) * 3 * sizeof(cl_double));
    //memcpy((*Fa).Weight, weight, (ndata + 3 + 1) * sizeof(double));

    (*Fa).maxLcPoints = gl.maxLcPoints;
    (*Fa).cl = lcoef;
    (*Fa).logCl = log(lcoef);
    (*Fa).Alamda_incr = a_lamda_incr;
    (*Fa).Alamda_start = a_lamda_start;
    (*Fa).Mmax = m_max;
    (*Fa).Lmax = l_max;
    (*Fa).Phi_0 = Phi_0;

    //string kernelSourceFile = "kernelSource.cl";
    const char* kernelFileName = "kernels.bin";

    std::ifstream f(kernelFileName);
    bool kernelExist = f.good();

    if (!kernelExist || rebuildBinaries)
    {
        std::string action = rebuildBinaries ? "Rebuilding" : "Building";
        std::cerr << action << " program" << std::endl;
        err_num = ClBuildProgramWithSource(name, device, deviceName, kernelFileName);
        if (err_num != CL_SUCCESS) return err_num;
    }
    else
    {
        std::cerr << "Kernels binary exists." << std::endl;
    }

    err_num = ClBuildProgramWithBinary(name, device, strBufSize, deviceName, kernelFileName);
    if (err_num != CL_SUCCESS) return err_num;

    delete[] name;

#pragma region Kernel creation
    cl_int kerr;
    try
    {
        kernelClCheckEnd = clCreateKernel(program, "ClCheckEnd", &kerr);
        kernelCalculatePrepare = clCreateKernel(program, string("ClCalculatePrepare").c_str(), &kerr);
        kernelCalculatePreparePole = clCreateKernel(program, string("ClCalculatePreparePole").c_str(), &kerr);
        kernelCalculateIter1Begin = clCreateKernel(program, string("ClCalculateIter1Begin").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Start = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Start").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Matrix = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Matrix").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Curve1 = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Curve1").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Curve2 = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Curve2").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1Curve1Last = clCreateKernel(program, string("ClCalculateIter1Mrqcof1Curve1Last").c_str(), &kerr);
        kernelCalculateIter1Mrqcof1End = clCreateKernel(program, string("ClCalculateIter1Mrqcof1End").c_str(), &kerr);
        kernelCalculateIter1Mrqmin1End = clCreateKernel(program, string("ClCalculateIter1Mrqmin1End").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Start = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Start").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Matrix = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Matrix").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Curve1 = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Curve1").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Curve2 = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Curve2").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2Curve1Last = clCreateKernel(program, string("ClCalculateIter1Mrqcof2Curve1Last").c_str(), &kerr);
        kernelCalculateIter1Mrqcof2End = clCreateKernel(program, "ClCalculateIter1Mrqcof2End", &kerr);
        kernelCalculateIter1Mrqmin2End = clCreateKernel(program, "ClCalculateIter1Mrqmin2End", &kerr);
        kernelCalculateIter2 = clCreateKernel(program, "ClCalculateIter2", &kerr);
        kernelCalculateFinishPole = clCreateKernel(program, "ClCalculateFinishPole", &kerr);
    }
    catch (Error& e)
    {
        cerr << "Error creating kernel: \"" << cl_error_to_str(e.err()) << "\"(" << e.err() << ") - " << e.what() << " | " << cl_error_to_str(kerr) <<
            " (" << kerr << ")" << std::endl;
        cout << "Error while creating kernel. Check stderr.txt for details." << endl;
        return(4);
    }
#pragma endregion

#ifndef CL_PROGRAM_NUM_KERNELS
#define CL_PROGRAM_NUM_KERNELS                      0x1167
#define CL_PROGRAM_KERNEL_NAMES                     0x1168
#endif

//#if defined _DEBUG
    // size_t numKernels;
    // err = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t), &numKernels, NULL);
    // size_t kernelNamesSize;
    // err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &kernelNamesSize);
    // char kernelNamesChars[kernelNamesSize];
    // // auto kernelNames = (char*) malloc(numKernels);
    // err = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, sizeof(kernelNamesChars), kernelNamesChars, NULL);
    // cerr << "Kernel names: " << kernelNamesChars << endl;
    // std::vector<char*> kernelNames;
    // char* kernel_chars = strtok(kernelNamesChars, ";");
    // while(kernel_chars)
    // {
    //     kernelNames.push_back(kernel_chars);
    //     kernel_chars = strtok(NULL, ";");
    // }

    // cerr << "Prefered kernel work group size - kernel | size:" << endl;
    // size_t preferedWGS[numKernels];
    // for(int k = 0; k < numKernels; k++){
    // 	cl_kernel kernel = clCreateKernel(program, kernelNames[k], &kerr);
    // 	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferedWGS[k], NULL);
    // 	cerr << kernelNames[k] << " | " << preferedWGS[k] << endl;
    // }
//#endif

    size_t preferedWGS;
    err_num = clGetKernelWorkGroupInfo(kernelCalculateIter1Mrqmin1End, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferedWGS, NULL);
    if (err_num != CL_SUCCESS) {
        std::cerr << " Error in clGetKernelWorkGroupInfo: " << cl_error_to_str(err_num) << "(" << err_num << ")\n";
        return(1);
    }
    cerr << "Preferred kernel work group size multiple: " << preferedWGS << endl;

    if (CUDA_grid_dim > devMaxWorkGroupSize) {
        CUDA_grid_dim = devMaxWorkGroupSize;
        cerr << "Setting Grid Dim to " << CUDA_grid_dim << endl;
    }

    return 0;
}

void PrintFreqResult(const int maxItterator, void* pcc, void* pfr)
{
    for (auto l = 0; l < maxItterator; l++)
    {
        mfreq_context* CC = &((mfreq_context*)pcc)[l];
        freq_result* FR = &((freq_result*)pfr)[l];
        cout << "freq[" << l << "] = " << (*CC).freq << " | la_best[" << l << "] = " << (*FR).la_best << std::endl;
    }
}

template <typename T>
void PrepareBufferFromFlatenArray(cl_mem &clBuf, size_t arraySize, size_t alignment)
{
    auto size = arraySize * sizeof(T);
    size_t padded_size = (size + alignment - 1) & ~(alignment - 1);
    #if defined _WIN32
      auto pBuf = static_cast<T *>(_aligned_malloc(padded_size, alignment));
    #else
      auto pBuf = static_cast<T *>(aligned_alloc(alignment, padded_size));
    #endif
    memset(pBuf, 0, padded_size);
    cl_int err = 0;
    clBuf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, pBuf, &err);
    clEnqueueWriteBuffer(queue, clBuf, CL_BLOCKING, 0, size, pBuf, 0, NULL, NULL);

    #if defined _WIN32
      _aligned_free(pBuf);
    #else
      free(pBuf);
    #endif
}

cl_int ClPrecalc(cl_double freq_start, cl_double freq_end, cl_double freq_step, cl_double stop_condition, cl_int n_iter_min, cl_double* conw_r,
    cl_int ndata, std::vector<int>& ia, cl_int* ia_par, cl_int* new_conw, std::vector<double>& cg_first, std::vector<double>& sig, cl_int Numfac,
    std::vector<double>& brightness, struct globals& gl, cl_double lcoef, int Ncoef)
{
    cl_int max_test_periods, iC;
    cl_int theEnd = -100;
    double sum_dark_facet, ave_dark_facet;
    cl_int i, n, m;
    cl_int n_iter_max;
    double iter_diff_max;
    auto n_max = static_cast<int>((freq_start - freq_end) / freq_step) + 1;
    auto r = 0;
    cl_int isPrecalc = 1;

    max_test_periods = 10;
    sum_dark_facet = 0.0;
    ave_dark_facet = 0.0;

    if (n_max < max_test_periods)
        max_test_periods = n_max;

    for (i = 1; i <= n_ph_par; i++)
    {
        ia[n_coef + 3 + i] = ia_par[i];
    }

    n_iter_max = 0;
    iter_diff_max = -1;
    if (stop_condition > 1)
    {
        n_iter_max = (int)stop_condition;
        iter_diff_max = 0;
        n_iter_min = 0; /* to not overwrite the n_iter_max value */
    }
    if (stop_condition < 1)
    {
        n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
        iter_diff_max = stop_condition;
    }

    cl_int err = 0;

    (*Fa).conw_r = *conw_r;
    (*Fa).Ncoef = n_coef; //Ncoef;
    (*Fa).Nphpar = n_ph_par;
    (*Fa).Numfac = Numfac;
    m = Numfac + 1;
    (*Fa).Numfac1 = m;
    (*Fa).ndata = ndata;
    (*Fa).Is_Precalc = isPrecalc;

    bufBrightness = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, brightness.size() * sizeof(double), brightness.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufBrightness, CL_BLOCKING, 0, brightness.size() * sizeof(double), brightness.data(), 0, NULL, NULL);

    bufSig = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sig.size() * sizeof(double), sig.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufSig, CL_BLOCKING, 0, sig.size() * sizeof(double), sig.data(), 0, NULL, NULL);

    bufIa = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ia.size() * sizeof(int), ia.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufIa, CL_BLOCKING, 0, ia.size() * sizeof(int), ia.data(), 0, NULL, NULL);

    memcpy((*Fa).Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
    memcpy((*Fa).Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    memcpy((*Fa).Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    memcpy((*Fa).Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
    memcpy((*Fa).Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
    memcpy((*Fa).Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
    //memcpy((*Fa).Brightness, brightness, (ndata + 1) * sizeof(double));		// sizeof(double)* (MAX_N_OBS + 1));
    //memcpy((*Fa).Sig, sig, (ndata + 1) * sizeof(double));							// sizeof(double)* (MAX_N_OBS + 1));
    //memcpy((*Fa).ia, ia, sizeof(cl_int) * (MAX_N_PAR + 1));

    /* number of fitted parameters */
    cl_int lmfit = 0;
    cl_int llastma = 0;
    cl_int llastone = 1;
    cl_int ma = n_coef + 5 + n_ph_par;
    for (m = 1; m <= ma; m++)
    {
        if (ia[m])
        {
            lmfit++;
            llastma = m;
        }
    }

    llastone = 1;
    for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
    {
        if (!ia[m]) break;
        llastone = m;
    }

    //(*Fa).Ncoef = n_coef;
    (*Fa).ma = ma;
    (*Fa).Mfit = lmfit;

    m = lmfit + 1;
    (*Fa).Mfit1 = m;

    (*Fa).lastone = llastone;
    (*Fa).lastma = llastma;

    m = ma - 2 - n_ph_par;
    (*Fa).Ncoef0 = m;

    size_t CUDA_grid_dim_precalc = CUDA_grid_dim;
    if (max_test_periods < CUDA_grid_dim_precalc)
    {
        CUDA_grid_dim_precalc = max_test_periods;
    }

    /* totalWorkItems = CUDA_grid_dim_precalc * BLOCK_DIM */
    size_t totalWorkItems = CUDA_grid_dim_precalc * BLOCK_DIM;

    m = (Numfac + 1) * (n_coef + 1);
    (*Fa).Dg_block = m;

#if defined (INTEL)
    auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
#else
    cl_mem cgFirst = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cg_first.size() * sizeof(double), cg_first.data(), &err);
    clEnqueueWriteBuffer(queue, cgFirst, CL_BLOCKING, 0, cg_first.size() * sizeof(double), cg_first.data(), 0, NULL, NULL);
#endif

#if !defined _WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    // cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    // auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);

    // cl_uint pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    // void* pcc = reinterpret_cast<mfreq_context*>(malloc(pccSize));

    // auto pcc __attribute__((aligned(8))) = new mfreq_context[CUDA_grid_dim_precalc];
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
    // auto mcc __attribute__((aligned(8))) = new mfreq_context[CUDA_grid_dim_precalc];

    // cl_uint pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    // auto memPcc = (mfreq_context *)aligned_alloc(128, pccSize);
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pccSize, pcc, err);

    // cl_uint pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    // auto pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc) / 128 + 1) * 128;
    auto pcc = (mfreq_context*)aligned_alloc(128, pccSize);

    // auto pcc = queue.enqueueMapBuffer(CUDA_MCC2, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, pccSize, NULL, NULL, err);
    // queue.flush();
    // void* pcc = clEnqueueMapBuffer(queue, CUDA_MCC2, CL_BLOCKING, CL_MAP_WRITE, 0, pccSize, 0, NULL, NULL, &err);

#elif NVIDIA
    int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#else // WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    auto pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim_precalc) / 128 + 1) * 128;
    auto pcc = (mfreq_context*)_aligned_malloc(pccSize, 128);
#elif NVIDIA
    int pccSize = CUDA_grid_dim_precalc * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim_precalc];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#endif

    cl_mem bufJpScale;
    cl_mem bufJpDphp1;
    cl_mem bufJpDphp2;
    cl_mem bufJpDphp3;
    cl_mem bufE1;
    cl_mem bufE2;
    cl_mem bufE3;
    cl_mem bufE01;
    cl_mem bufE02;
    cl_mem bufE03;
    cl_mem bufDe;
    cl_mem bufDe0;
    cl_mem bufDytemp;
    cl_mem bufYtemp;

    size_t commonSize = CUDA_grid_dim_precalc * (gl.maxLcPoints + 1);
    PrepareBufferFromFlatenArray<double>(bufJpScale, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufJpDphp1, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufJpDphp2, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufJpDphp3, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE1, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE2, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE3, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE01, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE02, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE03, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufYtemp, commonSize, 128);

    size_t deSize = commonSize * 4 * 4; // double de[POINTS_MAX + 1][4][4];
    PrepareBufferFromFlatenArray<double>(bufDe, deSize, 128);
    PrepareBufferFromFlatenArray<double>(bufDe0, deSize, 128);

    size_t dytempSize = commonSize * (MAX_N_PAR + 1); // double dytemp[(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
    PrepareBufferFromFlatenArray<double>(bufDytemp, dytempSize, 128);

    // NOTE: NOTA BENE - In contrast to Cuda, where global memory is zeroed by itself, here we need to initialize the values in each dimension. GV-26.09.2020
    // <<<<<<<<<<<
    for (m = 0; m < CUDA_grid_dim_precalc; m++)
    {
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Area), MAX_N_FAC + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Dg), (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].alpha), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].covar), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].beta), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].da), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].atry), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].dave), MAX_N_PAR + 1, 0.0);
        //std::fill_n(std::begin(((mfreq_context*)pcc)[m].dytemp), (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
        //std::fill_n(std::begin(((mfreq_context*)pcc)[m].ytemp), POINTS_MAX + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_big), BLOCK_DIM, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_icol), BLOCK_DIM, 0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_irow), BLOCK_DIM, 0);
        //pcc[m].conw_r = 0.0;
        ((mfreq_context*)pcc)[m].icol = 0;
        ((mfreq_context*)pcc)[m].pivinv = 0;
    }

#if defined (INTEL)
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#elif defined AMD
    cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error creating OpenCL buffer: %d\n", err);
    }
    err = clEnqueueWriteBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error creating OpenCL buffer: %d\n", err);
    }
#elif defined NVIDIA
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
#endif


#if !defined _WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    // int faSize = sizeof(freq_context);
    // cl_int faSize = sizeof(freq_context);
    // cl_uint faSize = ((sizeof(freq_context) - 1) / 64 + 1) * 64;
    // auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, err);
    // auto pFa = queue.enqueueMapBuffer(CUDA_CC, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, faSize);

    // auto memFa = (freq_context*)aligned_alloc(128, faSize);
    // cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, &err);
    // void* pFa = clEnqueueMapBuffer(queue, CUDA_CC, CL_BLOCKING, CL_MAP_WRITE, 0, faSize, 0, NULL, NULL, &err);
    // memcpy(pFa, Fa, faSize);
    // clEnqueueUnmapMemObject(queue, CUDA_CC, pFa, 0, NULL, NULL);
    // clFlush(queue);
    auto pFa = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, pFa, &err);
    clEnqueueWriteBuffer(queue, CUDA_CC, CL_BLOCKING, 0, faSize, Fa, 0, NULL, NULL);

#endif
#else // WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    auto pFa = (freq_context*)_aligned_malloc(faSize, 128);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, pFa, &err);
    clEnqueueWriteBuffer(queue, CUDA_CC, CL_BLOCKING, 0, faSize, Fa, 0, NULL, NULL);
#endif
#endif

#if !defined _WIN32
    auto pFb = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, pFb, &err);
#else
    auto pFb = (freq_context*)_aligned_malloc(faSize, 128);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, pFb, &err);
#endif

#if defined (INTEL)
    auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int), &theEnd, err);
#else
    cl_mem CUDA_End = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(theEnd), &theEnd, &err);
    err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);

#endif


#if !defined _WIN32
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    // cl_int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    // cl_uint frSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    // void *memIn = (void *)aligned_alloc(8, frSize);
    // void *memIn = (void *)aligned_alloc(8, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    // void* pfr;

    // void* pfr = reinterpret_cast<freq_result*>(malloc(frSize));
    // void *pfr = (void *)aligned_alloc(8, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, CUDA_grid_dim_precalc * sizeof(freq_result), pfr, err);
    // auto memFr __attribute__((aligned(8))) = new freq_result[CUDA_grid_dim_precalc];
    // auto memFr = (freq_result *)aligned_alloc(128, frSize);
    // auto memFr = new (freq_result *)aligned_alloc(128, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,  frSize, memFr, err);
    cl_uint frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    cl_uint frSizePadded = (frSize + 128 - 1) & ~(128 - 1);
    // auto pfr = new freq_result[CUDA_grid_dim_precalc];
    auto pfr = (freq_result*)aligned_alloc(128, frSize);
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
    // void *pfr;
#elif NVIDIA
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void* memIn = (void*)aligned_alloc(8, frSize);
#endif // NVIDIA
#else // WIN
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim_precalc - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    size_t frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    size_t frSizePadded = (frSize + 128 - 1) & ~(128 - 1);
    auto pfr = (freq_result*)_aligned_malloc(frSize, 128);
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
#elif NVIDIA
    int frSize = CUDA_grid_dim_precalc * sizeof(freq_result);
    void* memIn = (void*)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void* pfr;
#endif // NViDIA
#endif // WIN

#pragma region SetKernelArgs
    err = clSetKernelArg(kernelClCheckEnd, 0, sizeof(cl_mem), &CUDA_End);

    err = clSetKernelArg(kernelCalculatePrepare, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePrepare, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePrepare, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePrepare, 3, sizeof(freq_start), &freq_start);
    err = clSetKernelArg(kernelCalculatePrepare, 4, sizeof(freq_step), &freq_step);
    err = clSetKernelArg(kernelCalculatePrepare, 5, sizeof(max_test_periods), &max_test_periods);

    err = clSetKernelArg(kernelCalculatePreparePole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePreparePole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculatePreparePole, 2, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePreparePole, 3, sizeof(cl_mem), &cgFirst);
    err = clSetKernelArg(kernelCalculatePreparePole, 4, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePreparePole, 5, sizeof(cl_mem), &CUDA_CC2); // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ???
    err = clSetKernelArg(kernelCalculatePreparePole, 6, sizeof(cl_mem), &bufBrightness);

    err = clSetKernelArg(kernelCalculateIter1Begin, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Begin, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculateIter1Begin, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculateIter1Begin, 3, sizeof(int), &n_iter_min);
    err = clSetKernelArg(kernelCalculateIter1Begin, 4, sizeof(int), &n_iter_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 5, sizeof(double), &iter_diff_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 6, sizeof(double), &((*Fa).Alamda_start));

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 2, sizeof(cl_mem), &bufTim);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 3, sizeof(cl_mem), &bufEe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 4, sizeof(cl_mem), &bufEe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 5, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 6, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 7, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 8, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 9, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 10, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 11, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 12, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 13, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 14, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 15, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 16, sizeof(cl_mem), &bufDe0);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 2, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 3, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 4, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 5, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 6, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 7, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 8, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 9, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 10, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 11, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 12, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 13, sizeof(cl_mem), &bufDe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 14, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 15, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 2, sizeof(cl_mem), &bufBrightness);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 3, sizeof(cl_mem), &bufWeight);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 4, sizeof(cl_mem), &bufSig);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 5, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 6, sizeof(cl_mem), &bufYtemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 7, sizeof(cl_mem), &bufIa);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 2, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 3, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 2, sizeof(cl_mem), &bufIa);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 2, sizeof(cl_mem), &bufTim);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 3, sizeof(cl_mem), &bufEe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 4, sizeof(cl_mem), &bufEe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 5, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 6, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 7, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 8, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 9, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 10, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 11, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 12, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 13, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 14, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 15, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 16, sizeof(cl_mem), &bufDe0);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 2, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 3, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 4, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 5, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 6, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 7, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 8, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 9, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 10, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 11, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 12, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 13, sizeof(cl_mem), &bufDe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 14, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 15, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 2, sizeof(cl_mem), &bufBrightness);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 3, sizeof(cl_mem), &bufWeight);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 4, sizeof(cl_mem), &bufSig);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 5, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 6, sizeof(cl_mem), &bufYtemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 7, sizeof(cl_mem), &bufIa);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 2, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 3, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateFinishPole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateFinishPole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateFinishPole, 2, sizeof(cl_mem), &CUDA_FR);
#pragma endregion

    /* Sets local_work_size to BLOCK_DIM = 128 */
    size_t local = BLOCK_DIM;
    size_t sLocal = 1;

    for (n = 1; n <= max_test_periods; n += (int)CUDA_grid_dim_precalc)
    {

#if defined INTEL
        pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        queue.flush();
#elif defined AMD
        // pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        // pfr = clEnqueueMapBuffer(queue, CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, 0, NULL, NULL, &err);
        // queue.flush();
#endif
        for (m = 0; m < CUDA_grid_dim_precalc; m++)
        {
            ((freq_result*)pfr)[m].isInvalid = 1;
            ((freq_result*)pfr)[m].isReported = 0;
            ((freq_result*)pfr)[m].be_best = 0.0;
            ((freq_result*)pfr)[m].dark_best = 0.0;
            ((freq_result*)pfr)[m].dev_best = 0.0;
            ((freq_result*)pfr)[m].freq = 0.0;
            ((freq_result*)pfr)[m].la_best = 0.0;
            ((freq_result*)pfr)[m].per_best = 0.0;
            ((freq_result*)pfr)[m].dev_best_x2 = 0.0;
        }

#if defined INTEL
        queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frOptimizedSize, pfr);
#elif AMD
        clEnqueueWriteBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#elif NVIDIA
        queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        queue.flush();
#endif
        err = clSetKernelArg(kernelCalculatePrepare, 6, sizeof(n), &n);
        err = EnqueueNDRangeKernel(queue, kernelCalculatePrepare, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
        if (getError(err)) return err;

        for (m = 1; m <= N_POLES; m++)
        {
            theEnd = 0; //zero global End signal
            err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);
            err = clSetKernelArg(kernelCalculatePreparePole, 6, sizeof(m), &m);
            err = EnqueueNDRangeKernel(queue, kernelCalculatePreparePole, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;

            clEnqueueReadBuffer(queue, CUDA_CC2, CL_BLOCKING, 0, faSize, pFb, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
            clEnqueueUnmapMemObject(queue, CUDA_CC2, pFb, 0, NULL, NULL);
            clFlush(queue);
#ifdef _DEBUG
            printf(".");
            //cout << ".";
            //cout.flush();
#endif
            int count = 0;
            while (!theEnd)
            {
                count++;
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Begin, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                for (iC = 1; iC < gl.Lcurves; iC++)
                {
                                            // jpScale | mrqcof_matrix() -> matrixNew() | set
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                                            // jpScale | mrqcof_curve1 -> bright ()     | get
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 16, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 8, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 9, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                }
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 4, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 5, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 8, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 9, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1End, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin1End, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                for (iC = 1; iC < gl.Lcurves; iC++)
                {                            // jpScale | mrqcof_matrix() -> matrixNew() | set
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                                            // jpScale | mrqcof_curve1 -> bright ()     | get
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 16, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 8, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 9, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                }

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 4, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 5, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 8, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 9, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2End, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin2End, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = clEnqueueReadBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);

                theEnd = theEnd == CUDA_grid_dim_precalc;
            }

            err = EnqueueNDRangeKernel(queue, kernelCalculateFinishPole, 1, NULL, &CUDA_grid_dim_precalc, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;
        }

        printf("\n");

#if !defined _WIN32
#if defined (INTEL)
        fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
        queue.finish();
#elif AMD
        // queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, sizeof(frSize), pfr);
        // pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frSize, NULL, NULL, err);
        // pfr = clEnqueueMapBuffer(queue, CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frSize, 0, NULL, NULL, &err);
        //queue.flush(); // ***
        // queue.enqueueReadBuffer(CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc);
        clEnqueueReadBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#elif NVIDIA
        pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        queue.flush();
#endif
#else
#if defined (INTEL)
        fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
        queue.finish();
#elif AMD
        clEnqueueReadBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#elif NVIDIA
        pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        queue.flush();
#endif
#endif

#if defined (INTEL)
        auto res = (freq_result*)fres;
#else
        auto res = new freq_result[CUDA_grid_dim_precalc];
        memcpy(res, pfr, frSize);
#endif

        for (m = 1; m <= CUDA_grid_dim_precalc; m++)
        {
            if (res[m - 1].isReported == 1)
            {
                sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
//#if defined _DEBUG
//                printf("[%3d] res[%3d].dark_best: %10.16f, sum_dark_facet: %10.16f\n", m, m - 1, res[m - 1].dark_best, sum_dark_facet);
//#endif
            }
        }

#if !defined _WIN32
#if defined (INTEL)
        queue.enqueueUnmapMemObject(CUDA_FR, fres);
        queue.flush();
#elif AMD
        // queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        // queue.flush();
        // clEnqueueUnmapMemObject(queue, CUDA_FR, pfr, 0, NULL, NULL);
        // clFlush(queue);
        delete[] res;
#elif NVIDIA
#elif NVIDIA
        queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        queue.flush();
#endif
#else
#if defined (INTEL)
        queue.enqueueUnmapMemObject(CUDA_FR, fres);
        queue.flush();
#elif AMD
        delete[] res;
        _aligned_free(pcc);
        _aligned_free(pFa);
        _aligned_free(pFb);
        _aligned_free(pfr);
#elif NVIDIA
        queue.enqueueUnmapMemObject(CUDA_FR, pfr);
        queue.flush();
#endif
#endif
    } /* period loop */

    clReleaseMemObject(CUDA_MCC2);
    clReleaseMemObject(CUDA_CC);
    clReleaseMemObject(CUDA_CC2);
    clReleaseMemObject(CUDA_End);
    clReleaseMemObject(CUDA_FR);
    clReleaseMemObject(cgFirst);
    clReleaseMemObject(bufJpScale);
    clReleaseMemObject(bufJpDphp1);
    clReleaseMemObject(bufJpDphp2);
    clReleaseMemObject(bufJpDphp3);
    clReleaseMemObject(bufE1);
    clReleaseMemObject(bufE2);
    clReleaseMemObject(bufE3);
    clReleaseMemObject(bufE01);
    clReleaseMemObject(bufE02);
    clReleaseMemObject(bufE03);
    clReleaseMemObject(bufDe);
    clReleaseMemObject(bufDe0);
    clReleaseMemObject(bufDytemp);
    clReleaseMemObject(bufYtemp);

    ave_dark_facet = sum_dark_facet / max_test_periods;

    if (ave_dark_facet < 1.0)
        *new_conw = 1; /* new correct conwexity weight */
    if (ave_dark_facet >= 1.0)
        *conw_r = *conw_r * 2;

//#if defined _DEBUG
//    printf("ave_dark_facet: %10.17f\n", ave_dark_facet);
//    printf("conw_r:         %10.17f\n", *conw_r);
//#endif
    return 0;
}

cl_int ClStart(int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
    int ndata, std::vector<int>& ia, int* ia_par, std::vector<double>& cg_first, MFILE& mf, double escl, std::vector<double>& sig, int Numfac,
    std::vector<double>& brightness, struct globals& gl)
{
    double iter_diff_max;
    int retval, i, n, m, iC;
    int n_iter_max, theEnd, LinesWritten;

    int n_max = (int)((freq_start - freq_end) / freq_step) + 1;

    int isPrecalc = 0;
    auto r = 0;
    char buf[256];

    for (i = 1; i <= n_ph_par; i++)
    {
        ia[n_coef + 3 + i] = ia_par[i];
    }

    n_iter_max = 0;
    iter_diff_max = -1;
    if (stop_condition > 1)
    {
        n_iter_max = (int)stop_condition;
        iter_diff_max = 0;
        n_iter_min = 0; /* to not overwrite the n_iter_max value */
    }
    if (stop_condition < 1)
    {
        n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
        iter_diff_max = stop_condition;
    }

    (*Fa).conw_r = conw_r;
    (*Fa).Ncoef = n_coef; //Ncoef;
    (*Fa).Nphpar = n_ph_par;
    (*Fa).Numfac = Numfac;
    m = Numfac + 1;
    (*Fa).Numfac1 = m;
    (*Fa).ndata = ndata;
    (*Fa).Is_Precalc = isPrecalc;

    bufBrightness = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, brightness.size() * sizeof(double), brightness.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufBrightness, CL_BLOCKING, 0, brightness.size() * sizeof(double), brightness.data(), 0, NULL, NULL);

    bufSig = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sig.size() * sizeof(double), sig.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufSig, CL_BLOCKING, 0, sig.size() * sizeof(double), sig.data(), 0, NULL, NULL);

    bufIa = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ia.size() * sizeof(int), ia.data(), &err);
    err = clEnqueueWriteBuffer(queue, bufIa, CL_BLOCKING, 0, ia.size() * sizeof(int), ia.data(), 0, NULL, NULL);

    //here move data to device
    memcpy((*Fa).Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
    memcpy((*Fa).Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    memcpy((*Fa).Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    memcpy((*Fa).Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
    memcpy((*Fa).Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
    memcpy((*Fa).Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
    //memcpy((*Fa).Brightness, brightness, (ndata + 1) * sizeof(double));		// sizeof(double)* (MAX_N_OBS + 1));
    //memcpy((*Fa).Sig, sig, (ndata + 1) * sizeof(double));							// sizeof(double)* (MAX_N_OBS + 1));
    //memcpy((*Fa).ia, ia, sizeof(int) * (MAX_N_PAR + 1));

    /* number of fitted parameters */
    int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
    for (m = 1; m <= ma; m++)
    {
        if (ia[m])
        {
            lmfit++;
            llastma = m;
        }
    }
    llastone = 1;
    for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
    {
        if (!ia[m]) break;
        llastone = m;
    }

    (*Fa).Ncoef = n_coef;
    (*Fa).ma = ma;
    (*Fa).Mfit = lmfit;

    m = lmfit + 1;
    (*Fa).Mfit1 = m;

    (*Fa).lastone = llastone;
    (*Fa).lastma = llastma;

    m = ma - 2 - n_ph_par;
    (*Fa).Ncoef0 = m;

    auto totalWorkItems = CUDA_grid_dim * BLOCK_DIM; // 768 * 128 = 98304
    m = (Numfac + 1) * (n_coef + 1);
    (*Fa).Dg_block = m;

    cl_int err = 0;

#if defined (INTEL)
    auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
#else
    cl_mem cgFirst = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cg_first.size() * sizeof(double), cg_first.data(), &err);
    clEnqueueWriteBuffer(queue, cgFirst, CL_BLOCKING, 0, cg_first.size() * sizeof(double), cg_first.data(), 0, NULL, NULL);
#endif

#if !defined _WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    // cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    // auto pcc = (mfreq_context *)aligned_alloc(8, optimizedSize);
    // auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);

    cl_uint pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto pcc = new mfreq_context[CUDA_grid_dim];
#elif NVIDIA
    cl_uint pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#else  // WIN32
#if defined INTEL
    cl_uint optimizedSize = ((sizeof(mfreq_context) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pcc = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, optimizedSize, pcc, err);
#elif AMD
    //size_t pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    //auto pcc = new mfreq_context[CUDA_grid_dim];
    auto pccSize = ((sizeof(mfreq_context) * CUDA_grid_dim) / 128 + 1) * 128;
    auto pcc = (mfreq_context *)_aligned_malloc(pccSize, 128);
#elif NVIDIA
    int pccSize = CUDA_grid_dim * sizeof(mfreq_context);
    auto alignas(8) pcc = new mfreq_context[CUDA_grid_dim];
    auto CUDA_MCC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, err);
#endif // NVIDIA
#endif

    cl_mem bufJpScale;
    cl_mem bufJpDphp1;
    cl_mem bufJpDphp2;
    cl_mem bufJpDphp3;
    cl_mem bufE1;
    cl_mem bufE2;
    cl_mem bufE3;
    cl_mem bufE01;
    cl_mem bufE02;
    cl_mem bufE03;
    cl_mem bufDe;
    cl_mem bufDe0;
    cl_mem bufDytemp;
    cl_mem bufYtemp;

    size_t commonSize = CUDA_grid_dim * (gl.maxLcPoints + 1);
    PrepareBufferFromFlatenArray<double>(bufJpScale, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufJpDphp1, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufJpDphp2, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufJpDphp3, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE1, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE2, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE3, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE01, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE02, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufE03, commonSize, 128);
    PrepareBufferFromFlatenArray<double>(bufYtemp, commonSize, 128);

    size_t deSize = commonSize * 4 * 4; // double de[POINTS_MAX + 1][4][4];
    PrepareBufferFromFlatenArray<double>(bufDe, deSize, 128);
    PrepareBufferFromFlatenArray<double>(bufDe0, deSize, 128);

    size_t dytempSize = commonSize * (MAX_N_PAR + 1); // double dytemp[(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
    PrepareBufferFromFlatenArray<double>(bufDytemp, dytempSize, 128);

    for (m = 0; m < CUDA_grid_dim; m++)
    {
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Area), MAX_N_FAC + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].Dg), (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].alpha), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].covar), (MAX_N_PAR + 1) * (MAX_N_PAR + 1), 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].beta), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].da), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].atry), MAX_N_PAR + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].dave), MAX_N_PAR + 1, 0.0);
        //std::fill_n(std::begin(((mfreq_context*)pcc)[m].dytemp), (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
        //std::fill_n(std::begin(((mfreq_context*)pcc)[m].ytemp), POINTS_MAX + 1, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_big), BLOCK_DIM, 0.0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_icol), BLOCK_DIM, 0);
        std::fill_n(std::begin(((mfreq_context*)pcc)[m].sh_irow), BLOCK_DIM, 0);
        //pcc[m].conw_r = 0.0;
        ((mfreq_context*)pcc)[m].icol = 0;
        ((mfreq_context*)pcc)[m].pivinv = 0;
    }

#if !defined _WIN32
#if defined (INTEL)
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#else
    // queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
    cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, &err);
    clEnqueueWriteBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
#endif
#else // WIN32
#if defined (INTEL)
    queue.enqueueWriteBuffer(CUDA_MCC2, CL_BLOCKING, 0, optimizedSize, pcc);
#else
    cl_mem CUDA_MCC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pccSize, pcc, &err);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error creating OpenCL buffer: %d\n", err);
        // Handle error
    }
    err = clEnqueueWriteBuffer(queue, CUDA_MCC2, CL_BLOCKING, 0, pccSize, pcc, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error creating OpenCL buffer: %d\n", err);
        // Handle error
    }
#endif
#endif

#if !defined _WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    // cl_uint faSize = sizeof(freq_context);
    // auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, faSize, Fa, err);
    // queue.enqueueWriteBuffer(CUDA_CC, CL_BLOCKING, 0, faSize, Fa);
    auto memFa = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, &err);
    void* pFa = clEnqueueMapBuffer(queue, CUDA_CC, CL_BLOCKING, CL_MAP_WRITE, 0, faSize, 0, NULL, NULL, &err);
    memcpy(pFa, Fa, faSize);
    clEnqueueUnmapMemObject(queue, CUDA_CC, pFa, 0, NULL, NULL);
    clFlush(queue);
#endif
#else // WIN32
#if defined (INTEL)
    auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faOptimizedSize, Fa, err);
#else
    auto memFa = (freq_context*)_aligned_malloc(faSize, 128);
    memFa->maxLcPoints = gl.maxLcPoints;
    cl_mem CUDA_CC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFa, &err);
    void* pFa = clEnqueueMapBuffer(queue, CUDA_CC, CL_BLOCKING, CL_MAP_WRITE, 0, faSize, 0, NULL, NULL, &err);
    memcpy(pFa, Fa, faSize);
    clEnqueueUnmapMemObject(queue, CUDA_CC, pFa, 0, NULL, NULL);
    clFlush(queue);
#endif
#endif

#if defined (INTEL)
    auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int), &theEnd, err);
#else
    cl_mem CUDA_End = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(theEnd), &theEnd, &err);
    err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);
#endif

#if !defined _WIN32
    // freq_context* Fb;
    // auto CUDA_CC2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(freq_context), Fb, err);
    auto memFb = (freq_context*)aligned_alloc(128, faSize);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFb, &err);
#else
    auto memFb = (freq_context*)_aligned_malloc(faSize, 128);
    cl_mem CUDA_CC2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, faSize, memFb, &err);
#endif

#if !defined _WIN32
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)aligned_alloc(4096, optimizedSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    // cl_uint frSize = CUDA_grid_dim * sizeof(freq_result);
    // void *memIn = (void *)aligned_alloc(128, frSize);
    // auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    // void *pfr;
    cl_uint frSize = sizeof(freq_result) * CUDA_grid_dim;
    auto pfr = new freq_result[CUDA_grid_dim];
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
#elif NVIDIA
    cl_uint = CUDA_grid_dim * sizeof(freq_result);
    void* memIn = (void*)aligned_alloc(8, frSize);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void* pfr;
#endif // NVIDIA
#else  // WIN
#if defined INTEL
    cl_uint frOptimizedSize = ((sizeof(freq_result) * CUDA_grid_dim - 1) / 64 + 1) * 64;
    auto pfr = (mfreq_context*)_aligned_malloc(optimizedSize, 4096);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frOptimizedSize, pfr, err);
#elif defined AMD
    //size_t frSize = sizeof(freq_result) * CUDA_grid_dim;
    //auto pfr = new freq_result[CUDA_grid_dim];
    size_t frSize = CUDA_grid_dim * sizeof(freq_result);
    auto pfr = (freq_result *)_aligned_malloc(frSize, 128);
    cl_mem CUDA_FR = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, pfr, &err);
#elif NVIDIA
    int frSize = CUDA_grid_dim * sizeof(freq_result);
    void* memIn = (void*)_aligned_malloc(frSize, 256);
    auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, frSize, memIn, err);
    void* pfr;
#endif // NViDIA
#endif // WIN

#pragma region SetKernelArguments
    err = clSetKernelArg(kernelClCheckEnd, 0, sizeof(cl_mem), &CUDA_End);

    err = clSetKernelArg(kernelCalculatePrepare, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePrepare, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePrepare, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePrepare, 3, sizeof(freq_start), &freq_start);
    err = clSetKernelArg(kernelCalculatePrepare, 4, sizeof(freq_step), &freq_step);
    err = clSetKernelArg(kernelCalculatePrepare, 5, sizeof(n_max), &n_max);

    err = clSetKernelArg(kernelCalculatePreparePole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculatePreparePole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculatePreparePole, 2, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculatePreparePole, 3, sizeof(cl_mem), &cgFirst);
    err = clSetKernelArg(kernelCalculatePreparePole, 4, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculatePreparePole, 5, sizeof(cl_mem), &CUDA_CC2); // <<<<<<<<<<<<<<<<<<<< ??

    err = clSetKernelArg(kernelCalculateIter1Begin, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Begin, 1, sizeof(cl_mem), &CUDA_FR);
    err = clSetKernelArg(kernelCalculateIter1Begin, 2, sizeof(cl_mem), &CUDA_End);
    err = clSetKernelArg(kernelCalculateIter1Begin, 3, sizeof(int), &n_iter_min);
    err = clSetKernelArg(kernelCalculateIter1Begin, 4, sizeof(int), &n_iter_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 5, sizeof(double), &iter_diff_max);
    err = clSetKernelArg(kernelCalculateIter1Begin, 6, sizeof(double), &((*Fa).Alamda_start));

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 2, sizeof(cl_mem), &bufTim);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 3, sizeof(cl_mem), &bufEe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 4, sizeof(cl_mem), &bufEe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 5, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 6, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 7, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 8, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 9, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 10, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 11, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 12, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 13, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 14, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 15, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 16, sizeof(cl_mem), &bufDe0);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 2, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 3, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 4, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 5, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 6, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 7, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 8, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 9, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 10, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 11, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 12, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 13, sizeof(cl_mem), &bufDe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 14, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 15, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 2, sizeof(cl_mem), &bufBrightness);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 3, sizeof(cl_mem), &bufWeight);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 4, sizeof(cl_mem), &bufSig);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 5, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 6, sizeof(cl_mem), &bufYtemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 7, sizeof(cl_mem), &bufIa);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 2, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 3, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof1End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin1End, 2, sizeof(cl_mem), &bufIa);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Start, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 2, sizeof(cl_mem), &bufTim);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 3, sizeof(cl_mem), &bufEe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 4, sizeof(cl_mem), &bufEe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 5, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 6, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 7, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 8, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 9, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 10, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 11, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 12, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 13, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 14, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 15, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 16, sizeof(cl_mem), &bufDe0);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 2, sizeof(cl_mem), &bufJpScale);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 3, sizeof(cl_mem), &bufJpDphp1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 4, sizeof(cl_mem), &bufJpDphp2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 5, sizeof(cl_mem), &bufJpDphp3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 6, sizeof(cl_mem), &bufE1);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 7, sizeof(cl_mem), &bufE2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 8, sizeof(cl_mem), &bufE3);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 9, sizeof(cl_mem), &bufE01);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 10, sizeof(cl_mem), &bufE02);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 11, sizeof(cl_mem), &bufE03);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 12, sizeof(cl_mem), &bufDe);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 13, sizeof(cl_mem), &bufDe0);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 14, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 15, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 2, sizeof(cl_mem), &bufBrightness);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 3, sizeof(cl_mem), &bufWeight);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 4, sizeof(cl_mem), &bufSig);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 5, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 6, sizeof(cl_mem), &bufYtemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 7, sizeof(cl_mem), &bufIa);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 2, sizeof(cl_mem), &bufDytemp);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 3, sizeof(cl_mem), &bufYtemp);

    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqcof2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter1Mrqmin2End, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateIter2, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateIter2, 1, sizeof(cl_mem), &CUDA_CC);

    err = clSetKernelArg(kernelCalculateFinishPole, 0, sizeof(cl_mem), &CUDA_MCC2);
    err = clSetKernelArg(kernelCalculateFinishPole, 1, sizeof(cl_mem), &CUDA_CC);
    err = clSetKernelArg(kernelCalculateFinishPole, 2, sizeof(cl_mem), &CUDA_FR);
#pragma endregion

    auto oldFractionDone = 0.0001;
    int count = 0;
    size_t local = BLOCK_DIM;
    size_t sLocal = 1;

    for (n = n_start_from; n <= n_max; n += (int)CUDA_grid_dim)
    {
        auto fractionDone = (double)n / (double)n_max;

#ifndef INTEL
        // pfr = queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, frSize, NULL, NULL, err);
        // queue.flush();
#endif
        for (int j = 0; j < CUDA_grid_dim; j++)
        {
            ((freq_result*)pfr)[j].isInvalid = 1;
            ((freq_result*)pfr)[j].isReported = 0;
            ((freq_result*)pfr)[j].be_best = 0.0;
            ((freq_result*)pfr)[j].dark_best = 0.0;
            ((freq_result*)pfr)[j].dev_best = 0.0;
            ((freq_result*)pfr)[j].freq = 0.0;
            ((freq_result*)pfr)[j].la_best = 0.0;
            ((freq_result*)pfr)[j].per_best = 0.0;
        }

#if defined (INTEL)
        queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frOptimizedSize, pfr);
#else
        clEnqueueWriteBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#endif
        err = clSetKernelArg(kernelCalculatePrepare, 6, sizeof(n), &n);
        err = EnqueueNDRangeKernel(queue, kernelCalculatePrepare, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
        if (getError(err)) return err;

        for (m = 1; m <= N_POLES; m++)
        {
            auto mid = (double(fractionDone) - double(oldFractionDone));
            auto inner = (double(mid) / double(N_POLES) * (double(m)));
            auto fractionDone2 = oldFractionDone + inner;
            boinc_fraction_done(fractionDone2);

#ifdef _DEBUG
            float fraction2 = fractionDone2 * 100;
            std::time_t t = std::time(nullptr);   // get time now
            std::tm* now = std::localtime(&t);

            printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
            fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
#endif

            theEnd = 0;  //zero global End signal
            err = clEnqueueWriteBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);
            err = clSetKernelArg(kernelCalculatePreparePole, 6, sizeof(m), &m);
            err = EnqueueNDRangeKernel(queue, kernelCalculatePreparePole, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;

            count = 0;

            while (!theEnd)
            {
                count++;
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Begin, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                for (iC = 1; iC < gl.Lcurves; iC++)
                {
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Matrix, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 16, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 8, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 9, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                }

                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 4, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve1Last, 5, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 8, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof1Curve2, 9, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof1End, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin1End, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Start, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                for (iC = 1; iC < gl.Lcurves; iC++)
                {
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Matrix, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Matrix, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 16, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1, 17, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;

                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 8, sizeof(gl.Inrel[iC]), &(gl.Inrel[iC]));
                    err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 9, sizeof(gl.Lpoints[iC]), &(gl.Lpoints[iC]));
                    err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                    if (getError(err)) return err;
                }

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 4, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve1Last, 5, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve1Last, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 8, sizeof(gl.Inrel[gl.Lcurves]), &(gl.Inrel[gl.Lcurves]));
                err = clSetKernelArg(kernelCalculateIter1Mrqcof2Curve2, 9, sizeof(gl.Lpoints[gl.Lcurves]), &(gl.Lpoints[gl.Lcurves]));
                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2Curve2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqcof2End, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter1Mrqmin2End, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
                if (getError(err)) return err;

                err = EnqueueNDRangeKernel(queue, kernelCalculateIter2, 1, NULL, &totalWorkItems, &local, 0, NULL, NULL);
                if (getError(err)) return err;

                err = clEnqueueReadBuffer(queue, CUDA_End, CL_BLOCKING, 0, sizeof(theEnd), &theEnd, 0, NULL, NULL);

                theEnd = theEnd == CUDA_grid_dim;
            }

            //printf("."); fflush(stdout);
            err = EnqueueNDRangeKernel(queue, kernelCalculateFinishPole, 1, NULL, &CUDA_grid_dim, &sLocal, 0, NULL, NULL);
            if (getError(err)) return err;
        }

#if defined (INTEL)
        fres = (freq_result*)queue.enqueueMapBuffer(CUDA_FR, CL_BLOCKING, CL_MAP_READ, 0, frOptimizedSize, NULL, NULL, err);
        queue.finish();
#else
        clEnqueueReadBuffer(queue, CUDA_FR, CL_BLOCKING, 0, frSize, pfr, 0, NULL, NULL);
#endif

        oldFractionDone = fractionDone;
        LinesWritten = 0;
#if defined (INTEL)
        auto res = (freq_result*)fres;
#else
        auto res = new freq_result[CUDA_grid_dim];
        memcpy(res, pfr, frSize);
#endif
        for (m = 1; m <= CUDA_grid_dim; m++)
		{
			if (res[m - 1].isReported == 1)
			{
				LinesWritten++;
				double dark_best = n == 1 && m == 1
					? conw_r * escl * escl
					: res[m - 1].dark_best;

				/* output file */
				mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), dark_best, round(res[m - 1].la_best), round(res[m - 1].be_best));
			}
		}

        delete[] res;

        if (boinc_time_to_checkpoint() || boinc_is_standalone())
        {
            retval = DoCheckpoint(mf, (n - 1) + LinesWritten, 1, conw_r); //zero lines
            if (retval) { fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval); }
            boinc_checkpoint_completed();
        }
        //		break;//debug

        printf("\n");
        fflush(stdout);
    } /* period loop */

    printf("\n");

    clReleaseMemObject(CUDA_MCC2);
    clReleaseMemObject(CUDA_CC);
    clReleaseMemObject(CUDA_CC2);
    clReleaseMemObject(CUDA_End);
    clReleaseMemObject(CUDA_FR);
    clReleaseMemObject(cgFirst);
    clReleaseMemObject(bufJpScale);
    clReleaseMemObject(bufJpDphp1);
    clReleaseMemObject(bufJpDphp2);
    clReleaseMemObject(bufJpDphp3);
    clReleaseMemObject(bufE1);
    clReleaseMemObject(bufE2);
    clReleaseMemObject(bufE3);
    clReleaseMemObject(bufE01);
    clReleaseMemObject(bufE02);
    clReleaseMemObject(bufE03);
    clReleaseMemObject(bufDe);
    clReleaseMemObject(bufDe0);
    clReleaseMemObject(bufDytemp);
    clReleaseMemObject(bufYtemp);

#if !defined _WIN32
#if defined INTEL
    free(pcc);
#elif defined AMD
    // free(memIn);
    // free(pcc);
    delete[] pcc;
    delete[] pfr;
    //free(pFa);
#elif defined NVIDIA
    free(memIn);
    free(pcc);
    delete[] pcc;
#endif
#else // WIN
    //_aligned_free(pfr); // res does not need to be freed as it's just a pointer to *pfr.
#if defined(INTEL)
    _aligned_free(pcc);
#elif defined AMD
    _aligned_free(memFa);
    _aligned_free(memFb);
    _aligned_free(Fa);
    _aligned_free(pfr);
    _aligned_free(pcc);

#elif defined NVIDIA
    delete[] pcc;
#endif
#endif // WIN

    return 0;
}

void ReleaseGlobalClObjects()
{
    clReleaseKernel(kernelClCheckEnd);
    clReleaseKernel(kernelCalculatePrepare);
    clReleaseKernel(kernelCalculatePreparePole);
    clReleaseKernel(kernelCalculateIter1Begin);
    clReleaseKernel(kernelCalculateIter1Mrqcof1Start);
    clReleaseKernel(kernelCalculateIter1Mrqcof1Matrix);
    clReleaseKernel(kernelCalculateIter1Mrqcof1Curve1);
    clReleaseKernel(kernelCalculateIter1Mrqcof1Curve2);
    clReleaseKernel(kernelCalculateIter1Mrqcof1Curve1Last);
    clReleaseKernel(kernelCalculateIter1Mrqcof1End);
    clReleaseKernel(kernelCalculateIter1Mrqmin1End);
    clReleaseKernel(kernelCalculateIter1Mrqcof2Start);
    clReleaseKernel(kernelCalculateIter1Mrqcof2Matrix);
    clReleaseKernel(kernelCalculateIter1Mrqcof2Curve1);
    clReleaseKernel(kernelCalculateIter1Mrqcof2Curve2);
    clReleaseKernel(kernelCalculateIter1Mrqcof2Curve1Last);
    clReleaseKernel(kernelCalculateIter1Mrqcof2End);
    clReleaseKernel(kernelCalculateIter1Mrqmin2End);
    clReleaseKernel(kernelCalculateIter2);
    clReleaseKernel(kernelCalculateFinishPole);

    clReleaseMemObject(bufSig);
    clReleaseMemObject(bufTim);
    clReleaseMemObject(bufBrightness);
    clReleaseMemObject(bufEe);
    clReleaseMemObject(bufEe0);
    clReleaseMemObject(bufWeight);
    clReleaseMemObject(bufIa);

    clReleaseProgram(binProgram);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

//#endif
