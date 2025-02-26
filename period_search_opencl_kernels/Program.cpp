#include <fstream>
#include <iostream>
#include <sstream>
#include <windows.h>

#include "declarations.h"

int main(int argc, char** argv)
{
    char cwd[260];
    if (GetCurrentDirectoryA(sizeof(cwd), cwd)) {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    else {
        std::cerr << "GetCurrentDirectory() error" << std::endl;
    }

    std::time_t now = std::time(nullptr);
    char datetime[100];
    struct tm timeinfo;
    localtime_s(&timeinfo, &now);
    std::strftime(datetime, sizeof(datetime), "%Y-%m-%d %H:%M:%S", &timeinfo);

	std::string kernelSourceFile = "kernelSource.cl";

    // Load CL file, build CL program object, create CL kernel object
#if !defined _WIN32
    std::ifstream constantsFile("OpenCl/constants.h", std::ios::in | std::ios::binary);
    std::ifstream globalsFile("OpenCl/GlobalsCL.h", std::ios::in | std::ios::binary);
    std::ifstream intrinsicsFile("OpenCl/Intrinsics.cl", std::ios::in | std::ios::binary);
    std::ifstream swapFile("OpenCl/swap.cl", std::ios::in | std::ios::binary);
    std::ifstream blmatrixFile("OpenCl/blmatrix.cl", std::ios::in | std::ios::binary);
    std::ifstream curvFile("OpenCl/curv.cl", std::ios::in | std::ios::binary);
    std::ifstream curv2File("OpenCl/Curv2.cl", std::ios::in | std::ios::binary);
    std::ifstream mrqcofFile("OpenCl/mrqcof.cl", std::ios::in | std::ios::binary);
    std::ifstream startFile("OpenCl/Start.cl", std::ios::in | std::ios::binary);
    std::ifstream brightFile("OpenCl/bright.cl", std::ios::in | std::ios::binary);
    std::ifstream convFile("OpenCl/conv.cl", std::ios::in | std::ios::binary);
    std::ifstream mrqminFile("OpenCl/mrqmin.cl", std::ios::in | std::ios::binary);
    std::ifstream gauserrcFile("OpenCl/gauss_errc.cl", std::ios::in | std::ios::binary);
    //std::ifstream testFile("OpenCl/test.cl", std::ios::in | std::ios::binary);
#else
    std::ifstream constantsFile("OpenCl/constants.h");
    if (!constantsFile.good())
    {
        std::cerr << "Error: Failed to open constants.h" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream globalsFile("OpenCl/GlobalsCL.h");
    std::ifstream intrinsicsFile("OpenCl/Intrinsics.cl");
    std::ifstream swapFile("OpenCl/swap.cl");
    std::ifstream blmatrixFile("OpenCl/blmatrix.cl");
    std::ifstream curvFile("OpenCl/curv.cl");
    std::ifstream curv2File("OpenCl/Curv2.cl");
    std::ifstream mrqcofFile("OpenCl/mrqcof.cl");
    std::ifstream startFile("OpenCl/Start.cl");
    std::ifstream brightFile("OpenCl/bright.cl");
    std::ifstream convFile("OpenCl/conv.cl");
    std::ifstream mrqminFile("OpenCl/mrqmin.cl");
    std::ifstream gauserrcFile("OpenCl/gauss_errc.cl");
#endif

    // NOTE: The following order is crucial
    std::stringstream st;

    // 1. First load all helper and function Cl files which will be used by the kernels;
    st << constantsFile.rdbuf();

    st << globalsFile.rdbuf();
    st << intrinsicsFile.rdbuf();
    st << swapFile.rdbuf();
    st << blmatrixFile.rdbuf();
    st << curvFile.rdbuf();
    st << curv2File.rdbuf();
    st << brightFile.rdbuf();
    st << convFile.rdbuf();
    st << mrqcofFile.rdbuf();
    st << gauserrcFile.rdbuf();
    st << mrqminFile.rdbuf();

    //2. Load the files that contains all kernels;
    st << startFile.rdbuf();

    auto kernel_code = st.str(); //.c_str();
    st.flush();

    constantsFile.close();
    globalsFile.close();
    intrinsicsFile.close();
    startFile.close();
    blmatrixFile.close();
    curvFile.close();
    mrqcofFile.close();
    brightFile.close();
    curv2File.close();
    convFile.close();
    mrqminFile.close();
    gauserrcFile.close();
    swapFile.close();
    //testFile.close();

    // cerr << kernel_code << endl;
    std::ofstream out(kernelSourceFile, std::ios::out | std::ios::binary);
    out << kernel_code << std::endl;

    out << "// Created on: " << datetime << std::endl;

    out.close();

	std::cout << "Kernel source file created: " << kernelSourceFile << std::endl;
}
