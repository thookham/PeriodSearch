#include "globals.h"
#include "opencl_helper.h"

#include <cstdlib>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <CL/cl.h>

std::string transformKernelCode(const std::string &kernel_code)
{
    // Regular expressions to match tabs and multiple spaces
    std::regex tabRegex("\\t");
    std::regex spacesRegex(" {2,}");
    //std::regex quoteRegex("\"");

    // Replace tabs with a single space
    std::string transformed_code = std::regex_replace(kernel_code, tabRegex, " ");
    // Replace multiple spaces with a single space
    transformed_code = std::regex_replace(transformed_code, spacesRegex, " ");
    // Escape double quotes
    //transformed_code = std::regex_replace(transformed_code, quoteRegex, "\\\"");

    return transformed_code;
}

//std::string transformKernelCode(const std::string &kernel_code)
//{
//    // Regular expressions to match tabs and multiple spaces
//    std::regex tabRegex("\\t");
//    std::regex spacesRegex(" {2,}");
//    std::regex quoteRegex("\"");
//
//    // Replace tabs with a single space
//    std::string transformed_code = std::regex_replace(kernel_code, tabRegex, " ");
//    // Replace multiple spaces with a single space
//    transformed_code = std::regex_replace(transformed_code, spacesRegex, " ");
//    // Escape double quotes
//    transformed_code = std::regex_replace(transformed_code, quoteRegex, "\\\"");
//
//    // Resulting string to hold the transformed code
//    std::string result;
//
//    // Split the transformed_code into lines and append each line with formatting
//    std::istringstream stream(transformed_code);
//    std::string line;
//    while (std::getline(stream, line))
//    {
//        result += "\"" + line + "\\n\"\n";
//    }
//
//    // End the variable declaration properly
//    //result += ";";
//
//    return result;
//}

int GetProgramBuildInfo(const cl_program program, const cl_device_id device, const char *name, const char deviceName[], const cl_int err_num)
{
    size_t len;

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

    auto buffer = new char[len];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

    std::cerr << "Build log: " << name << " | " << deviceName << ":" << std::endl << buffer << std::endl;
    std::cerr << " Error creating queue: " << cl_error_to_str(err_num) << "(" << err_num << ")\n";
    delete[] buffer;

    return EXIT_FAILURE;
}
