@echo off
python oclProgramFileToString.py kernelSource.cl ./period_search/kernels.cpp

rem Specify the files
set sourceFile=kernels_hash.cpp
set targetFile=period_search\kernels.cpp

rem Append the content of sourceFile to targetFile
type "%sourceFile%" >> "%targetFile%"
