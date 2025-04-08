@echo off
REM 
set MATLAB_PATH=F:\MATLAB\bin\win64\MATLAB.exe
set LIBSVM_PATH=F:/MATLAB/libsvm-3.35/libsvm-3.35/matlab
set DATA_FILE=%1

REM 
set DATA_FILE=%DATA_FILE:"=%

REM 
echo addpath('%LIBSVM_PATH%'); > temp_script.m
echo disp('LIBSVM PATH'); >> temp_script.m
echo train_svm_model('%DATA_FILE%'); >> temp_script.m

REM 
"%MATLAB_PATH%" -batch "run('temp_script.m')"
