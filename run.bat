rem build\bin\Debug\optixHello.exe -f output\out.ppm
rem build\bin\Debug\optixSphere.exe -f output\out.ppm
del output\out.ppm
del output\out.png
build\bin\Debug\optixWhitted.exe unused output\out.ppm
