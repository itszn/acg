rem build\bin\Debug\optixHello.exe -f output\out.ppm
rem build\bin\Debug\optixSphere.exe -f output\out.ppm
del output\out.ppm
del output\out.png
build\bin\Debug\optixWhitted.exe src\optixWhitted\bunny_200.obj output\out.ppm
