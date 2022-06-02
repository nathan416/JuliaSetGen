# JuliaSetGen
This is a kivy app that can generate any julia set image based off a given expression.
It uses sympy and numba Cuda to interpret the expression into images and then displays them in the app.

it needs https://developer.nvidia.com/cuda-downloads if to you want to make use of a cuda(Nvidia) gpu.  
the gpu will take ~2 seconds to create an image  
the cpu will take ~60 seconds to create an image  
these times are very dependent on the amount of cores, speed, etc of cpu and gpu  

compatible color maps can be found at https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html  

examples to try:  
1 - x\*\*2 + x\*\*2 / (2 + 4 \* x) + 0.7885  
1 - x + x\*\*2 + 0.6\*1j  
x\*\*3/(x-1) + x\*\*2/(x\*\*3 + 4 \*x\*\*2 + 5) + 0.1185 - 0.551\*1j  
x\*\*4 + x\*\*3/(x-1) + x\*\*2/(x\*\*3 + 4 \*x\*\*2 + 5) + 0.255534\*cos(1) + 0.822292\*1j\*cos(0) - 2\*0.737292\*1j  
2\*\*x + 0.085877 + 0.211764\*1j  
x\*\*2 + 0.45534-0.337292\*1j  
x\*\*4 + x\*\*3 / (x - 1) + x\*\*2 / (x\*\*3 + 4 \* x\*\*2 + 5) - 0.375646 \* 1j + 0.377767  
x\*\*2 + 2.3\*.01 - 2.3\*.3\*1j  
2\*\*x + 0.9 - 0.8\*1j  
x\*\*x + 0.03  
2\*\*x -x\*\*2+0.71 - 0.49\*1j  

![img](https://i.imgur.com/QLtjBcw.png)
