# 1.算法简介  
本部分椭圆检测算法来自论文[Arc-support Line Segments Revisited: An Efﬁcient High-quality Ellipse Detection ](https://arxiv.org/pdf/1810.03243.pdf)  
# 2.环境搭建
- MATLAB
- OpenCV (Version 2.4.9)
- 64-bit Windows Operating System
#3.使用方法
-首先在Matlab中输入"mex generateEllipseCandidates.cpp -IF:\OpenCV\opencv2.4.9\build\include -IF:\OpenCV\opencv2.4.9\build\include\opencv -IF:\OpenCV\opencv2.4.9\build\include\opencv2 -LF:\OpenCV\opencv2.4.9\build\x64\vc11\lib -IF:\Matlab\settlein\extern\include -LF:\Matlab\settlein\extern\lib\win64\microsoft -lopencv_core249 -lopencv_highgui249 -lopencv_imgproc249 -llibmwlapack.lib  "  
其中”F:\OpenCV\opencv2.4.9“、”F:\Matlab“应替换成实际的opencv和matlab安装地址
- 将追踪生成的"detection.txt"放置在matlab工作路径下
- Matlab中输入"demo.m"进行椭圆检测



