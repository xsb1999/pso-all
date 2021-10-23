# 3 versions of improved PSO (Particle swarm optimization), including the original one.

**PSO****优化**

 

**PSO****速度和位置更新公式：**

 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image002.jpg)

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image004.jpg)

 

 

**1****、优化目的**

 

·PSO0 à PSO1

 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image006.png)

 

​    PSO0中w是常数，在PSO1中，w满足上式

​    w越大，粒子移动越快，每次移动的跨度也越大，因此适当增大w会缩短优化时间。

​    但w大同样会带来问题，即粒子可能会直接跳过极值点导致无法找出最优解。

​    因此对粒子按照当前适应度值是否大于平均适应度值进行分类。若该粒子的适应度值大于均值，说明该粒子距离最优位置还有很远的距离，因此要使w大一些，取![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image008.png)。若小于等于均值，则利用![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image010.png) 来求解w。首先，因为此时该粒子的适应度值小于均值，说明它已经比较接近最优位置了，因此w要偏小一些，以防w过大导致粒子跳过最佳位置。但这时又会带来一个问题：陷入局部最优解。为了解决这个问题，我采用了![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image012.png) 来进行w的约束，当该值较大时，说明均值很接近最小值，即粒子靠的很近，聚在一个点，这很可能是局部最优解，因此要增大w来跳出这个局部最优位置。但又不能使w过大，因此在这里，利用![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image014.png)加上一个部分：![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image016.png)，类似于惩罚因子。

 

 

·PSO1 à PSO2

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image018.png)

 

​    在PSO2中，我针对PSO1做出了以下几个改动：

**1）** **将**![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image020.png)**改成了**![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image022.png)

​    因为在实际测试中发现若取![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image014.png)则会导致迭代速度很慢并且常常因为w值不够大而无法跳出局部最优位置。因此在这里将其改成了![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image024.png)，这样一来w就足够大，既可以加快收敛速度，又能使粒子更好地跳出局部最优位置。

**2）** **改变了乘积项**

将原先的![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image016.png) 改为了

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image026.png) 

 

​       因为这里采用的是![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image028.png)，因此后面的多项式越小w就越大。而当当前粒子的适应度值接近所有粒子的平均适应度值、平均适应度值又十分接近最小值时，则说明所有粒子都聚集在一个位置了，而这个位置有可能是局部最优位置，因此要使w大一些，即减得少一些。

​    对于500这这个常数，是我经过测试得到的经验值，并无实际含义。

 

 

 

**2****、具体方法**

 

**PSO1****惯性权重****w****更新公式：**

 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image006.png)

 

 

 

**PSO2****惯性权重****w****更新公式：**

 

Step1：

 

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image030.png)

 

 

 

Step2：

![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image018.png)

 

 

 

 

 

 

**流程图：**

![说明: C:\Users\Administrator\Downloads\未命名文件.png](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image032.png)

 

 

 

**3****、优化效果**

 

![说明: D:\TBM_A\塌方\PSO优化\LGB.PNG](file:///C:\Users\ADMINI~1\AppData\Local\Temp\msohtmlclip1\01\clip_image033.png)
