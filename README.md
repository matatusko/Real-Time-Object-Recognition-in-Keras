## Real time object recognition in Keras

Very simple model built with Keras (tensorflow backend) and computer vision using pretrained weights from ResNet50. 
Mostly built for fun to play around with computer vision.

The project is hugely inspired by and adopted from [Chun's Machine Learning Page](https://chunml.github.io/ChunML.github.io/), where the 
whole process is explained very neatly, so I would hugely recommend the lecture. 
<br>To be more specific, the two following blog posts:
<br>[Real Time Object Recognition (Part 1)](https://chunml.github.io/ChunML.github.io/project/Real-Time-Object-Recognition-part-one/)
<br>[Real Time Object Recognition (Part 2)](https://chunml.github.io/ChunML.github.io/project/Real-Time-Object-Recognition-part-two/)

Nevertheless, some of the code provided by Chun did not work for me and I had to do some adjustments. So if you struggle with his code,
maybe my implementation will help you out. It works well with Keras v2.0.0 and tensorflow v1.4 at the backend of the whole operation.

As you can see with the examples below, it works *all right*. Most of the everyday objects get recognized correctly, however sometimes
you can see silly mistakes like my automatic pencil being classified as a revolver with nearly 50% prediction accuracy, which is ResNet50's pretrained weights fault. Nevertheless, I suppose that makes it funnier to play around with :) 

## Example
![alt text](https://github.com/matatusko/Keras_RealTime_Object_Recognition/blob/master/examples.png)
