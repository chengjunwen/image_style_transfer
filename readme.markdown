
### image style transfer in tensorflow，图片风格转换

requirement:   

1. tensorflow  
2. cuda, cudnn 加速训练   
3. VGG19 model parameter，训练好了的VGG19 model 的参数  

eg. `python image_style_transfer.py`

example：  
content_image:
![contentimage](http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/resized_content1_zpsj53o8ogq.jpg)
style_image:
![styleimage](http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/resized_stylehh_zpskjo0hdyl.jpg)  
output:  	
![](http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/mixed_image_5000_zps9o7uxqpm.png)
