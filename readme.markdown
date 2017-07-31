
### image style transfer in tensorflow，图片风格转换

requirement:   

1. tensorflow  
2. cuda, cudnn 加速训练   
3. VGG19 model parameter，训练好了的VGG19 model 的参数  

模型训练参数调整大部分在 `image _style_transfer.py`    
eg. `python image_style_transfer.py`  

example：  
alpha:beta = 1:100
content_image 和 style image :       
<p><img src="images/resized_content1.jpg" width="300" height="200"><img src="images/resized_stylehh.jpg" height="200" width="300"></p>
output:  	
<p><img src="images/mixed_image_5000.png" width=300 height=200></p>   
content_image 和 style image :       
<p><img src="images/resized_content1.jpg" width="300" height="200"><img src="images/resized_style3.jpg" height="200" width="300"></p>
output:  	
<p><img src="images/2_1.png" width=300 height=200></p>  

