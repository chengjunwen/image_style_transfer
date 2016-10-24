
### image style transfer in tensorflow，图片风格转换

requirement:   

1. tensorflow  
2. cuda, cudnn 加速训练   
3. VGG19 model parameter，训练好了的VGG19 model 的参数  

模型训练参数调整大部分在 `image _style_transfer.py`  
另外需要在`IST_model.py`里面设置 image_width 和image_height。
eg. `python image_style_transfer.py`

example：  
alpha:beta = 1:100
content_image 和 style image :       
<p><img src="http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/resized_content1_zpsj53o8ogq.jpg" width="300" height="200"><img src="http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/resized_stylehh_zpskjo0hdyl.jpg" height="200" width="300"></p>  
output:  	
<p><img src="http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/mixed_image_5000_zps9o7uxqpm.png" width=300 height=200></p>   
content_image 和 style image :       
<p><img src="http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/resized_content1_zpsj53o8ogq.jpg" width="300" height="200"><img src="http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/resized_style3_zpsftnh3fi4.jpg" height="200" width="300"></p>  
output:  	
<p><img src="http://i1156.photobucket.com/albums/p568/chengjunwen/image%20style%20transfer/2_1_zpswyz5oqpj.png" width=300 height=200></p>  

#alpha:beta = 3:100
#content_image 和 style image :       
#<p><img src="" width="300" height="200"><img src="" height="200" width="300"></p>  
#output:  	
#<p><img src="" width=300 height=200></p> 	  
