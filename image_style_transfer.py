from IST_model import *
from load_image import *

# conrent image path, style image path
CONTENT_IMAGE_PATH ='./data/resized_lss1.jpg' 
STYLE_IMAGE_PATH = './data/resized_style3.jpg'

# vgg19 model parameter
VGG_MODEL_PATH = './data/vgg19.npy'

# generated image save path
SAVE_PATH = './result/'
# train epochs
EPOCHS = 1000

# content loss and style loss ratio
ALPHA = 1
BETA = 100



def get_style_transfer_image():

# load content image, style image,noise image,load model
    content_image = load_image(CONTENT_IMAGE_PATH)
    style_image = load_image(STYLE_IMAGE_PATH)
    input_image = generate_noise_image(content_image,0.6)
    model = Model(VGG_MODEL_PATH)
    model.constructModel()

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
# get contentLoss and styleLoss        
        sess.run(model.graph['input'].assign(content_image))
        content_loss = model.get_content_loss(sess)
        sess.run(model.graph['input'].assign(style_image))
        style_loss = model.get_style_loss(sess)
# define loss and optimizer 
        total_loss = ALPHA * content_loss + BETA * style_loss
        optimizer= model.optimizerImage(total_loss)
# set input image(noise image)    
        init = tf.initialize_all_variables()
        sess.run(init)
        sess.run(model.graph['input'].assign(input_image))
# train
        for i in range(EPOCHS):
            _ ,loss,style_transfer_image= sess.run([optimizer,total_loss,model.graph['input']]) 
            if i %100 ==0: 
                print('epoch %d'%(i))
                print("total_loss: ",loss) 
                filename = SAVE_PATH + 'mixed_image_%d.png'%(i)
                save_image(style_transfer_image,filename)


get_style_transfer_image()


