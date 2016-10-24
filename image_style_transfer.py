from IST_model import *
from load_image import *

CONTENT_IMAGE_PATH ='./data/resized_content1.jpg' 
STYLE_IMAGE_PATH = './data/resized_style3.jpg'
VGG_MODEL_PATH = './data/vgg19.npy'
SAVE_PATH = './result/'
EPOCHS = 5100
ALPHA = 1
BETA = 100
def get_style_transfer_image():

# load content image, style image,noise image,load model
    content_image = load_image(CONTENT_IMAGE_PATH)
    style_image = load_image(STYLE_IMAGE_PATH)
    input_image = generate_noise_image(content_image,0.6)
    print(content_image)
    print(input_image)
    save_image(style_image,'./result/style.png')
    save_image(content_image,'./result/content.png')
    save_image(input_image,"./result/input.png")
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

        total_loss = ALPHA * content_loss + BETA * style_loss
        optimizer= model.optimizerImage(total_loss)
# get mixed image,means style transfer image    
        init = tf.initialize_all_variables()
        sess.run(init)
        sess.run(model.graph['input'].assign(input_image))

        for i in range(EPOCHS):
            _ ,loss,style_transfer_image= sess.run([optimizer,total_loss,model.graph['input']]) 
            if i %100 ==0: 
                print('epoch %d'%(i))
                print("total_loss: ",loss) 
                filename = SAVE_PATH + 'mixed_image_%d.png'%(i)
                save_image(style_transfer_image,filename)


get_style_transfer_image()


