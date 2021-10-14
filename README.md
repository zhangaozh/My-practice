# My-preactice
此部分主要备份了我自己的一些算法练习，目前只上传了一小部分，另一部分后续会更新

关于GAN练习的小demo:

python3 GAN_practice/gan.py

配置超参在GAN_practice/config.py, backbone在GAN_practice/model.py, 生成的效果图在GAN_practice/Generator_image, 模型在GAN_practice/model_mnist; 生成器和判别器分别用了简单的四层全连接，在判别器最后一层用了Sigmoid激活，这是GAN所用的激活函数(通常大家不会这些，大家会把sigmoid函数和GAN_loss绑定在一起，因为不同的GAN的激活函数不同)；

具体训练过程：我每个eopch，判别器训4次，生成器训1次；在训判别器时，生成器权重要冻结；训生成器时，判别器权重不冻结(因为生成器的最终目的是要骗过判别器)
