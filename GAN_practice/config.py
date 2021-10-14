import torch



configurations = {
    1: dict(
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

        BATCH_SIZE = 128,
        MODEL_ROOT = "/opt/data/private/codeface/GAN_practice/model_minist",  # the root to buffer your checkpoints
        Generator_root = "/opt/data/private/codeface/GAN_practice/Generator_image",  # the root to buffer your generated image
        NUM_EPOCH = 40,
        LR = 0.0003,
        Z_DIMENSION = 100,  # the dimension of noise tensor
    ),
}