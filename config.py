#可以根据自己的情况进行修改
class MyConfigs():

    data_folder = '../uefi/AccuEmoji/DataSet/train/l1/'
    test_data_folder = ""
    model_name = "savemodel" 
    weights = "./checkpoints/"
    logs = "./logs/"
    example_folder = "./example/"
    freeze = True
    #
    epochs = 9999991
    batch_size = 16
    img_height = 96  #网络输入的高和宽
    img_width = 96
    num_classes = 4
    lr = 0.004
    lr_decay = 0.00001
    weight_decay = 2e-4
    ratio = 0.3
config = MyConfigs()
