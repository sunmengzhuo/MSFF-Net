img_w = 512
img_h = 512

dataset_params = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True
}

learning_rate = 1e-6
step_size = 70
gamma = 0.5
epoches = 200
save_path = 'Result/weights/'
