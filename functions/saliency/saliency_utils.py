import torch
import torch.nn.functional as F
import cv2
import numpy as np

def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda = False):
    # do the pre-processing
    predict_idx = None
    gradients = []
    for input in inputs:
        input = pre_processing(input, cuda)
        output = model(input)
        output = F.softmax(output, dim=1)

        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()

        index = np.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        if cuda:
            index = index.cuda()
        output = output.gather(1, index)

        # clear grad
        model.zero_grad()
        output.backward()
        gradient_np = input.grad.data.cpu().numpy()[0]
        gradients.append(gradient_np)
    gradients = np.array(gradients)
    return gradients, target_label_idx


def pre_processing(obs, cuda):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    obs = (obs - mean) / std
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    if cuda:
        torch_device = torch.device('cuda:0')
    else:
        torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device, requires_grad=True)
    return obs_tensor


def get_smoothed_gradients(x_values, model, target_label_idx, predict_and_gradients, cuda=False, stdev_spread=.25,
                           nsamples=10, magnitude=True):

    stdev = stdev_spread * (np.max(x_values) - np.min(x_values))
    smoothgrads = []
    for x_value in x_values:
        total_gradients = np.zeros_like(x_value)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, np.shape(x_value))
            x_plus_noise = x_value + noise
            grad, _ = predict_and_gradients([x_plus_noise], model, target_label_idx, cuda)
            grad = np.transpose(grad[0], (1, 2, 0))
            if magnitude:
                grad = np.clip(grad, 0, 1)
                total_gradients += grad * grad
            else:
                total_gradients += grad
        avg_gradients = total_gradients / nsamples
        smoothgrads.append(avg_gradients)
    return np.array(smoothgrads, dtype=np.float64)


# generate the entire images
def generate_entrie_images(img_origin, img_grad, img_grad_overlay, img_integrad, img_integrad_overlay):
    blank = np.ones((img_grad.shape[0], 10, 3), dtype=np.uint8) * 255
    blank_hor = np.ones((10, 20 + img_grad.shape[0] * 3, 3), dtype=np.uint8) * 255
    upper = np.concatenate([img_origin, blank, img_grad_overlay, blank, img_grad], 1)
    down = np.concatenate([img_origin, blank, img_integrad_overlay, blank, img_integrad], 1)
    total = np.concatenate([upper, blank_hor, down], 0)
    total = cv2.resize(total, (550, 364))

    return total
