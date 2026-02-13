from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from datascaler_open_clip.filter_utils.easyocr.craft import CRAFT
from datascaler_open_clip.filter_utils.easyocr.craft_utils import adjustResultCoordinates, getDetBoxes
from datascaler_open_clip.filter_utils.easyocr.imgproc import normalizeMeanVariance, resize_aspect_ratio

def transform_batch(batch: torch.Tensor) -> torch.Tensor:
    batch = TF.resize(batch, size=[608, 608], interpolation=TF.InterpolationMode.BICUBIC)
    CLIP_MEAN = torch.tensor([0.485*255, 0.456*255, 0.406*255], device=batch.device).view(
        1, 3, 1, 1
    )
    CLIP_STD = torch.tensor([0.229*255, 0.224*255, 0.225*255], device=batch.device).view(
        1, 3, 1, 1
    )

    return (batch - CLIP_MEAN) / CLIP_STD

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(
    canvas_size,
    mag_ratio,
    net,
    image,
    text_threshold,
    link_threshold,
    low_text,
    poly,
    device,
    estimate_num_chars=False,
):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:  # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize

    ratio_h = 1
    ratio_w = 1

    im_b = np.array(image_arrs).swapaxes(1, 3)
    batch = torch.from_numpy(im_b)
    x = transform_batch(batch.to(device))

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()
        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text,
            score_link,
            text_threshold,
            link_threshold,
            low_text,
            poly,
            estimate_num_chars,
        )

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    return boxes_list, polys_list


def get_detector(trained_model, device="cpu", quantize=True, cudnn_benchmark=False):
    net = CRAFT()
    if device == "cpu":
        net.load_state_dict(
            copyStateDict(torch.load(trained_model, map_location=device, weights_only=False))
        )
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(
            copyStateDict(torch.load(trained_model, map_location=device, weights_only=False))
        )
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net


def get_textbox(
    detector,
    image,
    canvas_size,
    mag_ratio,
    text_threshold,
    link_threshold,
    low_text,
    poly,
    device,
    optimal_num_chars=None,
    **kwargs,
):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(
        canvas_size,
        mag_ratio,
        detector,
        image,
        text_threshold,
        link_threshold,
        low_text,
        poly,
        device,
        estimate_num_chars,
    )

    if estimate_num_chars:
        polys_list = [
            [p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
            for polys in polys_list
        ]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)

    return result
