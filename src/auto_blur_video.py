# author: Asmaa Mirkhan ~ 2019

import os
import argparse
import cv2
import numpy as np
from yunet import YuNet


def blurBoxes(image, boxes, blur_strength, extend_selection):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """

    for box in (boxes if boxes is not None else []):
        # unpack each box
        box = box[0:4].astype(np.int32)
        x1, y1 = box[0], box[1]
        x2, y2 = box[0]+box[2], box[1]+box[3]

        height, width, _ = image.shape

        x1 = max(0, x1 - extend_selection)
        x2 = min(width, x2 + extend_selection)
        y1 = max(0, y1 - extend_selection)
        y2 = min(height, y2 + extend_selection)

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (blur_strength, blur_strength))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image


def main(args):
    # assign model path and threshold
    model_path = args.model_path
    threshold = args.threshold
    blur_strength = args.blur_strength
    extend_selection = args.extend_selection
    backend = args.backend
    target = args.target
    nms_threshold = args.nms_threshold
    top_k = args.top_k

    # Instantiate YuNet
    model = YuNet(modelPath=model_path,
                  inputSize=[320, 320],
                  confThreshold=threshold,
                  nmsThreshold=nms_threshold,
                  topK=top_k,
                  backendId=backend,
                  targetId=target)

    # open video
    capture = cv2.VideoCapture(args.input_video)

    width = int(capture.get(3))
    height = int(capture.get(4))
    fps = capture.get(5)

    model.setInputSize([width, height])

    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output = cv2.VideoWriter(
            args.output_video, fourcc, fps, (width, height)
        )

    frame_counter = 0
    while True:
        # read frame by frame
        _, frame = capture.read()
        frame_counter += 1

        # the end of the video?
        if frame is None:
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        # real face detection
        faces = model.infer(frame)

        # apply blurring
        frame = blurBoxes(frame, faces, blur_strength, extend_selection)

        # show image
        cv2.imshow('blurred', frame)

        # if image will be saved then save it
        if args.output_video:
            output.write(frame)

    print('Blurred video has been saved successfully at',
          args.output_video, 'path')

    # when any key has been pressed then close window and stop the program

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # creating argument parser
    parser = argparse.ArgumentParser(description='Image blurring parameters')

    backends = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA]
    targets = [cv2.dnn.DNN_TARGET_CPU,
               cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16]
    help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
    help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
    try:
        backends += [cv2.dnn.DNN_BACKEND_TIMVX]
        targets += [cv2.dnn.DNN_TARGET_NPU]
        help_msg_backends += "; {:d}: TIMVX"
        help_msg_targets += "; {:d}: NPU"
    except:
        print('This version of OpenCV does not support TIM-VX and NPU. Visit https://gist.github.com/fengyuentau/5a7a5ba36328f2b763aea026c43fa45f for more information.')

    # adding arguments
    parser.add_argument('-i',
                        '--input_video',
                        help='Path to your video',
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to .pb model',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output_video',
                        help='Output file path',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help='Face detection confidence',
                        default=0.9,
                        type=float)
    parser.add_argument('-s',
                        '--blur_strength',
                        help='Blur strength, default 25',
                        default=25,
                        type=int)
    parser.add_argument('-e',
                        '--extend_selection',
                        help='Extend the selected area by x amount of pixels',
                        default=0,
                        type=int)
    parser.add_argument('--backend', type=int,
                        default=backends[0], help=help_msg_backends.format(*backends))
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                        help='Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type=int, default=5000,
                        help='Keep top_k bounding boxes before NMS.')
    parser.add_argument('--target', type=int,
                        default=targets[0], help=help_msg_targets.format(*targets))
    args = parser.parse_args()

    # if input image path is invalid then stop
    assert os.path.isfile(args.input_video), 'Invalid input file'

    # if output directory is invalid then stop
    if args.output_video:
        assert os.path.isdir(os.path.dirname(
            args.output_video)), 'No such directory'

    main(args)
