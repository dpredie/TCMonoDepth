#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import os

import cv2
import torch
from torchvision.transforms import Compose
import time

from networks.transforms import Resize
from networks.transforms import PrepareForNet
from tqdm import tqdm

inputfile = "/content/inputs/input.mp4"
outputfile= "/content/output/output.mp4"
fps=24.0


def write_video(filename, output_list, fpsz):
    assert (len(output_list) > 0)
    h, w = output_list[0].shape[0], output_list[0].shape[1]
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fpsz, (w, h))

    for img in output_list:
        writer.write(img)
    writer.release()

    return


def process_depth(dep):
    dep = dep - dep.min()
    dep = dep / dep.max()
    dep_vis = dep * 255

    return dep_vis.astype('uint8')


def load_video_paths(args):
    root_path = args.input
    scene_names = sorted(os.listdir(root_path))
    path_lists = []
    for scene in scene_names:
        frame_names = sorted(os.listdir(os.path.join(root_path, scene)))
        frame_paths = [os.path.join(root_path, scene, name) for name in frame_names]
        path_lists.append(frame_paths)
        print("Device: %s" % frame_paths)

    return path_lists, scene_names


def run(args):
    print("Initialize")

    # select device
    device = torch.device("cuda")
    print("Device: %s" % device)

    fps = args.fps
    outputfile = args.output_file
    inputfile = args.input_file

    # load network
    print("Creating model...")
    if args.model == 'large':
        from networks import MidasNet
        model = MidasNet(args)
    else:
        from networks import TCSmallNet
        model = TCSmallNet(args)

    if os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print("Loading model from " + args.resume)
    else:
        print("Loading model path fail, model path does not exists.")
        exit()

    model.cuda().eval()
    print("Loading model done...")

    transform = Compose([
        Resize(
            args.resize_size,  #width
            args.resize_size,  #height
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        PrepareForNet(),
    ])

    # get input
    #path_lists, scene_names = load_video_paths(args)

 
    # prepare output folder
    #os.makedirs(args.output, exist_ok=True)

    # Initialize video
    cap = cv2.VideoCapture(inputfile)

    print(cap)
    #out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc(*'MP4V'), fps,(int(cap.get(3)),int(cap.get(4))))

    if (cap.isOpened()== False):
      print("Error opening video stream or file")


    output_list = []
    with torch.no_grad():
      framecnt = 0
      while cap.isOpened():
        # Read frame from the video
        ret, img = cap.read()
        if framecnt == 0:
          img0 = img
      
        if ret:  
          start = time.time()
      
        
          frame = img
          #frame = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
          frame = transform({"image": frame})["image"]
          frame = torch.from_numpy(frame).to(device).unsqueeze(0)

          prediction = model.forward(frame)
          print(prediction.min(), prediction.max())
          prediction = (torch.nn.functional.interpolate(
              prediction,
              size=img0.shape[:2],
              mode="bicubic",
              align_corners=False,
          ).squeeze().cpu().numpy())
          output_list.append(prediction)


          end = time.time()
          print("processed frame: "+str(framecnt)+" "+str("%.2f" % (end-start))+"s")
          framecnt+=1
        else:
          print("image empty - exiting")    
          break  
        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
          break

      # save output
      #output_name = os.path.join(args.output_file, scene_names[i] + '.mp4')
      output_list = [process_depth(out) for out in output_list]

      color_list = []
      for j in range(len(output_list)):
          frame_color = cv2.applyColorMap(output_list[j], cv2.COLORMAP_INFERNO)
          color_list.append(frame_color)
          print("generating Depth Color frame "+str(j) ) 

      write_video(outputfile, color_list, fps)
      cap.release()


      cv2.destroyAllWindows()    
    print(args.outputfile + " Done.")





if __name__ == "__main__":
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Settings
    parser = argparse.ArgumentParser(description="A PyTorch Implementation of Video Depth Estimation")

    parser.add_argument('--model', default='large', choices=['small', 'large'], help='size of the model')
    parser.add_argument('--resume', type=str, required=True, help='path to checkpoint file')
    #parser.add_argument('--input', default='./videos', type=str, help='video root path')
    #parser.add_argument('--output', default='./output', type=str, help='path to save output')
    parser.add_argument('--resize_size',
                        type=int,
                        default=384,
                        help="spatial dimension to resize input (default small model256, large model384)")
    parser.add_argument('--output_file', type=str, required=True, help='copy path from output directory')
    parser.add_argument('--input_file', type=str, required=True, help='copy path from input file')
    parser.add_argument('--fps', type=float, required=True, help='fps value')                        

    args = parser.parse_args()

    print("Run Video Depth Sample ")
    run(args)
