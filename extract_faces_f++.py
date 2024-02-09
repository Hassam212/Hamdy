#!/usr/bin/env python
""" Extract faces from videos and create an index.csv file
Example usage:
    see -h
"""

# coding: utf-8

# In[35]:


import os
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
import argparse
import math


# In[36]:


# this function takes a path and "normalizes" it by replacing
# backslashes with forward slashes and trimming the last slash 
# and the leading ./ if any
# this aims to put paths in a normalized form
# - returns: normalized path
def normalize_path(dir_path):
    # make sure `dir_path` uses '/' as path separator
    dir_path = dir_path.replace('\\', '/')
    # make sure `dir_path` doesn't end with '/' (trim trailing slash)
    if dir_path.endswith('/'):
        dir_path = dir_path[:-1]
    if dir_path.startswith('./'):
        dir_path = dir_path[2:]
    # return the normalized dir_path
    return dir_path


# In[37]:


# Credits: This code is edited from https://github.com/balajisrinivas/Face-Mask-Detection


def detect_face(faceNet, frame, out_path=None, prev_coordinates=None, log_file=None):

    # calc image dimensions and create a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(224, 224),
        mean=(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # Check no face detected
    
    # get index of the face with highest confidence
    face_index = detections[0, 0, :, 2].argmax()
    confidence = detections[0, 0, face_index, 2]

    # Display warning for low confidence value
    if log_file and confidence < 0.25:
        msg = f'Warning: low face confidence value {confidence:.3f}'
        #print(msg)
        with open(log_file, 'a') as f: f.write(msg + '\n')
    
    # compute the (x, y)-coordinates of the bounding box for
    # the object
    box = detections[0, 0, face_index, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # if face is not detected (coordinates will be out of bounds)
    if (startX < 0    or startX >= w 
            or endX < 0   or endX >= w 
            or startY < 0 or startY >= h 
            or endY < 0   or endY >= h
            or endX <= startX
            or endY <= startY):
        # if prev_coordinates given, use them
        if prev_coordinates:
            (startX, startY, endX, endY) = prev_coordinates
            if log_file:
                msg = f'Warning: No faces were found in image "{out_path}", previous coordinates used'
                with open(log_file, 'a') as f: f.write(msg + '\n')
        # otherwise, raise an error
        else:
            raise ValueError(f'No faces were found in image which was to be saved in "{out_path}", and no previous coordinates given')

    # extract the face ROI
    if out_path:
        face = frame[startY:endY, startX:endX]
        if not cv2.imwrite(out_path, face):
            raise IOError(f"Can't write to '{out_path}'")
        
    return (startX, startY, endX, endY)


# In[38]:

def get_first_face_coordinates(faceNet, video_path):
    # capture video from given file path
    vidcap = cv2.VideoCapture(video_path)
    # read first frame
    success, image = vidcap.read()
    # while frame is successfully read
    while success:
        # extract face from frame and get its coordinates
        try:
            return detect_face(faceNet, image)
        except ValueError:
            pass
        # read next frame (if we can't extract face from current frame)
        success, image = vidcap.read()
    raise ValueError(f"Can't find any faces inside video '{video_path}'")
    
    
def extract_back_block(face_coordinates, image, dst_path):
    '''
        Extract a block from the background of image. The block is the farthest block
        from face_coordinates and has the same size as the face. The extracted block
        is saved to dst_path
        
        face_coordinates format: (x1, y1, x2, y2)
    '''
    # calculate face coordinates
    face_x1, face_y1, face_x2, face_y2 = face_coordinates
    face_cx = (face_x1 + face_x2)//2
    face_cy = (face_y1 + face_y2)//2
    # calculate block width and height (same as face width and height)
    block_width = face_x2 - face_x1
    block_height = face_y2 - face_y1
    # get the width and height of image
    img_height, img_width = image.shape[:2]
    # define coordinates of candidate blocks (the 4 blocks at the corners of the frame)
    candidate_blocks = [
        # top left corner
        {
            'x1':0, 
            'y1':0, 
            'x2':block_width, 
            'y2':block_height,
        },
        # top right corner
        {
            'x1':img_width-block_width, 
            'y1':0, 
            'x2':img_width, 
            'y2':block_height,
        },
        # bottom left corner
        {
            'x1':0, 
            'y1':img_height-block_height, 
            'x2':block_width, 
            'y2':img_height,
        },
        # bottom right corner
        {
            'x1':img_width-block_width, 
            'y1':img_height-block_height, 
            'x2':img_width, 
            'y2':img_height,
        }, 
    ]
    # calculate the Euclidean distance between (face_cx, face_cy) and 
    # each candidate block's center (x, y)
    for b in candidate_blocks:
        block_cx = (b['x1']+b['x2'])//2
        block_cy = (b['y1']+b['y2'])//2
        distance = math.sqrt((face_cx - block_cx)**2 + (face_cy + block_cy)**2)
        b['distance'] = distance
    # sort candidate_blocks by distance (ascending)
    candidate_blocks.sort(key=lambda b: b['distance'], reverse=True)
    # crop the best candidate block (the one with largest distance)
    block = candidate_blocks[0]
    x1, x2, y1, y2 = block['x1'], block['x2'], block['y1'], block['y2']
    cropped_block = image[y1:y2, x1:x2]
    if not cv2.imwrite(dst_path, cropped_block):
        raise IOError(f"Can't write to '{dst_path}'")


def extract_faces_from_video(faceNet, video_path, dst_dir, extract_background_blocks):
    '''
        takes a path to a video and extracts faces from it to jpg images in dst_dir
        the images are named as follows: '00001f.jpg', '00002f.jpg', '00003f.jpg', etc.
        where 'f' stands for face.
        
        If the argument "extract_background_blocks" is True, this function will also
        extract a block from the background (the farthest block from the face) with
        the same size as the face. The saved block image will have the same name as the
        face image but ending with 'b' instead of 'f'.
        
        params: 
         - video_path: the path of the video
         - dst_dir: the path to the directory where the face images will be written
         - extract_background_blocks (boolean): Whether or not to extract blocks from
           the background as well.
        returns:
         - a dict with keys: 
            - 'face': A list of saved face image filenames (with extension)
            - 'back' (only if extract_background_blocks is True): A list of saved back block 
              image filenames (with extension)
    '''
    # check if video exists
    if not os.path.exists(video_path):
        raise ValueError(f'Video "{video_path}" not found')
    # normalize dst_dir by using '/' as path separator and trimming the 
    # last '/'
    dst_dir = normalize_path(dst_dir)
    # create destination directory if not existing
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # get coordinates of first face and store them in prev_coordinates
    # previous face coordinates (used as a fallback solution when no face detected)
    prev_coordinates = get_first_face_coordinates(faceNet, video_path)
    # capture video from given file path
    vidcap = cv2.VideoCapture(video_path)
    # read first frame
    success, image = vidcap.read()
    # accumulator to store number of read frames
    count = 1
    # keep lists of saved image paths
    img_face_path_list = []
    img_back_path_list = []
    # while frame is successfully read
    while success:
        # destination filename (00001f.jpg', '00002f.jpg', '00003f.jpg', etc.)
        img_face_fname = f'{count:05}f.jpg'
        img_back_fname = f'{count:05}b.jpg'
        # obtain the full path for destination file
        dst_face_path = f'{dst_dir}/{img_face_fname}'
        dst_back_path = f'{dst_dir}/{img_back_fname}'
        # extract face from frame and write the output image to disk
        prev_coordinates = detect_face(
            faceNet, 
            image, 
            dst_face_path, 
            prev_coordinates
        )
        # add the path of the image to img_face_path_list
        img_face_path_list.append(img_face_fname)
        # extract back block from frame and write the output image to disk 
        if extract_background_blocks:
            extract_back_block(prev_coordinates, image, dst_back_path)
            img_back_path_list.append(img_back_fname)
        # read next frame (if any)
        success, image = vidcap.read()
        # increment number of read frames
        count += 1
    # return a list of saved frames' paths
    return {'face': img_face_path_list, 'back': img_back_path_list}


# In[39]:


def extract_faces_from_dataset(
        faceNet, 
        src_dir, 
        dst_dir, 
        compression, 
        output_csv_path
):
    '''
        Takes the path of the dataset (videos) directory that contains
        "manipulated_sequences" and "original_sequences" and extracts faces
        from all videos. The extracted faces are saved inside the given dst_dir
        inside a sub-directory for each video.
        
        Also saves an index.csv file which contains data about the extracted images 
        (video_key, path, output)
    '''
    # Loop through the videos to create the following lists:
    # - videos_keys: unique keys to identify each video
    # - videos_src_paths: list of videos source paths
    # - videos_dst_paths: list of paths where we want to save extracted faces
    # - videos_outputs: list of outputs: 0 for real and 1 for fake videos
    outputs_dict = {'manipulated_sequences': 1, 'original_sequences': 0}
    videos_keys = []
    videos_src_paths = []
    videos_dst_paths = []
    videos_outputs = []
    for class_dir, output in outputs_dict.items():
        class_dir_path = os.path.join(src_dir, class_dir)
        for class_subdir in os.listdir(class_dir_path):
            class_subdir_videos_path = os.path.join(
                class_dir_path, class_subdir, compression, 'videos')
            class_subdir_videos_path = normalize_path(class_subdir_videos_path)
            for video_fname in os.listdir(class_subdir_videos_path):
                video_path = class_subdir_videos_path + '/' + video_fname
                video_number = os.path.splitext(video_fname)[0]
                video_key = f"{class_subdir}-{compression}-{video_number}" 
                videos_keys.append(video_key)
                videos_src_paths.append(video_path)
                videos_dst_paths.append(
                    normalize_path(
                        os.path.join(dst_dir, class_dir, class_subdir, video_number)
                    )
                )
                videos_outputs.append(output)
    
    # Loop through the previous lists in parallel to process each video
    all_frames_keys = []
    all_frames_face_paths = []
    all_frames_back_paths = []
    all_frames_outputs = []
    for key, src_path, dst_path, output in tqdm(list(zip(
            videos_keys, videos_src_paths,  videos_dst_paths, videos_outputs))):
        saved_img_paths = extract_faces_from_video(faceNet, src_path, dst_path, True)
        all_frames_keys.extend([key for _ in range(len(saved_img_paths['face']))])
        all_frames_face_paths += [normalize_path(os.path.join(dst_path, f)) for f in saved_img_paths['face']]
        all_frames_back_paths += [normalize_path(os.path.join(dst_path, f)) for f in saved_img_paths['back']]
        all_frames_outputs.extend([output for _ in range(len(saved_img_paths['face']))])
    
    # Write the csv index file
    pd.DataFrame({
        'key': all_frames_keys,
        'face_path': all_frames_face_paths,
        'back_path': all_frames_back_paths,
        'output': all_frames_outputs,
    }).to_csv(output_csv_path, index=False)


# In[40]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src_dir", 
        help=(
            "path to dataset videos dir which contains manipulated_sequences"
            " and original_sequences folders"
        ),
    )
    parser.add_argument(
        "dst_dir", 
        help="path to output folder where the extracted faces will be placed.",
    )
    parser.add_argument(
        "-i", "--index", 
        help="path to output index csv file",
        type=str,
        default='index.csv',
    )
    parser.add_argument(
        "-f", "--facenet-dir", 
        help="path to faceNet directory which contains the .prototxt and .caffemodel files",
        type=str,
        default='face-detector/faceNet',
    )
    parser.add_argument(
        "-c", "--compression-type", 
        help="Type of compression",
        type=str,
        choices=('c23', 'c40'),
        default='c23',
    )
    args = parser.parse_args()

    # read faceNet model
    prototxtPath = os.path.join(args.facenet_dir, "deploy.prototxt")
    weightsPath = os.path.join(args.facenet_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # extract faces from dataset
    extract_faces_from_dataset(
        faceNet, 
        src_dir=args.src_dir, 
        dst_dir=args.dst_dir, 
        compression=args.compression_type,
        output_csv_path=args.index
    )


# In[ ]:




