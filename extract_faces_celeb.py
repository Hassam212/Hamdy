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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


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

def get_videos_df(src_dir, balanced):
    '''
        Return a DataFrame object with the following columns:
         - video_key: in format <category>-<filename> without extension
         - video_src_path: path to video relative to dataset directory
         - video_dst_path: path to directory where the video images will be saved 
           relative to the output folder (which will be specified in another function)
         - is_train: in {0, 1}
         - is_val: in {0, 1}
         - is_test: in {0, 1}
         - video_output: in {0, 1}
         
        The src_dir is the folder containing the dataset. It must contain 3 folders:
        'Celeb-real', 'Celeb-synthesis', and 'YouTube-real', and a text file: 
        'List_of_testing_videos.txt' which has two columns: output, and video path
        (the 2 columns are separated by a space).
        
        The test videos are the videos whose path is inside the 'List_of_testing_videos.txt'
        file. The remaining videos are under-sampled (if the argument `balanced` is True) 
        and are randomly split into 70% train and 30% validation.
    '''
    # get the path to the 'List_of_testing_videos.txt' file
    test_videos_txt_file_path = os.path.join(src_dir, 'List_of_testing_videos.txt')
    # read the 'List_of_testing_videos.txt' file as a DataFrame object with 2 columns: 
    # (video_output, video_src_path)
    test_df = pd.read_csv(test_videos_txt_file_path, sep=' ', names=['video_output', 'video_src_path'])
    # invert the output of test (because in List_of_testing_videos.txt file, the 0 is for fake and 1 is for real)
    test_df['video_output'] = 1 - test_df['video_output']
    # create a list of test videos' paths
    test_videos_paths = test_df['video_src_path'].values
    # define category_outputs_dict to use it to loop through folders and read non-test video paths
    category_outputs_dict = {
        'Celeb-real': 0,
        'Celeb-synthesis': 1,
        'YouTube-real': 0,
    }
    # loop through video folders (categories) using the category_outputs_dict
    train_val_paths = []
    train_val_outputs = []
    for category, output in category_outputs_dict.items():
        category_dir_path = os.path.join(src_dir, category)
        for video in os.listdir(category_dir_path):
            video_rel_path = normalize_path(os.path.join(category, video))
            if video_rel_path not in test_videos_paths:
                train_val_paths.append(video_rel_path)
                train_val_outputs.append(output)
    # create a DataFrame object containing non-test videos
    train_val_df = pd.DataFrame({
        'video_output': train_val_outputs, 
        'video_src_path': train_val_paths
    })
    # check under-sampling to make train_val dataset balanced
    if balanced:
        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=0)
        train_val_df, _ = rus.fit_resample(train_val_df, train_val_df['video_output'])
    # splitting the train_val_df into train and validation
    train_df, val_df = train_test_split(train_val_df, test_size=0.30, random_state=0)
    # adding 'part' column {train, val, test}
    train_df['part'] = 'train'
    val_df['part'] = 'val'
    test_df['part'] = 'test'
    # concat train, val, test dataframes
    df_all = pd.concat([train_df, val_df, test_df])
    # convert 'part' column to one-hot (is_train, is_val, is_test)
    df_all = pd.get_dummies(df_all, columns=['part'], prefix='is')
    # create video_key in format <category>-<filename> without extension
    df_all['video_key'] = df_all['video_src_path'].apply(
        lambda path: os.path.split(path)[0] + '-' + os.path.splitext(os.path.basename(path))[0]
    )
    # create video_dst_path column
    df_all['video_dst_path'] = df_all['video_src_path'].apply(
        lambda path: os.path.splitext(path)[0]
    )
    # re-order columns
    df_all = df_all[[
        'video_key',
        'video_src_path',
        'video_dst_path',
        'is_train',
        'is_val',
        'is_test',
        'video_output',
    ]]
    return df_all
                
    


def extract_faces_from_dataset(
        faceNet, 
        src_dir, 
        dst_dir, 
        extract_background_blocks,
        balanced,
):
    '''
        Takes the path of the dataset (videos) directory that contains category folders 
        (i.e. "Celeb-real", "Celeb-synthesis", and "YouTube-real") and the 
        'List_of_testing_videos.txt' file, and extracts faces from all videos. 
        The extracted faces are saved inside the given dst_dir inside a sub-directory 
        for each category, and a sub-sub-directory for each video.
        
        This function saves an index.csv file inside the dst_dir which contains data about 
        the extracted images (video_key, face_path, is_train, is_val, is_test, output)
        
        If extract_background_blocks equals True, this function will extract a block from 
        the background of each frame with the same size as the face, the block will be saved 
        in the same folder as the extracted face with filename ending with 'b' (whereas the 
        face ending with 'f'). and a new column "back_path" will be added to the index.csv file.
        
        If balanced equals True, the non-test videos will be randomly under-sampled to convert
        the dataset to balanced. Otherwise, the function will process all videos inside
        src_dir.
    '''
    # generate a dataframe of videos (with option to make the dataset balanced)
    # (see get_videos_df function for documentation)
    videos_df = get_videos_df(src_dir, balanced)
    
    # Loop through the videos_df to process each video
    all_frames_keys = []
    all_frames_face_paths = []
    all_frames_back_paths = []
    all_frames_is_train = []
    all_frames_is_val = []
    all_frames_is_test = []
    all_frames_outputs = []
    for i, row in tqdm(list(videos_df.iterrows())):
        key = row['video_key']
        src_path = row['video_src_path']
        dst_path = row['video_dst_path']
        output = row['video_output']
        is_train = row['is_train']
        is_val = row['is_val']
        is_test = row['is_test']
        saved_img_paths = extract_faces_from_video(
                faceNet, 
                os.path.join(src_dir, src_path), 
                os.path.join(dst_dir, dst_path), 
                extract_background_blocks
        )
        saved_face_img_paths = saved_img_paths['face']
        saved_back_img_paths = []
        if extract_background_blocks:
            saved_back_img_paths = saved_img_paths['back']
        all_frames_keys.extend([key for _ in range(len(saved_face_img_paths))])
        all_frames_face_paths += [normalize_path(os.path.join(dst_path, f)) for f in saved_face_img_paths]
        all_frames_back_paths += [normalize_path(os.path.join(dst_path, f)) for f in saved_back_img_paths]
        all_frames_is_train.extend([is_train for _ in range(len(saved_face_img_paths))])
        all_frames_is_val.extend([is_val for _ in range(len(saved_face_img_paths))])
        all_frames_is_test.extend([is_test for _ in range(len(saved_face_img_paths))])
        all_frames_outputs.extend([output for _ in range(len(saved_face_img_paths))])
    
    # Write the csv index file
    csv_data_dict = {}
    csv_data_dict['key'] = all_frames_keys
    csv_data_dict['face_path'] = all_frames_face_paths
    if extract_background_blocks:
        csv_data_dict['back_path'] = all_frames_back_paths
    csv_data_dict['is_train'] = all_frames_is_train
    csv_data_dict['is_val'] = all_frames_is_val
    csv_data_dict['is_test'] = all_frames_is_test
    csv_data_dict['output'] = all_frames_outputs
    pd.DataFrame(csv_data_dict).to_csv(os.path.join(dst_dir, 'index.csv'), index=False)


# In[40]:


if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src_dir", 
        help=(
            "path to dataset videos dir which contains Celeb-real, "
            " Celeb-synthesis, and YouTube-real folders"
        ),
    )
    parser.add_argument(
        "dst_dir", 
        help="path to output folder where the extracted faces will be placed.",
    )
    parser.add_argument(
        "-f", "--facenet-dir", 
        help="path to faceNet directory which contains the .prototxt and .caffemodel files",
        type=str,
        default='face-detector/faceNet',
    )
    parser.add_argument(
        "-b", "--extract-back-blocks", 
        help="Whether or not (1 or 0) to extract a block from the background of each frame",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--balanced", 
        help="Whether or not (1 or 0) to make the dataset balanced by under-sampling non-test videos",
        type=int,
        choices=(0, 1),
        default=1,
    )
    args = parser.parse_args()
    if args.extract_back_blocks == 1:
        args.extract_back_blocks = True
    else:
        args.extract_back_blocks = False
    if args.balanced == 1:
        args.balanced = True
    else:
        args.balanced = False

    # read faceNet model
    prototxtPath = os.path.join(args.facenet_dir, "deploy.prototxt")
    weightsPath = os.path.join(args.facenet_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # extract faces from dataset
    extract_faces_from_dataset(
        faceNet, 
        src_dir=args.src_dir, 
        dst_dir=args.dst_dir,
        extract_background_blocks=args.extract_back_blocks,
        balanced=args.balanced,
    )


# In[ ]:




