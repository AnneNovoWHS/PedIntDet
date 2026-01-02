import os
import re
import json

import torch
from tqdm import tqdm
import cv2
    
##################################################################################################

def get_annotations(file_path):
    data = json.load(open(file_path))
    actions = data["openlabel"]["actions"]

    records_blinks, records_yawning = [], []

    for action in actions.values():
        if 'blinks' in action["type"]:
            records_blinks = [[seg["frame_start"], seg["frame_end"]] for seg in action.get("frame_intervals", [])]
        if 'yawning' in action["type"]:
            records_yawning = [[seg["frame_start"], seg["frame_end"]] for seg in action.get("frame_intervals", [])]
       
    return records_blinks, records_yawning

##################################################################################################

def make_clips_and_annotations(video, records_blinks, records_yawns, clip_len_s):
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_len_f = int(clip_len_s * fps)

    # from 720x1280 to 360x640
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))//2
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

    # all segments that are not ok to cut
    forbidden = set()
    for a, b in records_blinks + records_yawns:
        if b - a > 1: forbidden.update(range(a+1, b))

    # Collect only fully-contained events per clip
    ws = int(10 * fps) # skip the first 15 seconds 
    we = ws+clip_len_f-1  
    shift = 0
    i = 0
    while we+shift < total_frames:

        # extend the frames till the event fits in the video
        while (we+shift in forbidden) or (ws+shift in forbidden):
            shift += 3

            # if there is not enough data in the video for another clip end 
            if (we+shift) >= total_frames: 
                return True
        
        # get start and end frame
        s0, e0 = ws + shift, we + shift

        # create video writer
        video.set(cv2.CAP_PROP_POS_FRAMES, s0)
        clip_path = os.path.join(out_dir, f"{file_name}_clip_{i:02d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height)) 
        # save video
        for _ in range(s0, e0+1):
            ok, frame = video.read()
            if not ok:  break  # graceful exit if the stream ends early
            resized = cv2.resize(frame, (width, height))
            writer.write(resized)
        writer.release()
        

        # get annotations
        clip_blinks = [(s - s0, e - s0) for (s, e) in records_blinks if s0 <= s and e <= e0]
        clip_yawns  = [(s - s0, e - s0) for (s, e) in records_yawns  if s0 <= s and e <= e0]
        torch.save({"blinks": clip_blinks, "yawns": clip_yawns}, 
                   os.path.join(out_dir, f"{file_name}_clip_{i:02d}_ann.pt"))
        
        # move windows for next clip
        ws = ws+clip_len_f
        we = we+clip_len_f
        i = i+1

    return True

##################################################################################################

def to_snake(path_or_stem: str) -> str:
    stem = os.path.splitext(os.path.basename(path_or_stem))[0]
    snake = re.sub(r'[^A-Za-z0-9]+', '_', stem).strip('_').lower()
    return snake

##################################################################################################

if __name__ == "__main__":

    # get all video files in the dataset folder
    path = "/home/datasets/dmd/"

    file_paths_videos = []
    file_paths_annotations = []
    for dirpath, dirnames, filenames in os.walk(path):
        if 'processed' in dirpath: continue
        for fname in filenames:
            if "rgb_face" in fname and "mp4" in fname:
                file_paths_videos.append(dirpath+'/'+fname)
            if "rgb_ann" in fname and 'hands' not in fname and "json" in fname:
                file_paths_annotations.append(dirpath+'/'+fname)

    file_paths_videos.sort()
    file_paths_annotations.sort()
    assert len(file_paths_videos) == len(file_paths_annotations), "mismatch"

    #=================================================================================

    # make output dir once
    out_dir = os.path.join(path, "processed")
    os.makedirs(out_dir, exist_ok=True)

    #=================================================================================

    # go thorugh all videos and annotations
    session_names = []
    for video_path, ann_path in tqdm(zip(file_paths_videos, file_paths_annotations), total=len(file_paths_videos)):

        # base name
        file_name = to_snake(video_path)   
        session_name = file_name.split('_')[:3]

        # skip the second video, as the second video from s6 (gaze) is simulation footage
        if session_name in session_names: continue 
        else: session_names.append(session_name)

        # get video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # get annotatsions
        records_blinks, records_yawning = get_annotations(ann_path)

        # make clips for current video
        make_clips_and_annotations(video, records_blinks, records_yawning, clip_len_s=10)     
