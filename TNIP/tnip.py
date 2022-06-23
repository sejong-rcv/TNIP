
import os
import time
import tqdm
import pandas as pd
import numpy as np
import torchvision
import torch
import cv2
import copy
import pickle
import tarfile
import json
import utils as T

from torch.utils.data import Dataset
from torchvision.io import read_video
from multiprocessing import Process

from args import get_args
from func import *

import lib.multiprocessing as mpu

class UntrimmedVideoDataset(Dataset):
    def __init__(self, args,
                       data_path, 
                       keyframe_path, 
                       clip_length, 
                       seed=42, 
                       transforms=None, 
                       bitrate=['16','64','256','DB'],
                       dist_rank=1,
                       dist_world=1,
                       extract="keyframe"):

        super(UntrimmedVideoDataset,self).__init__()
        self.data_path = data_path
        self.keyframe_path = keyframe_path
        self.clip_length = clip_length
        self.random_seed = seed
        self.transform = transforms
        file_list_buffer = list()
        self.extract = extract
        self.file_list = []
        self.args = args

        if args.resume is None:
            for folder in ['index','queries']:
                for bit in bitrate:
                    txt_list = os.listdir(os.path.join(keyframe_path,args.dataset,'keyframes',folder,bit)) 
                    if len(txt_list) != 0:
                        for txt in txt_list:
                            file_list_buffer.append({str(bit):os.path.join(keyframe_path,args.dataset,'keyframes',folder,bit,txt)})
                    else : continue
        else : 
            file_list_buffer = self.check_omitted_feature_files()

        if dist_world!=1:
            share = len(file_list_buffer) // dist_world
            start = dist_rank*share if dist_rank>0 else 0
            end = (dist_rank+1)*share if (dist_rank+1)!=dist_world else -1
            file_list_buffer = file_list_buffer[start:end] if end!=-1 else file_list_buffer[start:]
        else:
            start = 0
            end = len(file_list_buffer)       

        self.dist_world = dist_world
        self.dist_rank = dist_rank

        meta_format = "TNIP_{}_{}_{}_interval_{}_{}of{}.pickle".format(args.dataset, args.backbone, args.ext_type,args.clip_interval, self.dist_rank, self.dist_world).lower()
        meta_path = os.path.join(args.meta_root, meta_format)

        print("Collecting Meta Info!! -> {}".format(meta_format))
        total_num = len(file_list_buffer)
        if os.path.isfile(meta_path) is False:
            self.get_vid_meta(file_list_buffer)
            f = open(meta_path, "wb")
            pickle.dump(self.file_list, f)
        else:
            f = open(meta_path, "rb")
            self.file_list = pickle.load(f)

        print("DataSet Init Finish!! [index {:>2d} of {:<2d}] #{:<6d} -> #{:<6d} (start:{:6d}, end:{:6d}, drop:{:6d})".format(
                dist_rank, 
                dist_world, 
                len(file_list_buffer),
                len(self.file_list),
                start, 
                end-1 if end!=-1 else total_num,
                total_num-len(self.file_list))
                )


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,index): 
        transformed_clips = list()

        curr_meta = self.file_list[index]
        clips = self.get_clips(curr_meta) # Numpy array

        for i in range(clips.shape[0]):
            transformed_clips.append(self.transform(clips[i]))

        transformed_clips = torch.stack(transformed_clips)

        return transformed_clips, index

    def check_omitted_feature_files(self):

        key_frame_status = dict()
        file_list_buffer = list()
        for folder in ['index','queries']:
            for bit in ['16','64','256','DB']:
                txt_list = os.listdir(os.path.join(self.args.keyframe_path,self.args.dataset,'keyframes',folder,bit)) 
                if len(txt_list) != 0:
                    for txt in txt_list:
                        file_list_buffer.append({str(bit):os.path.join(self.args.keyframe_path,self.args.dataset,'keyframes',folder,bit,txt)})
                else : continue
        
        for file in tqdm.tqdm(file_list_buffer):
            # Explore the total number of keyframes and videos without keyframes.
            path = list(file.values())
            keyframes = self.get_keyframes(path[0])
            if len(keyframes) == 0:
                continue # Videos with zero keyframe are excluded from statistical calculations.
            else :
                key = path[0].split('/')[-2] + ':' + path[0].split('/')[-1].split('.')[0]
                key_frame_status[key] = len(keyframes)

        keys = list(key_frame_status.keys())
        extract_video_list = list()
        for bitrate in os.listdir(os.path.join(self.args.save_path,'extract')):
            for video_folder in os.listdir(os.path.join(self.args.save_path,'extract',bitrate)):
                extract_video_list.append(bitrate+':'+video_folder)

        return_list = list()
        ommited_list = list(set(keys)-set(extract_video_list))
        for ommited_file in ommited_list:
            bitrate = ommited_file.split(':')[0]
            if bitrate in ['16','64','256']:
                type = 'queries'
            else: type = 'index'
            name = ommited_file.split(':')[1]
            item = dict()
            path = 'keyframes/'+self.args.dataset+'/keyframes/'+type+'/'+bitrate+'/'+name+'.mp4.txt'
            if os.path.isfile(path):
                item[bitrate]=path
                return_list.append(item)
            else : import pdb;pdb.set_trace()
            
        return return_list

    def get_vid_meta(self, file_list_buffer):
        vid_meta_buffer = dict()
        for index in tqdm.tqdm(range(len(file_list_buffer)), desc="Meta Inform Construction"):
            file = list(file_list_buffer[index].values())
            keys = list(file_list_buffer[index].keys())[0]
            video_file = file[0]
            video_name = video_file.replace('.txt','').split('/')[-1]

            # path to raw video data file
            if self.args.dataset.upper() == 'FIVR5K' or 'FIVR200K':
                if os.path.isfile(os.path.join(self.data_path,'video',video_name)):
                    video_path = os.path.join(self.data_path,'video',video_name)
                elif os.path.isfile(os.path.join(self.data_path,'missing_video',video_name)):
                    video_path = os.path.join(self.data_path,'missing_video',video_name)
                else : import pdb;pdb.set_trace() ## Video File Not FOUND!!

            # path to raw video data file
            elif self.args.dataset.upper() == 'CC_WEB_VIDEO':
                category = video_name.split('_')[0]
                if os.path.isfile(os.path.join(self.data_path,category,video_name)):
                    video_path = os.path.join(self.data_path,category,video_name)
                else: import pdb;pdb.set_trace()

            cap = cv2.VideoCapture(video_path)
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
            if (cap.isOpened() is False) or (total_frame < 1):
                if self.args.dataset.upper() != 'CC_WEB_VIDEO':
                    print("[Except] Video is not loaded! (video: {})".format(video_name))
                    continue
                elif self.args.dataset.upper() == 'CC_WEB_VIDEO':
                    f = open(os.path.join('cc_web_video',video_file.split('/')[-1]),'r')
                    total_frame = f.read()
                    total_frame = int(total_frame)
  
            shots = self.get_shots(video_file)
            keynums = []
            keyframes = []
            for shot in shots:
                keynums.append(len(shot))
                keyframes.extend(shot)
            keynums = np.asarray(keynums)
            keyframes = np.asarray(keyframes)
            
            if len(keyframes) == 0:
                print("[Except] Keyframe is not extracted! (video: {})".format(video_name))
                continue
                
            if self.extract == "keyframe":
                snippets = self.keyframe_to_snippet(keyframes, total_frame, self.clip_length, self.args.clip_interval)
            elif self.extract == "shot":
                snippets = self.shot_to_snippet(shots, total_frame)
            elif self.extract == "video":
                snippets = self.video_to_snippet(total_frame)

            if snippets.ndim == 1:
                snippets = np.expand_dims(snippets, axis=0)

            vid_meta_buffer.update({index : {"keyframes" : keyframes, "keynums": keynums, "total_frame" : total_frame, 
                                             "shots":shots, "snippets":snippets, "bitrate":keys, "video_file":video_file}})

        self.file_list = []
        for i, (k, v) in enumerate(vid_meta_buffer.items()):
            self.file_list.append(v)

    def get_clips(self, curr_meta):
        video_file = curr_meta['video_file']
        video_name = video_file.replace('.txt','').split('/')[-1]

        # path to raw video data file
        if self.args.dataset.upper() == 'FIVR5K' or 'FIVR200K':
            if os.path.isfile(os.path.join(self.data_path,'video',video_name)):
                video_path = os.path.join(self.data_path,'video',video_name)
            elif os.path.isfile(os.path.join(self.data_path,'missing_video',video_name)):
                video_path = os.path.join(self.data_path,'missing_video',video_name)
            else : import pdb;pdb.set_trace() 

        # path to raw video data file
        elif self.args.dataset.upper() == 'CC_WEB_VIDEO':
            category = video_name.split('_')[0]
            if os.path.isfile(os.path.join(self.data_path,category,video_name)):
                video_path = os.path.join(self.data_path,category,video_name)
            else : import pdb;pdb.set_trace() 

        keyframes = curr_meta['keyframes']
        snippets = curr_meta['snippets']
        total_frame = curr_meta['total_frame']
        clip_nmpy = self.get_frames(video_path, snippets, total_frame)

        return clip_nmpy

    def get_frames(self, video_path, snippets,total_frame):

        frames = dict()

        cap = cv2.VideoCapture(video_path)

        # Robust frame loader
        robust_flag = False
        idx_buffer = []
        # "frame_buffer" is used only, when "idx_buffer" is not empty but the video load iteration is finished.
        frame_buffer = None
        for idx in range(total_frame):
            if idx in snippets:
                ret, frame = cap.read()
                if ret:
                    frames[idx] = frame
                    if robust_flag:
                        for idx_b in idx_buffer:
                            frames[idx_b] = frame
                        robust_flag = False
                        idx_buffer = []
                    if frame_buffer is None:
                        frame_buffer = frame
                else:
                    # If there's no frame to find, turn on.
                    robust_flag = True
                    idx_buffer.append(idx)
            else:
                if robust_flag:
                    ret, frame = cap.read()
                    if ret:
                        for idx_b in idx_buffer:
                            frames[idx_b] = frame
                        robust_flag = False
                        idx_buffer = []
                    if frame_buffer is None:
                        frame_buffer = frame
                else:
                    if frame_buffer is None:
                        ret, frame = cap.read()
                        if ret:
                            frame_buffer = frame
                    else:
                        cap.grab()

        if len(idx_buffer) != 0:
            for idx_b in idx_buffer:
                frames[idx_b] = frame_buffer

        try:
            clip_nmpy = []
            for i,snippet in enumerate(snippets):
                sni_nmpy = []
                for idx in snippet:
                    try:
                        sni_nmpy.append(frames[idx])
                    except:
                        import pdb;pdb.set_trace()
                clip_nmpy.append(sni_nmpy)
        except:
            import pdb;pdb.set_trace()
            print("[Except] Frame load error in process({}/{} - index:{})".format(self.dist_rank, self.world_size, i))

        clip_nmpy = np.asarray(clip_nmpy)

        return clip_nmpy

    def get_keyframes(self,keyframe_file):
        f = open(os.path.join(keyframe_file),'r')
        txt = f.readlines()
        keyframes = list()
        for shot in txt:
            kframes = shot.split(',')
            for i in range(len(kframes)-1):
                if i == 0 : keyframes.append(int(kframes[i].split(':')[1]))
                else : keyframes.append(int(kframes[i]))
        return keyframes

    def get_shots(self,keyframe_file):
        f = open(os.path.join(keyframe_file),'r')
        txt = f.readlines()
        shots = []
        for shot in txt:
            keyframes = list()
            kframes = shot.split(',')
            for i in range(len(kframes)-1):
                if i == 0 : keyframes.append(int(kframes[i].split(':')[1]))
                else : keyframes.append(int(kframes[i]))
            shots.append(keyframes)
        return shots

    def video_to_snippet(self, video_length):
        snippet = np.linspace(0, video_length-1, self.clip_length+1).astype(int)
        snippet = snippet[:-1]
        return snippet

    def shot_to_snippet(self, shots, video_length): # Based on the CDVA shot, it creates a 16-long snippet
        snippets = list()
        for shot in shots:
            start = min(shot)
            end = max(shot)
            snippet = np.linspace(start, end, self.clip_length).astype(int)
            snippet = np.clip(snippet, 0, video_length-1)
            snippets.append(snippet)
        
        return np.stack(snippets)
        
    def keyframe_to_snippet(self, keyframes, video_length, clip_length, clip_interval): # Based on the CDVA keyframe, it creates a 16-long snippet

        # The maximum interval of snippet must not exceed the video length.

        original_clip_length=clip_length
        clip_length = clip_length*clip_interval

        snippets = list()

        if video_length>=clip_length:
            for keyframe in keyframes:
                if 0 <= keyframe - ((clip_length/2)-1) and keyframe + clip_length/2 < video_length:
                    snippet = np.arange(int(keyframe-(clip_length/2-1)),int(keyframe+(clip_length/2+1)),clip_interval)
                elif 0 <= keyframe - ((clip_length/2)-1) and keyframe + clip_length/2 >= video_length: # At the end of the video,
                    snippet = np.arange(video_length-(clip_length),video_length,clip_interval)
                elif 0 > keyframe - ((clip_length/2)-1) and keyframe + clip_length/2 < video_length: # At the front of the video,
                    snippet = np.arange(0,clip_length,clip_interval)
 
                for idx in snippet: # Exception handler
                    if idx >= video_length:
                        import pdb;pdb.set_trace()

                if len(snippet)==original_clip_length:
                    snippets.append(snippet)
                else: import pdb;pdb.set_trace()

                
        elif video_length < clip_length or video_length <= 0: # Exception handler
            for keyframe in keyframes:
                snippet = np.linspace(0,video_length-1,original_clip_length,dtype=int)
                for idx in snippet:
                    if idx >= video_length:
                        import pdb;pdb.set_trace()
                if len(snippet)==original_clip_length:
                    snippets.append(snippet)
                else: import pdb;pdb.set_trace()

        return np.stack(snippets)

def custom_collate_fn(batch):
    total_clips = torch.tensor([])
    total_keyinds = torch.tensor([]).long()
    total_index = []
    spliter = []
    for clips, index in batch:
        total_clips = torch.cat((total_clips, clips), dim=0)
        s = total_clips.shape[0]
        spliter.append(s)
        total_index.append(index)

    return total_clips, total_index, spliter

def get_data_loader(args):
    
    transform = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((args.res, args.res))
    ])

    dataset = UntrimmedVideoDataset(args=args,
                                    data_path=args.data_root_path,
                                    keyframe_path=args.keyframe_path,
                                    clip_length=args.clip_length,
                                    transforms=transform,
                                    dist_rank=args.rank,
                                    dist_world=args.world_size,
                                    extract=args.ext_type
                                    )

    if args.num_worker!=0:
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_worker, 
                                                collate_fn = custom_collate_fn,
                                                shuffle=False)
    else:
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_worker,
                                                collate_fn = custom_collate_fn,
                                                shuffle=False)

    return dataloader

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)

def bin_saver(total_feature, save_folder, curr_meta, ext_type):
    if ext_type == "keyframe":
        for bin_ind, _ in enumerate(curr_meta['keyframes']):
            f = open(os.path.join(save_folder,"clip{:07d}.bin".format(bin_ind)), 'wb')
            total_feature[bin_ind].tofile(f)

    elif ext_type == "shot":
        bin_ind = 0
        for i, kind in enumerate(curr_meta["keynums"]): 
            curr_feature = total_feature[i]
            for ki in range(kind):
                f = open(os.path.join(save_folder,"clip{:07d}.bin".format(bin_ind)), 'wb')
                curr_feature.tofile(f)
                bin_ind+=1

    elif ext_type == "video":
        f = open(os.path.join(save_folder,"clip{:07d}.bin".format(0)), 'wb')
        total_feature.squeeze().tofile(f)
    else:
        import pdb; pdb.set_trace()
        
def main_worker(args):

    if args.backbone.upper() == 'R3D':
        model = torchvision.models.video.r3d_18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-2])) # remove classifier

    elif args.backbone.upper() == 'R2_1D':
        model = torchvision.models.video.r2plus1d_18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-2])) # remove classifier


    model = model.cuda()
    model.eval()
    dataloader = get_data_loader(args)
    data_times = []
    transform_times = []
    feature_times = []
    tnip_times = []
    save_times = []

    pbar = tqdm.tqdm(dataloader)
    start = time.time()
    with torch.no_grad():
        for iter_i, (total_clips, total_index,spliter) in enumerate(pbar):
            data_time = time.time() - start
            data_times.append(data_time)

            total_clips = total_clips.cuda()

            spliter.insert(0,0)
            for bi in range(len(total_index)):

                curr_clips = total_clips[spliter[bi]:spliter[bi+1]]
                curr_index = total_index[bi]
                
                total_feature = torch.tensor([])

                start = time.time()
                transform_time = 0
                for flip in args.flip:
                    for rotate in args.rotate:
                        clips_num = curr_clips.shape[0]
                        if args.mini_batch_size > clips_num:
                            trans_start = time.time()
                            X = rotate_flip(curr_clips, flip, rotate)
                            transform_time += time.time()-trans_start
                            feature = model(X)
                            total_feature = torch.cat((total_feature, feature.squeeze(2).unsqueeze(0).detach().cpu()), dim=0)
                        else: # use mini batch
                            share = clips_num // args.mini_batch_size
                            rest = clips_num % args.mini_batch_size
                            mini_spt = np.arange(0, share+1) * args.mini_batch_size
                            if rest != 0:
                                mini_spt = np.append(mini_spt,  mini_spt[-1]+rest)

                            intermid_feature = torch.tensor([])
                            for mi in range(len(mini_spt)-1):
                                s = mini_spt[mi]
                                e = mini_spt[mi+1]

                                trans_start = time.time()
                                X = rotate_flip(curr_clips[s:e], flip, rotate)
                                transform_time += time.time()-trans_start

                                feature = model(X)
                                intermid_feature = torch.cat((intermid_feature, feature.squeeze(2).detach().cpu()), dim=0)
                            total_feature = torch.cat((total_feature, intermid_feature.unsqueeze(0)), dim=0)
                
                feature_time = time.time() - start - transform_time
                transform_times.append(transform_time)
                feature_times.append(feature_time)
                start = time.time()
                spatial_feature = L2_normalization(Nested_Invariance_Pooling(total_feature[:4], args.region_factor))
                
                if args.temporal_flip:
                    temporal_feature = L2_normalization(Nested_Invariance_Pooling(total_feature[4:8], args.region_factor))
                    tnip_feature = torch.cat([spatial_feature,temporal_feature],dim=1)
                    out_feature = L2_normalization(tnip_feature)
                else : 
                    out_feature = L2_normalization(spatial_feature)
                    
                out_feature = out_feature.numpy()

                tnip_time = time.time() - start
                tnip_times.append(tnip_time)

                start = time.time()

                curr_meta = dataloader.dataset.file_list[curr_index]
                curr_bitrate = curr_meta['bitrate']
                curr_vid = curr_meta["video_file"].split('/')[-1].split('.')[0]

                save_folder = os.path.join(args.extract_path, curr_bitrate, curr_vid)
                mkdir(save_folder)
                try:
                    bin_saver(out_feature, save_folder, curr_meta, args.ext_type)
                except:
                    print("{} -> {}".format(args.rank, iter_i))
                save_time = time.time() - start
                save_times.append(save_time)

            damean, trmean, femean, tnmean, samean =  np.mean(data_times), \
                np.mean(transform_times), \
                np.mean(feature_times), \
                np.mean(tnip_times), \
                np.mean(save_times)

            mean_zip = [damean, trmean, femean, tnmean, samean]
            mean_zip = [torch.tensor([i]).cuda() for i in mean_zip]

            if args.num_gpus > 1:
                mean_zip = mpu.all_gather(mean_zip)

            mean_zip = [np.mean(i.cpu().numpy()) for i in mean_zip]

            pbar.set_description("Time: (Data-{:5.2f}), (Trans-{:5.2f}), (Feat-{:5.2f}), (TNIP-{:5.2f}), (Save-{:5.2f})".format(
                mean_zip[0],
                mean_zip[1],
                mean_zip[2],
                mean_zip[3],
                mean_zip[4],
                ))

            start = time.time()

if __name__ == '__main__':
    args = get_args()
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Save data
    tar = tarfile.open( os.path.join(args.source_path, 'sources.tar'), 'w' )
    tar.add( 'docker' )
    tar.add( 'lib' )
    curr_file = os.listdir(os.getcwd())
    curr_file = [tar.add(i) for i in curr_file if os.path.isdir(i) is False]
    tar.close()
    with open(os.path.join(args.source_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    start = time.time()
    init_method="tcp://localhost:{:4d}".format(torch.randint(8000, 9999, (1,)).item())
    mpu.launch_job(func=main_worker, args=args, init_method=init_method)
    end = time.time()

    logout = "[Total Execution Time] {:15.2f}(sec)".format(end-start)
    print("\n\n\n"+"*"*len(logout)+"\n")
    print(logout)
    print("\n"+"*"*len(logout)+"\n\n\n")

