import os
import datetime
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0,1", help="gpu")

    parser.add_argument("--mini_batch_size", type=int, default=150, help="mini_batch_size (per clip)")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--num_worker", type=int, default=0, help="num_worker")
    parser.add_argument("--clip_length", type=int, default=16, help="clip_length")
    parser.add_argument("--clip_interval", type=int, default=2, help="clip_length")

    parser.add_argument("--res", type=int, default=224, help="resolution")

    parser.add_argument('--dataset', default=None, choices=['fivr5k', 'cc_web_video','fivr200k'],help="dataset")
    parser.add_argument("--data_root_path", type=str, default=None, help="path_to_dataset_directory")
    parser.add_argument("--keyframe_path", type=str, default=None, help="path_to_keyframe_directory")

    parser.add_argument("--ext_type", type=str, default="keyframe", help="ext_type: keyframe, shot, video")
    parser.add_argument("--backbone", type=str, default="R3D", help="backbone")
    parser.add_argument("--temporal_flip",action='store_true', help="temporal_flip")
    parser.add_argument("--no_rotate",action='store_true', help="no_rotate")

    parser.add_argument("--region_factor", type=list, default=[(3, 3), (5, 5), (7, 7)], help="backbone")

    parser.add_argument("--meta_root", type=str, default="tnip_meta_info", help="meta_root")
    parser.add_argument("--save_root", type=str, default="tnip_jobs", help="save_root")
    parser.add_argument("--save_path", type=str, default="debug", help="save_path")

    parser.add_argument("--resume", type=str, default=None, help="resume_path")

    args = parser.parse_args()
    make_save_dir(args)


    if args.no_rotate:
        args.rotate = [0]
    else:
        args.rotate = [0,90,180,270]

    if args.temporal_flip:
        args.flip = [0, 1]
    else:
        args.flip = [0]

    return args

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)

def make_save_dir(args):
    
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST).strftime('%Y%m%d-%H%M%S')

    args.save_path = "{}_{}_{}_{}_interval_{}_{}_{}_res{}_{}".format(
        now, 
        args.dataset, 
        args.backbone, 
        args.ext_type,
        args.clip_interval, 
        "flip" if args.temporal_flip else "nonflip", 
        "nonrotate" if args.no_rotate else "rotate", 
        args.res, 
        args.save_path
        ).lower()
        
    args.save_path = os.path.join(args.save_root, args.save_path)
    args.extract_path = os.path.join(args.save_path, "extract")
    args.source_path = os.path.join(args.save_path, "source")

    if args.resume is None :
        mkdir(args.meta_root)
        mkdir(args.save_root)
        mkdir(args.save_path)
        mkdir(args.extract_path)
        mkdir(args.source_path)

        for bitrate in ['16','64','256','DB']:
            mkdir(os.path.join(args.extract_path, bitrate))
    else :
        args.save_path = args.resume
        args.extract_path = os.path.join(args.save_path, "extract")
        args.source_path = os.path.join(args.save_path, "source")

