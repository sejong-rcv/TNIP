import numpy as np
import pandas as pd
import tqdm
import json
import os
import math
import pickle


def retrieval_mapatn_metric(csv_path, topks, ann, normalize_factor='r', positive_labels='ESLMV', clean=False, all_videos=True):
    
    
    csv = pd.read_csv(csv_path)
    csv_columns = csv.columns
    csv_array = np.asarray(csv)

    queries = np.unique(csv_array[:,1])
    if clean:
        gt = ann['cleaned_database']
    else:
        gt = ann['database']
    result = {}
    for curr_topk in topks:
        
        APx = {}

        i_APx = []
        if curr_topk == "total":
            descriptor = "  Evaluate P@total !"
        else:
            descriptor = "  Evaluate P@{:5d} !".format(curr_topk)
        for qname in tqdm.tqdm(queries, desc=descriptor):
            qid = qname.split("/")[-2]
    
            curr_gt = gt[int(qid)]

            rel_num = 0
            for k, v in curr_gt.items():
                if v==-1:
                    continue
                if v in positive_labels:
                    rel_num+=1

            if all_videos is False:
                slash = csv_path.split("/")
                underbar = slash[-1].split("_")
                underbar.insert(3, qid)
                slash[-1] = "_".join(underbar)
                csv_path_new = "/".join(slash)
                csv_path_new = csv_path_new.replace('R2_1D_base','')
                csv = pd.read_csv(csv_path_new)
                csv_columns = csv.columns
                csv_array = np.asarray(csv)

            
            csv_array_name = np.array(["/".join(i.split("/")[-2:]) for i in csv_array[:,1]])

            candidates_ind = np.where(csv_array_name=="/".join(qname.split("/")[-2:]))[0]
            candidates_data = csv_array[candidates_ind, :]
            sorted_ind = np.argsort(-candidates_data[:,3])
            sorted_data = candidates_data[sorted_ind, :]

            i_bool = []

            for idata in sorted_data[:,2]:
                iid = idata.split("/")[-2]
                if qid!=iid:
                    i_bool.append(False)
                    continue
                iname = idata.split("/")[-1]
                ilabel = curr_gt[iname]

                if ilabel==-1:
                    i_bool.append(False)
                else:
                    if ilabel in positive_labels:
                        i_bool.append(True)
                    else:
                        i_bool.append(False)

            i_bool = np.asarray(i_bool)


            if curr_topk == "total":
                n = sorted_data.shape[0]
            else:
                if sorted_data.shape[0] < curr_topk:
                    n = sorted_data.shape[0]
                else:
                    n = curr_topk

            topk_data = sorted_data[:n]
 

            imatch = i_bool[:n]
            if normalize_factor == 'r':
                nf = np.sum(imatch)
            elif normalize_factor == 'n':
                nf = n
            elif normalize_factor == 'min(n,R)':
                nf = min(n, rel_num)
            else:
                import pdb; pdb.set_trace()

            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, n + 1, 1)
            if nf != 0:
                i_APx.append(np.sum(Px * imatch) / nf)
            if nf == 0:  # even no relevant image, still need add in APx for calculating the mean
                i_APx.append(0)

        i_APx = np.mean(np.array(i_APx))


        APx.update({"mAP" : [i_APx]})
        result.update({curr_topk : APx})


    topn = list(result.keys())
    tasks = list(result[topn[0]].keys())

    save_content = []

    first_line = "| task |"
    for n in topn:
        if n =="total":
            first_line+=" mAP@tot |"
        else:
            first_line+=" mAP@{:3d} |".format(n)
    first_line+="\n"
    spliter = "-"*len(first_line) + "\n"
    if clean:
        tag = "mAP@N normalized by {} Cleaned Evaluator".format(normalize_factor)
    else:
        tag = "mAP@N normalized by {} Evaluator".format(normalize_factor)


    lm = int((len(first_line)-len(tag))/2)
    rm = len(first_line)-len(tag) - lm
    tag = "\n" + "-"*lm + tag + "-"*rm + "\n"
    save_content.append(tag)
    save_content.append(first_line)
    save_content.append(spliter)

    for ts in tasks:
        line = "| {:4s} |".format(ts.lower())
        for n in topn:
            line+="  {:.4f} |".format(result[n][ts][0])
        line+="\n"
        save_content.append(line)
        save_content.append(spliter)

    for s in save_content:
        print(s[:-1])
    
    return save_content

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)

if __name__=="__main__":

    source = ["cc_web_video_retrieve_queries.256K.csv"]

    ann_path = "cc_web_video_annotation.pickle"

    f = open(ann_path, "rb")
    ann = pickle.load(f)
        
    topks = ["total"] # top k
    
    save_root = "./results_cdva"
    mkdir(save_root)

    # normalize_factor : 'r' - the num of rel in retrieved list, 'n' - the num of retrieved list, 'min(n,R)' - the min value between the num of (n) retrieved list and of (R) rel in all dataset.
    normalize_factors = ['r']
    
    for csv_path in source:

        print(csv_path)
        for nf in normalize_factors:
            print("Normalize Factors : {}".format(nf))
            result_mapatn_all = retrieval_mapatn_metric(csv_path, topks, ann, normalize_factor=nf, clean=False, all_videos=True)
            result_mapatn_all_cleaned = retrieval_mapatn_metric(csv_path, topks, ann, normalize_factor=nf, clean=True, all_videos=True)
            result_mapatn = retrieval_mapatn_metric(csv_path, topks, ann, normalize_factor=nf, clean=False, all_videos=False)
            result_mapatn_cleaned = retrieval_mapatn_metric(csv_path, topks, ann, normalize_factor=nf, clean=True, all_videos=False)
            path_spliter = csv_path.split("/")

            pin_dir = [i for i, spt in enumerate(path_spliter) if spt == "arxiv_cdva"]
            start_ind = pin_dir[0]
            path_spliter[start_ind] = "results_cdva"
            path_spliter[start_ind+3] = "result"
            for ci in range(start_ind+1, len(path_spliter)-1):
                curr_path = "/".join(path_spliter[:ci])
                mkdir(curr_path)

            save_txt_path = os.path.join(curr_path, "mAP@N_normalized_by_{}_all.txt".format(nf))
            with open(save_txt_path, 'w') as f:
                f.writelines(result_mapatn_all)

            save_txt_path = os.path.join(curr_path, "mAP@N_normalized_by_{}_all_cleaned.txt".format(nf))
            with open(save_txt_path, 'w') as f:
                f.writelines(result_mapatn_all_cleaned)

            save_txt_path = os.path.join(curr_path, "mAP@N_normalized_by_{}.txt".format(nf))
            with open(save_txt_path, 'w') as f:
                f.writelines(result_mapatn)

            save_txt_path = os.path.join(curr_path, "mAP@N_normalized_by_{}_cleaned.txt".format(nf))
            with open(save_txt_path, 'w') as f:
                f.writelines(result_mapatn_cleaned)
