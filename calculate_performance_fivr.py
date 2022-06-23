import numpy as np
import pandas as pd
import tqdm
import json
import os
import math
import pickle


def retrieval_mapatn_metric(csv_path, topks, ann, normalize_factor='r' ):
    set_type = csv_path.split("/")[-1].split("_")[0].split("fivr")[1]
    
    csv = pd.read_csv(csv_path)
    csv_columns = csv.columns
    csv_array = np.asarray(csv)

    queries = np.unique(csv_array[:,1])
    
    result = {}
    for curr_topk in topks:
        
        APx = {}

        d_APx = []
        c_APx = []
        i_APx = []
        if curr_topk == "total":
            descriptor = "  Evaluate P@total !"
        else:
            descriptor = "  Evaluate P@{:5d} !".format(curr_topk)
        for qname in tqdm.tqdm(queries, desc=descriptor):

            candidates_ind = np.where(csv_array[:,1]==qname)[0]
            candidates_data = csv_array[candidates_ind, :]
            q_class = qname.split("/")[-1].split(".mp4")[0]
            q_label = ann[set_type]['queries'][q_class]['label']


            d_rel_num = 0
            for k, v in ann[set_type]['database']['index']['dsvr'].items():
                if q_label[0] in v['label']:
                    d_rel_num+=1
            
            c_rel_num = 0
            for k, v in ann[set_type]['database']['index']['csvr'].items():
                if q_label[0] in v['label']:
                    c_rel_num+=1

            i_rel_num = 0
            for k, v in ann[set_type]['database']['index']['isvr'].items():
                if q_label[0] in v['label']:
                    i_rel_num+=1
            sorted_ind = np.argsort(-candidates_data[:,3])
            sorted_data = candidates_data[sorted_ind, :]

            d_bool = []
            c_bool = []
            i_bool = []
            for iname in sorted_data[:,2]:
                i_class = iname.split("/")[-1].split(".mp4")[0]
                dsvr_label = ann[set_type]['database']['index']['dsvr'][i_class]['label']
                csvr_label = ann[set_type]['database']['index']['csvr'][i_class]['label']
                isvr_label = ann[set_type]['database']['index']['isvr'][i_class]['label']


                q_lnum = len(q_label)
                d_lnum = len(dsvr_label)
                c_lnum = len(csvr_label)
                i_lnum = len(isvr_label)
                d_bool.append(len(np.union1d(q_label, dsvr_label)) != (q_lnum+d_lnum))
                c_bool.append(len(np.union1d(q_label, csvr_label)) != (q_lnum+c_lnum))
                i_bool.append(len(np.union1d(q_label, isvr_label)) != (q_lnum+i_lnum))

            d_bool = np.asarray(d_bool)
            c_bool = np.asarray(c_bool)
            i_bool = np.asarray(i_bool)
                


            if curr_topk == "total":
                n = sorted_data.shape[0]
            else:
                if sorted_data.shape[0] < curr_topk:
                    n = sorted_data.shape[0]
                else:
                    n = curr_topk


            topk_data = sorted_data[:n]
 

            imatch = d_bool[:n]
            if normalize_factor == 'r':
                nf = np.sum(imatch)
            elif normalize_factor == 'n':
                nf = n 
            elif normalize_factor == 'min(n,R)':
                nf = min(n, d_rel_num)
            else:
                import pdb; pdb.set_trace()
            
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, n + 1, 1)
            if nf != 0:
                d_APx.append(np.sum(Px * imatch) / nf)
            if nf == 0:  # even no relevant image, still need add in APx for calculating the mean
                d_APx.append(0)
            

            imatch = c_bool[:n]
            if normalize_factor == 'r':
                nf = np.sum(imatch)
            elif normalize_factor == 'n':
                nf = n
            elif normalize_factor == 'min(n,R)':
                nf = min(n, c_rel_num)
            else:
                import pdb; pdb.set_trace()

            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, n + 1, 1)
            if nf != 0:
                c_APx.append(np.sum(Px * imatch) / nf)
            if nf == 0:  # even no relevant image, still need add in APx for calculating the mean
                c_APx.append(0)

            
            imatch = i_bool[:n]
            if normalize_factor == 'r':
                nf = np.sum(imatch)
            elif normalize_factor == 'n':
                nf = n
            elif normalize_factor == 'min(n,R)':
                nf = min(n, i_rel_num)
            else:
                import pdb; pdb.set_trace()

            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, n + 1, 1)
            if nf != 0:
                i_APx.append(np.sum(Px * imatch) / nf)
            if nf == 0:  # even no relevant image, still need add in APx for calculating the mean
                i_APx.append(0)

        d_APx = np.mean(np.array(d_APx))
        c_APx = np.mean(np.array(c_APx))
        i_APx = np.mean(np.array(i_APx))


        APx.update({"dsvr" : [d_APx]})
        APx.update({"csvr" : [c_APx]})
        APx.update({"isvr" : [i_APx]})
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
 
    source = ["../arxiv_cdva/fivr5k_tnip/scfv+nip/csv/retrieval/q256_iDB/scfv+nip/R2_1D_base/fivr5k_retrieve_queries.256K.csv",
            "../arxiv_cdva/fivr200k_tnip/scfv+nip/csv/retrieval/q256_iDB/scfv+nip/R2_1D_base/fivr200k_retrieve_queries.256K.csv"]

    ann_path = "/data3/datasets/fivr/annotation/fivr_annotation.pickle"

    f = open(ann_path, "rb")
    ann = pickle.load(f)
    
    topks = ["total"] # top k
    # topks = [ "total"] # top k
    
    # normalize_factor : 'r' - the num of rel in retrieved list, 'n' - the num of retrieved list, 'min(n,R)' - the min value between the num of (n) retrieved list and of (R) rel in all dataset.
    normalize_factors = ['r']
    
    for csv_path in source:

        print(csv_path)
        for nf in normalize_factors:
            print("Normalize Factors : {}".format(nf))
            result_mapatn = retrieval_mapatn_metric(csv_path, topks, ann, normalize_factor=nf)
            path_spliter = csv_path.split("/")

            pin_dir = [i for i, spt in enumerate(path_spliter) if spt == "arxiv_cdva"]
            start_ind = pin_dir[0]
            path_spliter[start_ind] = "results_cdva"
            path_spliter[start_ind+3] = "result"

            for ci in range(start_ind+1, len(path_spliter)):
                curr_path = "/".join(path_spliter[:ci])
                mkdir(curr_path)
            save_path = os.path.join(curr_path, "mAP@N_normalized_by_{}.txt".format(nf))

            with open(save_path, 'w') as f:
                f.writelines(result_mapatn)
