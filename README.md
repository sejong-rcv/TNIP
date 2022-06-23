## Exploring the temporal cues to enhance video retrieval on standardized CDVA

### Official Pytorch Implementation of [Exploring the temporal cues to enhance video retrieval on standardized CDVA](https://ieeexplore.ieee.org/document/9754362) (IEEE Access)

---

<img src="https://user-images.githubusercontent.com/46413594/154407230-eae08576-8a06-40de-b211-0e10f125b3b3.png" width="700">

> **Exploring the temporal cues to enhance video retrieval on standardized CDVA**<br>
> Jo won (Sejong Univ.), GeunTaek Lim (Sejong Univ.), Yukyung Choi (Sejong Univ.)
>
> **Paper**: [Exploring the temporal cues to enhance video retrieval on standardized CDVA](https://ieeexplore.ieee.org/document/9754362) 
>
> **Abstract:** *As the demand for large-scale video analysis increases, video retrieval research is also becoming more active. In 2014, ISO/IEC MPEG began standardizing compact descriptors for video analysis, known as CDVA, and it is now adopted as a standard. However, the standardized CDVA is not easily compared to other methods because the MPEG-CDVA dataset used for performance verification is not disclosed, despite the fact that follow-up studies are underway with multiple versions of the CDVA experimental model. In addition, analyses of modules constituting the CDVA framework are insufficient in previous studies. Therefore, we conduct self-evaluations of CDVA to analyze the impact of each module on the retrieval task. Furthermore, to overcome the obstacles identified through these self-evaluations, we suggest temporal nested invariance pooling, abbreviated as TNIP, which implies temporal robustness realized by improving nested invariance pooling, abbreviated as NIP, one of the features in CDVA. Finally, benchmarks of the existing CDVA and the proposed approach are provided on several public datasets. Through this, we show that the CDVA framework is capable of boosting the retrieval performance if utilizing the proposed approach.*

## Prerequisites

### Recommended Environment
* Python 3.7
* Pytorch 1.1

### Depencencies
You can set up the environments by using dockerfile 

  `$ make docker-base`.
  
  `$ make docker-run`.

### Data Preparation

#### FIVR 

Fine grained incident Video Retrieval dataset used in our work can be downloaded from the [FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K). The experiment was conducted using FIVR-5K disclosed by the author of the [ViSiL](https://github.com/MKLab-ITI/visil). The data should be located like the structure below.
~~~~
├── dataset
   └── FIVR
       ├── video
         ├── video_1
         ├── video_2
         └── ...
       └── missing_video
         ├── missing_video_1
         ├── missing_video_1
         └── ...
~~~~

#### CC_WEB_VIVDEO

Near duplicate video retrieval dataset used in our work can be downloaded from the [CC_WEB](http://vireo.cs.cityu.edu.hk/webvideo/Download.htm). The data should be located like the structure below.
~~~~
├── dataset
   └── CC_WEB
       ├── 1
         ├── 1_1_Y.flv
         ├── 1_2_Y.flv
         └── ...
       └── 2
         ├── 2_1_Y.flv
         ├── 2_2_Y.flv
         └── ...
       └── ...
~~~~
## Usage

### Extract TNIP Feature
You can easily extract the TNIP feature.
~~~~
$ bash TNIP_FIVR5K.sh
$ bash TNIP_CC_WEB.sh
~~~~
If you want to try other extract options, please refer to `args.py`.

### Evaluate CDVA Retreival

#### Data Preparing

Annotation files can be download from

1. [FIVR Annotation](https://drive.google.com/file/d/1raMkthLdxhnWbZGC-JnrcIipB3KzigA9/view?usp=sharing)

2. [CC_WEB Annotation](https://drive.google.com/file/d/11DozzzM4IN5f3QwIq8xT9Q8cBWy5QFLe/view?usp=sharing)

To check out experiments, you can evaluate our retrieval csv file.

* [FIVR5K](https://drive.google.com/file/d/1IPd583iXBjfUH30VcFnllo0XAk_68byX/view?usp=sharing)
* [FIVR200K](https://drive.google.com/file/d/1icnqmsGzT5OZR5VDXSWi79aFXaJ7bWrn/view?usp=sharing)
* [CC_WEB](https://drive.google.com/file/d/1CG22XKlNnLSVFfaH-ORxnDT5jbRzFCad/view?usp=sharing)

~~~~
$ python calculate_performance_fivr.py
$ python calculate_performance_cc_web_video.py
~~~~

## Experiments

### FIVR5K

| method | DSVR  | CSVR  | ISVR  |
| --------- | ---- | ---- | ---- |
| TCA<sub>c</sub>      | 0.609 | 0.617 | 0.578 |
| ViSiL<sub>f</sub>       | 0.838 | 0.832 | 0.739 |
| ViSiL<sub>sym</sub>       | 0.830 | 0.823 | 0.731 |
| ViSiL<sub>v</sub>       | **0.880** | **0.869** | **0.777** |
| TCA<sub>f</sub>       | 0.844 | 0.834 | 0.763 |
| TCA<sub>sym</sub>       | 0.763 | 0.766 | 0.711 |
| SCFV+NIP<sub>256</sub>       | 0.813 | 0.781 | 0.673 |
| **SCFV+TNIP<sub>256</sub>**       | **0.880** | **0.862** | **0.744** |

* [Result file](https://drive.google.com/file/d/1IPd583iXBjfUH30VcFnllo0XAk_68byX/view?usp=sharing)


### FIVR200K (Additional Benchmark)

| method | DSVR  | CSVR  | ISVR  |
| --------- | ---- | ---- | ---- |
| TCA<sub>c</sub>      | 0.570 | 0.553 | 0.473 |
| ViSiL<sub>f</sub>      | 0.843 | 0.797 | 0.660 |
| ViSiL<sub>sym</sub>       | 0.833 | 0.792 | 0.654 |
| ViSiL<sub>v</sub>       | 0.892 | **0.841** | 0.702 |
| TCA<sub>f</sub>       | 0.877 | 0.830 | **0.703** |
| TCA<sub>sym</sub>       | 0.728 | 0.698 | 0.592 |
| SCFV+NIP<sub>256</sub>      | 0.819 | 0.764 | 0.622 |
| **SCFV+TNIP<sub>256</sub>**       | **0.896** | **0.833** | **0.674** |

* [Result file](https://drive.google.com/file/d/1icnqmsGzT5OZR5VDXSWi79aFXaJ7bWrn/view?usp=sharing)

### CC_WEB_VIDEO

| method | cc_web  | cc_web*  | cc_web<sub>c</sub>  | cc_web<sub>c</sub>*  |
| --------- | ---- | ---- | ---- | ---- |
| DML      | 0.971 | 0.941 | 0.979 | 0.959 |
| TCA<sub>c</sub>      | 0.973 | 0.949 | 0.983 | 0.965 |
| DP      | 0.975 | 0.958 | 0.990 | 0.982 |
| TN      | 0.978 | 0.965 | 0.991 | 0.987 |
| ViSiL<sub>f</sub>       | 0.984 | 0.969 | 0.993 | 0.987 |
| ViSiL<sub>sym</sub>       | 0.982 | 0.969 | 0.991 | 0.988 |
| ViSiL<sub>v</sub>       | **0.985** | **0.971** | **0.996** | **0.993** |
| TCA<sub>f</sub>      | 0.983 | 0.969 | 0.994 | 0.990 |
| TCA<sub>sym</sub>       | 0.982 | 0.962 | 0.992 | 0.981 |
| SCFV+NIP<sub>256</sub>       | 0.973 | 0.953 | 0.976 | 0.959 |
| **SCFV+TNIP<sub>256</sub>**       | **0.978** | **0.969** | **0.983** | **0.975** |

* [Result file](https://drive.google.com/file/d/1CG22XKlNnLSVFfaH-ORxnDT5jbRzFCad/view?usp=sharing)

## References
We referenced the repos below for the code.

* [ViSiL](https://github.com/MKLab-ITI/visil)
* [FIVR](https://github.com/MKLab-ITI/FIVR-200K)
* [CC_WEB](http://vireo.cs.cityu.edu.hk/webvideo/)

## Contact
If you have any question or comment, please contact using the issue.
