# Self Aligning VLMs with a focus on Image Modality


"train_with_cap.csv" and "val_with_cap.csv" are also in the google drive ^^ 

## Installation:
### Requirements
1. Linux machine
1. At least one NVIDIA GPU
1. At least CUDA 10.2

### Install Dependencies
To install the required dependencies, first, clone the repository and navigate to the cloned directory:  
```shell script
git clone self_align_clip  
cd self_align_clip 
```  
Next, install the required dependencies using the following command:  
```shell script
pip install -r requirements.txt # install the python dependencies
cd src
```

## Data Preparations
### Sampling Data
Download the LAION dataset `(images.zip, blip_laion_cc_sbu_558k.json)` from https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main and place it in `self_align_clip/`  
Then run the following command to sample the data:  

```shell script
python dataset_pipeline/sampling.py --data_dir /path/to/self_align_clip/
``` 

### Generating synthetic data
To generate synthetic data, run the following command:  
```shell script
python dataset_pipeline/generate_synthetic_data.py --data_dir /path/to/self_align_clip/
```

### Evaluation data
Prepare vl checklist dataset as described in https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md  
Then move the vl dataset to `TSVLC/vl_datasets/`  
If you followed the instructions correctly, you should have the following folders inside vl_datasets: **'hake', 'swig', 'vg'**. 

## Training

### Run the training script
First, navigate to the src directory:
```shell script
cd src
```
The model will be saved in `TSVLC/Outputs/exp_name/checkpoints`

```shell script
python training/main.py --name exp_name --vl_pos --lora 4 --pretrained openai
```

## Evaluation
### Run the evaluation script
All vl_checklist jsons will be saved in `TSVLC/eval_jsons/clip/exp_name/` and the result will be printed. 
To prepare the vl checklist evaluate results for the experiment **exp_name** run the following command:
```shell script
python training/main.py  --lora 4 --pretrained openai --eval_vl_cklist --eval_only --resume /path/to/checkpoint
```





