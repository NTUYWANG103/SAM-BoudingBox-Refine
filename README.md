# Environment
My environment uses python 3.9 with cuda11.3
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -r requirements.txt
```

# SAM Model Download
Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

# Inference
```
python run.py --image_dir datasets/images --label_dir datasets/labels --refined_label_dir datasets/labels_refine --checkpoint sam_vit_h_4b8939.pth --model_type vit_h
```

# Visualization
```
streamlit run visualization.py
```



python run.py --image_dir /home/wangyuxi/codes/yolov7/datasets/apex/train/images --label_dir /home/wangyuxi/codes/yolov7/datasets/apex/train/labels --refined_label_dir /home/wangyuxi/codes/yolov7/datasets/apex/train/labels_refine --checkpoint sam_vit_h_4b8939.pth --model_type vit_h