export PYTHONPATH=src
datapath=MPDD
# datasets=('bracket_black' 'bracket_brown' 'bracket_white' 'connector' 'metal_plate' 'tubes')
datasets=('bottle' 'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
python bin/run_patchcore.py --gpu 0 --seed 123 --save_segmentation_images --log_group all_e50 --log_project MPDD_GRE results \
patch_core \
    --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 128 --vq_embedding_dim 128 --vq_commitment_cost 0.25 \
    --vq_lr 0.001 --vq_epochs 50 \
    --vq_use_ema_codebook --vq_ema_decay 0.99 --vq_ema_eps 1e-5 \
    --patchsize 3 \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

# Parameter reference:
# --vq_num_hiddens        encoder/decoder backbone channels
# --vq_num_embeddings     codebook size K
# --vq_embedding_dim      codebook vector dimension D
# --vq_commitment_cost    commitment loss weight β
# --vq_lr                 learning rate
# --vq_epochs             training epochs
# --patchsize             patch size for PatchMaker
# --vq_backbone_name      (optional) pretrained backbone, e.g. resnet50
# --vq_layers_to_extract_from  (optional) layers to extract features from
# --resize / --imagesize  resize image then center-crop
