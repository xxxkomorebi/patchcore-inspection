export PYTHONPATH=src
datapath=MPDD
#datasets=('carpet')
datasets=('bracket_black' 'bracket_brown' 'bracket_white' 'connector' 'metal_plate' 'tubes')
# datasets=('bottle' 'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
# VQ-VAE-PatchCore: VQ-VAE Training (20 Epochs), Codebook: 512x64, Coreset Percentage: 10%, neighbours: 5, seed: 0
# NOTE: This replaces the original PatchCore training command.
python bin/run_patchcore.py --gpu 0 --seed 123 --save_segmentation_images --log_group all_sample4_e20_topk0.001 --log_project MPDD_sample results \
patch_core \
    --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 32 --vq_embedding_dim 128 --vq_commitment_cost 0.25 \
    --vq_lr 0.001 --vq_epochs 20 \
    --patchsize 3 \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

# vq_num_hiddens 编码器解码器的主干通道数
# --vq_num_embeddings codebook 大小（码本向量数 K）
# vq_embedding_dim 码本向量的维度 D
# --vq_commitment_cost 损失函数中承诺损失的权重 β
# --vq_lr 学习率
# --vq_epochs 训练轮数
# patchsize 3 patch补丁大小

# resize 先把图像缩放到256，再裁剪到imagesize 224


# python bin/run_patchcore.py --gpu 0 --seed 123 --save_segmentation_images --log_group all_sample4_e40 --log_project MPDD_sample results \
# patch_core \
#     --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 512 --vq_embedding_dim 128 --vq_commitment_cost 0.25 \
#     --vq_lr 0.001 --vq_epochs 40 \
#     --patchsize 3 \
# dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

# python bin/run_patchcore.py --gpu 0 --seed 123 --save_segmentation_images --log_group all_sample4_e50 --log_project MPDD_sample results \
# patch_core \
#     --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 512 --vq_embedding_dim 128 --vq_commitment_cost 0.25 \
#     --vq_lr 0.001 --vq_epochs 50 \
#     --patchsize 3 \
# dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

# python bin/run_patchcore.py --gpu 0 --seed 123 --save_segmentation_images --log_group all_sample4_e60 --log_project MPDD_sample results \
# patch_core \
#     --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 512 --vq_embedding_dim 128 --vq_commitment_cost 0.25 \
#     --vq_lr 0.001 --vq_epochs 60 \
#     --patchsize 3 \
# dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath

# python bin/run_patchcore.py --gpu 0 --seed 123 --save_segmentation_images --log_group all_sample4_e70 --log_project MPDD_sample results \
# patch_core \
#     --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 512 --vq_embedding_dim 128 --vq_commitment_cost 0.25 \
#     --vq_lr 0.001 --vq_epochs 70 \
#     --patchsize 3 \
# dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mpdd $datapath


#--vq_use_ema_codebook --vq_ema_decay 0.99 --vq_ema_eps 1e-5
# Original Ensemble command removed for VQ-VAE integration.
# If ensemble functionality is desired, a VQ-VAE ensemble would need to be implemented.
# We keep the original command commented out for reference.
# python bin/run_patchcore.py --gpu 0 --seed 3 --save_patchcore_model --log_group IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S3 --log_online --log_project MVTecAD_Results results \
# patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath


### IM320:
# VQ-VAE-PatchCore: VQ-VAE Training (20 Epochs), Codebook: 512x64, Coreset Percentage: 1%, neighbours: 1, seed: 22
# python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --log_group VQVAE_PC_IM320_S22 --log_project MVTecAD_VQVAE_CPU_Results results \
# patch_core \
#     --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 512 --vq_embedding_dim 64 --vq_commitment_cost 0.25 \
#     --vq_lr 0.0001 --vq_epochs 20 \
#     --anomaly_scorer_num_nn 1 --patchsize 3 \
# sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath

# Original Ensemble command removed for VQ-VAE integration.
# python bin/run_patchcore.py --gpu 0 --seed 40 --save_patchcore_model --log_group IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S40 --log_online --log_project MVTecAD_Results results \
# patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath


############# Segmentation
### IM320
# VQ-VAE-PatchCore Segmentation: VQ-VAE Training (20 Epochs), Codebook: 512x64, Coreset Percentage: 1%, neighbours: 3, seed: 39
# python bin/run_patchcore.py --gpu 0 --seed 39 --save_patchcore_model --log_group VQVAE_PC_IM320_SEG_S39 --log_project MVTecAD_VQVAE_Results results \
# patch_core \
#     --vq_in_channels 3 --vq_out_channels 3 --vq_num_hiddens 128 --vq_num_embeddings 512 --vq_embedding_dim 64 --vq_commitment_cost 0.25 \
#     --vq_lr 0.0001 --vq_epochs 20 \
#     --anomaly_scorer_num_nn 3 --patchsize 5 \
# sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath

# Original Ensemble command removed for VQ-VAE integration.
# python bin/run_patchcore.py --gpu 0 --seed 88 --save_patchcore_model --log_group IM320_Ensemble_L2-3_P001_D1024-384_PS-5_AN-5_S88 --log_online --log_project MVTecAD_Results results \
# patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 5 --patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath