datapath=mvtec
loadpath=/models

modelfolder=VQVAE_PC_IM320_S22
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu -1 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
