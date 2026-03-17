import contextlib
import csv
import logging
import os
import sys

import click
import numpy as np
import torch
import PIL.Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.utils
import torch.utils.data

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
    "mpdd": ["patchcore.datasets.mpdd", "MPDDDataset"],
    "cardd": ["patchcore.datasets.cardd", "CarDDDataset"],
}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--save_recon", is_flag=True, help="保存重建图，并计算 PSNR/SSIM")
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    save_recon,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, with_torch=True)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            PatchCore_list = methods["get_patchcore"](imagesize,device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                # VQ-VAE architecture does not rely on backbone seeds.
                patchcore.utils.fix_seeds(seed, with_torch=True)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            # 保存 perplexity_history 供后续写入每个类别目录
            perplexity_history = getattr(PatchCore, "perplexity_history", None)

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            recon_collector = []
            input_collector = []
            fetch_recon = save_recon or save_segmentation_images
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                if fetch_recon:
                    scores, segmentations, labels_gt, masks_gt, recons, inputs_raw = PatchCore.predict(
                        dataloaders["testing"], return_recon=True
                    )
                    recon_collector.append(recons)
                    input_collector.append(inputs_raw)
                else:
                    scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                        dataloaders["testing"]
                    )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores + 1e-10)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # 需要原始路径时一次性提取
            if save_segmentation_images or save_recon:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

            if save_segmentation_images:
                # 若需要重建图，可在此同时保存重建和分割可视化
                if len(recon_collector) > 0:
                    recons = recon_collector[0]
                    inputs_raw = input_collector[0]
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)

                    recon_vis_path = os.path.join(
                        run_save_path, "recon_segmentation_images", dataset_name
                    )
                    os.makedirs(recon_vis_path, exist_ok=True)
                    # 仅保存四联图与指标，指标随四联图放在同一目录
                    metrics_path = os.path.join(
                        recon_vis_path, "reconstruction_metrics.csv"
                    )
                    with open(metrics_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["image", "psnr", "ssim"])
                        for idx, (img_path, recon, inp) in enumerate(zip(image_paths, recons, inputs_raw)):
                            recon_np = recon.numpy()
                            inp_np = inp.numpy()
                            recon_denorm = np.clip((recon_np * in_std + in_mean), 0, 1)
                            inp_denorm = np.clip((inp_np * in_std + in_mean), 0, 1)
                            recon_img = np.clip(recon_denorm * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                            inp_img = np.clip(inp_denorm * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

                            psnr_val = peak_signal_noise_ratio(inp_img, recon_img, data_range=255)
                            ssim_val = structural_similarity(inp_img, recon_img, channel_axis=2, data_range=255)

                            savename = "_".join(img_path.split("/")[-4:])
                            writer.writerow([img_path, f"{psnr_val:.3f}", f"{ssim_val:.3f}"])

                            seg_map = segmentations[idx]
                            if mask_paths[idx] is not None:
                                mask_np = dataloaders["testing"].dataset.transform_mask(
                                    PIL.Image.open(mask_paths[idx]).convert("RGB")
                                ).numpy()
                                if mask_np.ndim == 3:
                                    mask_np = mask_np[0]
                                mask_np = np.squeeze(mask_np)
                            else:
                                mask_np = np.zeros_like(seg_map)

                            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                            axes[0].imshow(inp_img)
                            axes[0].set_title("input")
                            axes[0].axis("off")

                            axes[1].imshow(mask_np, cmap="gray")
                            axes[1].set_title("gt mask")
                            axes[1].axis("off")

                            axes[2].imshow(seg_map, cmap="viridis")
                            axes[2].set_title("anomaly map")
                            axes[2].axis("off")

                            axes[3].imshow(recon_img)
                            axes[3].set_title("recon")
                            axes[3].axis("off")

                            plt.tight_layout()
                            fig.savefig(os.path.join(recon_vis_path, savename))
                            plt.close(fig)

                    # 保存 perplexity 到每个类别的 recon 目录
                    if perplexity_history is not None:
                        recon_dir = os.path.join(
                            run_save_path, "recon_segmentation_images", dataset_name
                        )
                        os.makedirs(recon_dir, exist_ok=True)
                        per_csv_path = os.path.join(recon_dir, "perplexity_epoch.csv")
                        with open(per_csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["epoch", "perplexity"])
                            for idx, val in enumerate(perplexity_history, start=1):
                                writer.writerow([idx, val])

            LOGGER.info("Computing evaluation metrics.")

            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--vq_in_channels", type=int, default=3)
@click.option("--vq_out_channels", type=int, default=3)
@click.option("--vq_num_hiddens", type=int, default=128)
@click.option("--vq_num_residual_layers", type=int, default=0)
@click.option("--vq_num_residual_hiddens", type=int, default=0)
@click.option("--vq_num_embeddings", type=int, default=512)
@click.option("--vq_embedding_dim", type=int, default=64)
@click.option("--vq_commitment_cost", type=float, default=0.25)
# EMA codebook parameters
@click.option("--vq_use_ema_codebook", is_flag=True, help="Use EMA-based codebook update instead of direct gradient update.")
@click.option("--vq_ema_decay", type=float, default=0.99, show_default=True)
@click.option("--vq_ema_eps", type=float, default=1e-5, show_default=True)
# New parameters for pre-trained backbone
@click.option("--vq_backbone_name", type=str, default=None, help="Name of the pre-trained backbone to use for the VQ-VAE encoder.")
@click.option("--vq_layers_to_extract_from", type=str, multiple=True, default=None, help="Layers to extract features from for the VQ-VAE encoder.")
# VQ-VAE Training Parameters
@click.option("--vq_lr", type=float, default=1e-4)
@click.option("--vq_epochs", type=int, default=10)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--vq_use_fsq", is_flag=True, help="Enable Finite Scalar Quantization (FSQ) instead of VQ.")
@click.option("--vq_fsq_levels", type=str, default="5,5,5,5,5", help="Comma-separated levels for FSQ (e.g. '3,3,3'). Controls dimension and codebook size.")
def patch_core(
    vq_in_channels,
    vq_out_channels,
    vq_num_hiddens,
    vq_num_residual_layers,
    vq_num_residual_hiddens,
    vq_num_embeddings,
    vq_embedding_dim,
    vq_commitment_cost,
    vq_use_ema_codebook,
    vq_ema_decay,
    vq_ema_eps,
    vq_backbone_name,
    vq_layers_to_extract_from,
    vq_lr,
    vq_epochs,
    patchsize,
    vq_use_fsq,
    vq_fsq_levels,
):
    # We assume a single PatchCore instance using the VQ-VAE architecture.
    # Ensemble logic is removed for VQ-VAE integration simplicity.

    # 解析 levels 字符串为列表
    if vq_fsq_levels:
        fsq_levels_list = [int(x) for x in vq_fsq_levels.split(",")]
    else:
        fsq_levels_list = None
    
    def get_patchcore(input_shape,device):
        patchcore_instance = patchcore.patchcore.PatchCore(device)
        patchcore_instance.load(
            device=device,
            input_shape=input_shape,
            # VQ-VAE Params
            vq_in_channels=vq_in_channels,
            vq_out_channels=vq_out_channels,
            vq_num_hiddens=vq_num_hiddens,
            vq_num_residual_layers=vq_num_residual_layers,
            vq_num_residual_hiddens=vq_num_residual_hiddens,
            vq_num_embeddings=vq_num_embeddings,
            vq_embedding_dim=vq_embedding_dim,
            vq_commitment_cost=vq_commitment_cost,
            vq_use_ema_codebook=vq_use_ema_codebook,
            vq_ema_decay=vq_ema_decay,
            vq_ema_eps=vq_ema_eps,
            vq_use_fsq=vq_use_fsq,          # 传入 patchcore.py
            vq_fsq_levels=fsq_levels_list,  # 传入 patchcore.py
            vq_backbone_name=vq_backbone_name,
            vq_layers_to_extract_from=vq_layers_to_extract_from,
            # PatchCore Params
            patchsize=patchsize,
            vq_lr=vq_lr,
            vq_epochs=vq_epochs,
        )
        return [patchcore_instance] # Return as list for compatibility with ensemble logic

    return ("get_patchcore", get_patchcore)

@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            setattr(train_dataloader, 'name', name)
            if subdataset is not None:
                setattr(train_dataloader, 'name', getattr(train_dataloader, 'name') + "_" + subdataset)

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
