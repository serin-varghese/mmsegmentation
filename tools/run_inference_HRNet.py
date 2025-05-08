from mmseg.apis import init_model, inference_model, show_result_pyplot
import glob
import os

config_path = "/home/ross/BA/mmsegmentation/configs/hrnet/fcn_hr48_4xb2-160k_cityscapes-512x1024.py"
checkpoint_path = "/home/ross/BA/mmsegmentation/checkpoints/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth"
img_list = glob.glob(
    "/home/ross/dataset_cs/Cityscapes/leftImg8bit/test/**/*.png", recursive=True
)
# img_list = ['/home/ross/dataset_cs/Cityscapes/leftImg8bit/test/stuttgart_00/stuttgart_00_000000_000001_leftImg8bit.png']


def create_batches(img_list, batch_size):
    new_list = list()
    i = 0
    while i < len(img_list):
        new_list.append(img_list[i : i + batch_size])
        i = i + batch_size

    return new_list


img_list = create_batches(img_list, 4)
model = init_model(config_path, checkpoint_path, device="cuda:0")
for img_batch in img_list:
    results = inference_model(model, img_batch, "HRNet_Pseudo_Labels_SegMasks")
    for i, result in enumerate(results):
        show_result_pyplot(
            model,
            img_batch[i],
            result,
            save_dir="test",
            out_file=f"HRNet_Pseudo_Labels_ImageResults/result_{os.path.basename(img_batch[i])}",
            show=False,
        )
