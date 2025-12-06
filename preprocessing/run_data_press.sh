# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tools.py convert --source=/home/user/414lab_public/zty_code/repa_dataset/ILSVRC/Data/CLS-LOC/train \
    --dest=/home/user/414lab_public/zty_code/repa_dataset/ILSVRC/repa/images --resolution=256x256 --transform=center-crop-dhariwal

# Convert the pixel data to VAE latents
python dataset_tools.py encode --source=/home/user/414lab_public/zty_code/repa_dataset/ILSVRC/repa/images \
    --dest=/home/user/414lab_public/zty_code/repa_dataset/ILSVRC/repa/vae-sd