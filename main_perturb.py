import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attentionControl import AttentionControlEdit
import diff_latent_perturb
from PIL import Image
import numpy as np
import os
import glob
from other_attacks import model_transfer
import random
import sys
from natsort import ns, natsorted
import argparse
from custom_loss import BoundedLogitLossFixedRef
from torch import optim
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="output", type=str,
                    help='Where to save the adversarial examples, and other results')
parser.add_argument('--images_root', default=r"demo\images", type=str,
                    help='The clean images root directory')
parser.add_argument('--label_path', default=r"demo\labels.txt", type=str,
                    help='The clean images labels.txt')
parser.add_argument('--is_test', default=False, type=bool,
                    help='Whether to test the robustness of the generated adversarial examples')
parser.add_argument('--pretrained_diffusion_path',
                    default=r"/home/zhaowei/diffusion/local/stable-diffusion2",
                    type=str,
                    help='Change the path to `stabilityai/stable-diffusion-2-base` if want to use the pretrained model')

parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
parser.add_argument('--start_step', default=15, type=int, help='Which DDIM step to start the attack')
parser.add_argument('--iterations', default=30, type=int, help='Iterations of optimizing the adv_image')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--model_name', default="inception", type=str,
                    help='The surrogate model from which the adversarial examples are crafted')
parser.add_argument('--is_apply_mask', default=False, type=bool,
                    help='Whether to leverage pseudo mask for better imperceptibility (See Appendix D)')
parser.add_argument('--is_hard_mask', default=False, type=bool,
                    help='Which type of mask to leverage (See Appendix D)')

parser.add_argument('--guidance', default=2.5, type=float, help='guidance scale of diffusion models')
parser.add_argument('--attack_loss_weight', default=10, type=int, help='attack loss weight factor')
parser.add_argument('--cross_attn_loss_weight', default=10000, type=int, help='cross attention loss weight factor')
parser.add_argument('--self_attn_loss_weight', default=100, type=int, help='self attention loss weight factor')


def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(42)


def run_diffusion_attack(image, label, diffusion_model,uap,custom_loss,optimizer_uap,
                        target_label, diffusion_steps, guidance=2.5,
                          save_dir=r"/home/zhaowei/diffusion/output", res=224,
                         model_name="inception", start_step=15, iterations=30, args=None):

    adv_image, clean_acc, adv_acc , uap = diff_latent_perturb.diffattack(diffusion_model, label, uap,custom_loss,optimizer_uap,target_label,
                                                                  num_inference_steps=diffusion_steps,
                                                                  guidance_scale=guidance,
                                                                  image=image,
                                                                  save_path=save_dir, res=res, model_name=model_name,
                                                                  start_step=start_step,
                                                                  iterations=iterations, args=args)

    return adv_image, clean_acc, adv_acc, uap


if __name__ == "__main__":
    args = parser.parse_args()
    guidance = args.guidance
    diffusion_steps = args.diffusion_steps  # Total DDIM sampling steps.
    start_step = args.start_step  # Which DDIM step to start the attack.
    iterations = args.iterations  # Iterations of optimizing the adv_image.
    res = args.res  # Input image resized resolution.
    model_name = args.model_name  # The surrogate model from which the adversarial examples are crafted.
    save_dir = args.save_dir  # Where to save the adversarial examples, and other results.
    os.makedirs(save_dir, exist_ok=True)

   

    is_test = args.is_test  # Whether to test the robustness of the generated adversarial examples.

    print("\n******Attack based on Diffusion*********")

    # Change the path to "stabilityai/stable-diffusion-2-base" if you want to use the pretrained model.
    pretrained_diffusion_path = args.pretrained_diffusion_path

    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to('cuda:0')
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    base_path = r"/home/zhaowei/adv_test/"
    df = pd.read_csv(base_path + 'dev_dataset.csv')
    path = df['ImageId']
    labels = df['TrueLabel']
    import numpy as np
    import os
    pathList = np.array(path)
    labels = np.array(labels)  - 1
    filterDatapath = []
    filterLabels = []
    for i in range(len(pathList)):
        imagePath = base_path +'images/'+ pathList[i] +'.png'
        #print(imagePath)
        if os.path.exists(imagePath):
            filterDatapath.append(imagePath)
            filterLabels.append(labels[i])
    all_images = np.array(filterDatapath)
    label = np.array(filterLabels)
    print('There are %d images' % len(all_images))
    print('There are %d images' % len(label))
    adv_images = []
    images = []
    clean_all_acc = 0
    adv_all_acc = 0
    if is_test:
        uap = np.load('uap/train_uap')
        uap = torch.tensor(uap,device='cuda')
        uap.requires_grad_(False)
        custom_loss = None
        optimizer_uap = None
        target_label = 483
    else:
        uap = torch.nn.Parameter(torch.zeros(size=(4, 28, 28), requires_grad=True))
        uap = torch.tensor(uap,device='cuda')
        uap.requires_grad_(True)
        custom_loss = BoundedLogitLossFixedRef(use_cuda=True)
        optimizer_uap = optim.AdamW([uap], lr=5e-2)
        target_label = 483
    for ind, image_path in enumerate(all_images):
        tmp_image = Image.open(image_path).convert('RGB')
        tmp_image.save(os.path.join(save_dir, str(ind).rjust(4, '0') + "_originImage.png"))

        adv_image, clean_acc, adv_acc, uap = run_diffusion_attack(tmp_image, label[ind:ind + 1],
                                                             ldm_stable,
                                                             uap,
                                                            custom_loss,
                                                                optimizer_uap,
                                                            target_label,
                                                             diffusion_steps, guidance=guidance,
                                                             res=res, model_name=model_name,
                                                             start_step=start_step,
                                                             iterations=iterations,
                                                             save_dir=os.path.join(save_dir,
                                                                                   str(ind).rjust(4, '0')), args=args)
        adv_image = adv_image.astype(np.float32) / 255.0
        adv_images.append(adv_image[None].transpose(0, 3, 1, 2))

        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
        images.append(tmp_image)

        clean_all_acc += clean_acc
        adv_all_acc += adv_acc

    print("Clean acc: {}%".format(clean_all_acc / len(all_images) * 100))
    print("Adv acc: {}%".format(adv_all_acc / len(all_images) * 100))

    np.save('uap/train_uap',uap.detach().cpu().numpy())
    
    images = np.concatenate(images)
    adv_images = np.concatenate(adv_images)
    model_transfer(images, adv_images, label, res, save_path=save_dir)
