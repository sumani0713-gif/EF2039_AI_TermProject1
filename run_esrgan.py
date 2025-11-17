import cv2
import torch
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

# 1. 모델 네트워크 설정
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

# 2. 장치 선택
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. RealESRGANer 설정
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

# 4. 이미지 로드
img = cv2.imread('low.jpg')

# 5. 업스케일 실행
output, _ = upsampler.enhance(img, outscale=4)

# 6. 저장
cv2.imwrite('output.png', output)
print('완료!')
