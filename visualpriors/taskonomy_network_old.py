import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

TASKONOMY_PRETRAINED_WEIGHT_FILES="""autoencoding_decoder-a4a006b5a8b314b9b0ae815c12cf80e4c5f2e6c703abdf65a64a020d3fef7941.pth
autoencoding_encoder-e35146c09253720e97c0a7f8ee4e896ac931f5faa1449df003d81e6089ac6307.pth
class_object_decoder-3cdb6d9ec5a221ca39352e62412c2ab5ae7a00258a962b9b67fe398566ce6c5d.pth
class_object_encoder-4a4e42dad58066039a0d2f9d128bb32e93a7e4aa52edb2d2a07bcdd1a6536c18.pth
class_scene_decoder-517010623d64eb108ca3225fde2a87e72e3e97137b744aa12deeff2fa4f097dc.pth
class_scene_encoder-ad85764467cddafd98211313ceddebb98adf2a6bee2cedfe0b922a37ae65eaf8.pth
colorization_encoder-5ed817acdd28d13e443d98ad15ebe1c3059a3252396a2dff6a2090f6f86616a5.pth
curvature_decoder-b93aed18d7510ad9502755f05c1ef569c00d1fc9c4620333a764ad0d6d131fd3.pth
curvature_encoder-3767cf5d06d9c6bca859631eb5a3c368d66abeb15542171b94188ffbe47d7571.pth
denoising_decoder-5c4e343e885ac13ed0093b4f357680437b8a81f4d36c0b27b6ac831ba5c9fce6.pth
denoising_encoder-b64cab95af4a2c565066a7e8effaf37d6586c3b9389b47fff9376478d849db38.pth
depth_euclidean_decoder-f8d7d0d2bdaf55fac3bdfc8c2812c599bac84985d55503ec92960a4c8b5db7e8.pth
depth_euclidean_encoder-88f18d41313de7dbc88314a7f0feec3023047303d94d73eb8622dc40334ef149.pth
depth_zbuffer_decoder-4833f06833899a8d81b29c6d7eda8adf69b394a91a8c0389b0d58db523097de9.pth
depth_zbuffer_encoder-cc343a8ed622fd7ee3ce54398be8682bbbbfb5d11fa80e8d03a56a5ae4e11b09.pth
edge_occlusion_decoder-1b74d29a2b5afd9eb1a2cf2179289a31e2757909135615d5ba0a9164eb22505f.pth
edge_occlusion_encoder-5ac3f3e918131f61e01fe95e49f462ae2fc56aa463f8d353ca84cd4e248b9c08.pth
edge_texture_decoder-e241e823d6417a0c9b36b7616aad759380dfd3eb83362124e90f9ed5daa92c73.pth
edge_texture_encoder-be2d686a6a4dfebe968d16146a17176eba37e29f736d5cd9a714317c93718810.pth
egomotion_encoder-9aa647c34bf98f9e491e0b37890d77566f6ae35ccf41d9375c674511318d571c.pth
fixated_pose_encoder-78cf321518abc16f9f4782b9e5d4e8f5d6966c373d951928a26f872e55297567.pth
inpainting_decoder-5982904d2a3ce470ce993d89572134dd835dd809f5cfd6290334dc0fe8b1277f.pth
inpainting_encoder-bf96fbaaea9268a820a19a1d13dbf6af31798f8983c6d9203c00fab2d236a142.pth
jigsaw_encoder-0c2b342c9080f8713c178b04aa6c581ed3a0112fecaf78edc4c04e0a90516e39.pth
keypoints2d_decoder-0157a2a18c4e1f861c725d8d4cf3701b02e9444f47e22bdd1262c879dd2d0839.pth
keypoints2d_encoder-6b77695acff4c84091c484a7b128a1e28a7e9c36243eda278598f582cf667fe0.pth
keypoints3d_decoder-724ea6f255cbe4c487a984230242ec7c3557fa8234bde2487d69eacc7b9b75af.pth
keypoints3d_encoder-7e3f1ec97b82ae30030b7ea4fec2dc606b71497d8c0335d05f0be3dc909d000d.pth
nonfixated_pose_encoder-3433a600ca9ff384b9898e55d86a186d572c2ebbe4701489a373933e3cfd5b8b.pth
normal_decoder-8f18bfb30ee733039f05ed4a65b4db6f7cc1f8a4b9adb4806838e2bf88e020ec.pth
normal_encoder-f5e2c7737e4948e3b2a822f584892c342eaabbe66661576ba50db7cdd40561c5.pth
point_matching_encoder-4bd2a6b2909d9998fabaf0278ab568f42f2b692a648e28555ede6c6cda5361f4.pth
reshading_decoder-5bda58f921a3065992ab0034aa0ed787af97f26ac9e5668746dae49c299606cb.pth
reshading_encoder-de456246e171dc8407fb2951539aa60d75925ae0f1dbb43f110b7768398b36a6.pth
room_layout_encoder-1e1662f43b834261464b1825227a04efba59b50cc8883bee9adc3ddafd4796c1.pth
segment_semantic_decoder-d74c6fcf4e0f2bbdce9afe21f9064453a2ac5c7131226527b1d0748f701d04a0.pth
segment_semantic_encoder-bb3007244520fc89cd111e099744a22b1e5c98cd83ed3f116fbff6376d220127.pth
segment_unsup25d_decoder-64c1553cadf76e7efd59138321dc94d186a27eb2bb21e5e6c2624ae825bd4da1.pth
segment_unsup25d_encoder-7d12d2500c18c003ffc23943214f5dfd74932f0e3d03dde2c3a81ebc406e31a0.pth
segment_unsup2d_decoder-a0f3975a22032f116d36e3f3a49f33ddcd6e798cced3ac0962eef5bdccfc397f.pth
segment_unsup2d_encoder-b679053a920e8bcabf0cd454606098ae85341e054080f2be29473971d4265964.pth
vanishing_point_encoder-afd2ae9b71d46a54efc5231b3e38ebc3e35bfab78cb0a78d9b75863a240b19a8.pth""".split()
TASKONOMY_PRETRAINED_WEIGHT_URL_TEMPLATE = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/{filename}'
TASKONOMY_PRETRAINED_URLS = {k.split("-")[0]: TASKONOMY_PRETRAINED_WEIGHT_URL_TEMPLATE.format(filename=k)
                             for k in TASKONOMY_PRETRAINED_WEIGHT_FILES}

class TaskonomyEncoder(nn.Module):

    def __init__(self, normalize_outputs=True, eval_only=True, train_penultimate=False, train=False):
        self.inplanes = 64
        super(TaskonomyEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        block = Bottleneck
        layers = [3,4,6,3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.compress1 = nn.Conv2d(2048, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.compress_bn = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.groupnorm = nn.GroupNorm(8, 8, affine=False)
        self.normalize_outputs = normalize_outputs
        self.eval_only = eval_only
        if self.eval_only:
            self.eval()
        for p in self.parameters():
            p.requires_grad = False

        if train_penultimate:
            for name, param in self.named_parameters():
                if 'compress' in name:  # last layers: compress1.weight, compress_bn.weight, compress_bn.bias
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if train:
            for p in self.parameters():
                p.requires_grad = True


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []

        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers.append(block(self.inplanes, planes, downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))

        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.MaxPool2d( kernel_size=1, stride=stride ),
            )
        layers.append(block(self.inplanes, planes, stride, downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, pad=(3,3,3,3), mode='constant', value=0)
        #  other modes are reflect, replicate, constant

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = F.pad(x, (0,1,0,1), 'constant', 0)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.compress1(x)
        x = self.compress_bn(x)
        x = self.relu1(x)
        if self.normalize_outputs:
            x = self.groupnorm(x)
        return x

    def train(self, val):
        if val and self.eval_only:
            warnings.warn("Ignoring 'train()' in TaskonomyEncoder since 'eval_only' was set during initialization.", RuntimeWarning)
        else:
            return super(TaskonomyEncoder, self).train(val)