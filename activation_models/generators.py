#load the activations of models from pytorch, vvs, taskonomy, etc.
#get_activation_models used in scripts/compute_eigenspectra, fit_encoding_models, 
# - taskonomy import should work on lab server

#* make sure import from candidate_models and model_tools is set to current task*
# - for classification performance brainscore,candidate_models, and model_tools will not be installed in the environment

import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
from torchvision.models import resnet18, resnet50
#from candidate_models.base_models.unsupervised_vvs import ModelBuilder
from classification_custom.candidate_models import ModelBuilder
#from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from classification_custom.model_tools import PytorchWrapper, load_preprocess_images
from visualpriors.taskonomy_network import TASKONOMY_PRETRAINED_URLS, TaskonomyEncoder
from functools import partial
from utils import properties_to_id
from activation_models.AtlasNet.model_2L_eig import EngineeredModel2L
#from counter_example.train_imagenet import LitResnet

atlasnet_layers = ['c1', 'c2']

resnet18_pt_layers = [f'layer1.{i}.relu' for i in range(2)] + \
                     [f'layer2.{i}.relu' for i in range(2)] + \
                     [f'layer3.{i}.relu' for i in range(2)] + \
                     [f'layer4.{i}.relu' for i in range(2)]

resnet50_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                     [f'layer2.{i}.relu' for i in range(4)] + \
                     [f'layer3.{i}.relu' for i in range(6)] + \
                     [f'layer4.{i}.relu' for i in range(3)]

resnet18_tf_layers = [f'encode_{i}' for i in range(2, 10)]


def get_activation_models(pytorch=False, vvs=False, taskonomy=False, pytorch_hub=False,
                          atlasnet=True):
    if atlasnet:
        for model, layers in atlas_net():
            yield model, layers
    if pytorch:
        for model, layers in pytorch_models():
            yield model, layers
    if vvs:
        for model, layers in vvs_models():
            yield model, layers
    if taskonomy:
        for model, layers in taskonomy_models():
            yield model, layers
    if pytorch_hub:
        for model, layers in pytorch_hub_models():
            yield model, layers

def atlas_net():
    model = EngineeredModel2L(filters_2=1000).Build()
    identifier = properties_to_id('AtlasNet', 'default', 'Untrained', 'Engineered')
    model = wrap_pt(model, identifier)
    yield model, atlasnet_layers

def pytorch_models():
    model = resnet18(weights=None)
    identifier = properties_to_id('ResNet18', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = resnet50(weights=None)
    identifier = properties_to_id('ResNet50', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers

    model = resnet18(weights='DEFAULT')
    identifier = properties_to_id('ResNet18', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = resnet50(weights='DEFAULT')
    identifier = properties_to_id('ResNet50', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers


def pytorch_hub_models():
    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    identifier = properties_to_id('ResNet50', 'Barlow-Twins', 'Self-Supervised', 'Pytorch Hub')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers


def vvs_models():
    configs = [('resnet18-simclr', 'SimCLR', 'Self-Supervised'),
               ('resnet18-supervised', 'Object Classification', 'Supervised'),
               ('resnet18-la', 'Local Aggregation', 'Self-Supervised'),
               ('resnet18-ir', 'Instance Recognition', 'Self-Supervised'),
               ('resnet18-ae', 'Auto-Encoder', 'Self-Supervised'),
               ('resnet18-cpc', 'Contrastive Predictive Coding', 'Self-Supervised'),
               ('resnet18-color', 'Colorization', 'Self-Supervised'),
               ('resnet18-rp', 'Relative Position', 'Self-Supervised'),
               ('resnet18-depth', 'Depth Prediction', 'Supervised'),
               ('resnet18-deepcluster', 'Deep Cluster', 'Self-Supervised'),
               ('resnet18-cmc', 'Contrastive Multiview Coding', 'Self-Supervised')]

    for vvs_identifier, task, kind in configs:
        tf.reset_default_graph()

        model = ModelBuilder()(vvs_identifier)
        identifier = properties_to_id('ResNet18', task, kind, 'VVS')
        model.identifier = identifier

        if vvs_identifier in ModelBuilder.PT_MODELS:
            layers = resnet18_pt_layers
        else:
            layers = resnet18_tf_layers

        yield model, layers


def taskonomy_models():
    configs = [('autoencoding', 'Auto-Encoder', 'Self-Supervised'),
               ('curvature', 'Curvature Estimation', 'Supervised'),
               ('denoising', 'Denoising', 'Self-Supervised'),
               ('edge_texture', 'Edge Detection (2D)', 'Supervised'),
               ('edge_occlusion', 'Edge Detection (3D)', 'Supervised'),
               ('egomotion', 'Egomotion', 'Supervised'),
               ('fixated_pose', 'Fixated Pose Estimation', 'Supervised'),
               ('jigsaw', 'Jigsaw', 'Self-Supervised'),
               ('keypoints2d', 'Keypoint Detection (2D)', 'Supervised'),
               ('keypoints3d', 'Keypoint Detection (3D)', 'Supervised'),
               ('nonfixated_pose', 'Non-Fixated Pose Estimation', 'Supervised'),
               ('point_matching', 'Point Matching', 'Supervised'),
               ('reshading', 'Reshading', 'Supervised'),
               ('depth_zbuffer', 'Depth Estimation (Z-Buffer)', 'Supervised'),
               ('depth_euclidean', 'Depth Estimation', 'Supervised'),
               ('normal', 'Surface Normals Estimation', 'Supervised'),
               ('room_layout', 'Room Layout', 'Supervised'),
               ('segment_unsup25d', 'Unsupervised Segmentation (25D)', 'Self-Supervised'),
               ('segment_unsup2d', 'Unsupervised Segmentation (2D)', 'Self-Supervised'),
               ('segment_semantic', 'Semantic Segmentation', 'Supervised'),
               ('class_object', 'Object Classification', 'Supervised'),
               ('class_scene', 'Scene Classification', 'Supervised'),
               ('inpainting', 'Inpainting', 'Self-Supervised'),
               ('vanishing_point', 'Vanishing Point Estimation', 'Supervised')]

    for taskonomy_identifier, task, kind in configs:
        model = TaskonomyEncoder()
        model.eval()
        pretrained_url = TASKONOMY_PRETRAINED_URLS[taskonomy_identifier + '_encoder']
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_url)
        model.load_state_dict(checkpoint['state_dict'])

        identifier = properties_to_id('ResNet50', task, kind, 'Taskonomy')
        model = wrap_pt(model, identifier, res=256)

        yield model, resnet50_pt_layers


def counterexample_models():
    def most_recent_ckpt(run_name):
        ckpt_path = f'counter_example/saved_runs/{run_name}/lightning_logs'
        latest_version = sorted([int(f.split('_')[1]) for f in os.listdir(ckpt_path)])[-1]
        ckpt_path = f'{ckpt_path}/version_{latest_version}/checkpoints/best.ckpt'
        return ckpt_path

    model = LitResnet.load_from_checkpoint(most_recent_ckpt('imagenet_resnet18')).model
    identifier = properties_to_id('ResNet18', 'Object Classification', 'Supervised', 'Counter-Example')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = LitResnet.load_from_checkpoint(most_recent_ckpt('imagenet_resnet18_scrambled_labels')).model
    identifier = properties_to_id('ResNet18', 'Object Classification', 'Supervised Random', 'Counter-Example')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers


def wrap_pt(model, identifier, res=224, norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
    preprocess = partial(load_preprocess_images, image_size=res,
                         normalize_mean=norm[0], normalize_std=norm[1])
    return PytorchWrapper(model=model,
                          preprocessing=preprocess,
                          identifier=identifier)
