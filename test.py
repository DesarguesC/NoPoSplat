import torch, pdb, os, sys
import numpy as np
import cv2
from basicsr.utils import tensor2img, img2tensor
from einops import rearrange, repeat
from transformers import VivitModel
from transformers.models.vivit.modeling_vivit import VivitEmbeddings, VivitPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

class myVivitEmbeddings(VivitEmbeddings):

    def forward(self, pixel_values, interpolate_pos_encoding: bool=False):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        print(f'overwrite worked: embeddings.shape = {embeddings.shape}')
        cls_tokens = self.cls_token.tile([batch_size, 1, 1])
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # pdb.set_trace()
        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings

@add_start_docstrings(
    "The bare ViViT Transformer model outputting raw hidden-states without any specific head on top.",
)
class myVivitModel(VivitModel):
    def __init__(self, config, add_pooling_layer=True, new_embedding=myVivitEmbeddings):
        super().__init__(config, add_pooling_layer)
        # pdb.set_trace()
        if new_embedding is not None:
            self.embeddings = new_embedding(config)

def main1():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = myVivitModel.from_pretrained("../google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", torch_dtype=torch.float32)
    model = model.to(device)
    folder_path = "./datasets/point_odyssey/val/ani10_new_f/rgbs"
    img_names = [os.path.join(folder_path, x) for x in os.listdir(folder_path)]
    img_list = [cv2.resize(cv2.imread(img), (224,224)) for img in img_names]
    batch_size = 1

    pdb.set_trace()
    monocular_tensor = [torch.cat([img2tensor(img_list[i])[None,:,:,:] for i in range(u, u+32)]) for u in range(len(img_list)-31)]
    # tensor_list = [torch.cat([monocular_tensor[i][None,:,:,:,:] for i in range(u, u+batch_size)], dim=0).to(device) for u in range(len(monocular_tensor)+1-batch_size)]

    from random import randint
    u = randint(0, 100)
    video_tensor = torch.cat([monocular_tensor[i][None,:,:,:,:] for i in range(u, u+batch_size)], dim=0).to(device)

    pdb.set_trace()
    out = model(video_tensor, interpolate_pos_encoding=True) # first check video_tensor shape
    # batch_size = 4 -> 23GB

    pdb.set_trace()
    print(out)
    """
    last_hidden_state: torch.FloatTensor = None √
    pooler_output: torch.FloatTensor = None √
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    """



if __name__ == "__main__":
    from src.model.encoder.backbone.backbone_videomamba import main, VideoMamba
    # video_mamba = VideoMamba(
    #     mamba_choice = 'middle',
    #     num_frames = 20,
    #     decoder_weights_path = './pretrained_weights/mixRe10kDl3dv.ckpt'
    # )
    pdb.set_trace()
    main()

