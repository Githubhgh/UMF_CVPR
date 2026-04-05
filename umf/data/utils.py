import torch


def lengths_to_mask(lengths, max_len=None):
    #max_len = max(lengths)
    max_len = 300 if max_len is None else max_len
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def all_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["motion"] for b in notnone_batches]
    # labelbatch = [b['target'] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    # labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = (lengths_to_mask(
        lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)
                       )  # unqueeze for broadcasting

    motion = databatchTensor
    cond = {"y": {"mask": maskbatchTensor, "lengths": lenbatchTensor}}

    if "text" in notnone_batches[0]:
        textbatch = [b["text"] for b in notnone_batches]
        cond["y"].update({"text": textbatch})

    # collate action textual names
    if "action_text" in notnone_batches[0]:
        action_text = [b["action_text"] for b in notnone_batches]
        cond["y"].update({"action_text": action_text})

    return motion, cond


# an adapter to our collate func
def mld_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "length": [b[5] for b in notnone_batches],
        "word_embs":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "pos_ohot":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "text_len":
        collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
        "tokens": [b[6] for b in notnone_batches],
    }
    return adapted_batch


import os
import numpy as np

def ih_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    #notnone_batches.sort(key=lambda x: x[3], reverse=True)
    # batch.sort(key=lambda x: x[3], reverse=True)

    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "text": [b[1] for b in notnone_batches],
        "length": [b[-4] for b in notnone_batches],
        "type": [b[-3] for b in notnone_batches],
        "native_motion": collate_tensors([torch.tensor(b[-2]).float() for b in notnone_batches]),
        "rest": collate_tensors([torch.tensor(b[-1]).float() for b in notnone_batches]),
        "word_embs":
        collate_tensors(torch.tensor(np.array([b[2] for b in notnone_batches]))),
        "pos_ohot":
        collate_tensors(torch.tensor(np.array([b[3] for b in notnone_batches]))),
        "text_len":
        collate_tensors(torch.tensor(np.array([b[4] for b in notnone_batches]))),
    }

        # adapted_batch = {
        #     "motion":
        #     collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        #     "text": [b[2] for b in notnone_batches],
        #     "length": [b[5] for b in notnone_batches],
        #     "word_embs":
        #     torch.tensor([0 for b in notnone_batches]),
        #     "pos_ohot":
        #     torch.tensor([0 for b in notnone_batches]),
        #     "text_len":
        #     torch.tensor([0 for b in notnone_batches]),
        #     "tokens": torch.tensor([0 for b in notnone_batches]),
        #     "name": [b[7] for b in notnone_batches],
        #     #"pair_motion": 
        #     #collate_tensors([torch.tensor(b[7]).float() for b in notnone_batches]),
        #     #"V": [b[7] for b in notnone_batches],
        #     #"entities": [b[8] for b in notnone_batches],
        #     #"relations": [b[9] for b in notnone_batches],
        # }



    return adapted_batch


# def ih_collate(batch):
#     """Custom collate function for rotation data"""
#     rotations, texts, lengths = zip(*batch)
    
#     # Handle rotations - pad if necessary
#     max_len = max(lengths)
#     rotation_batch = []
#     # for rotation in rotations:
#     #     if len(rotation) < max_len:
#     #         padding = np.zeros((max_len - len(rotation), rotation.shape[1]))
#     #         rotation = np.concatenate([rotation, padding], axis=0)
#     #     rotation_batch.append(rotation)
        
#     rotations = torch.FloatTensor(rotation_batch)
#     lengths = torch.LongTensor(lengths)
    
#     return {
#         'rotations': rotations,  # [batch_size, max_len, 21*6]
#         'texts': texts,          # List[str]
#         'lengths': lengths       # [batch_size]
#     }



def a2m_collate(batch):

    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]
    labeltextbatch = [b[3] for b in batch]

    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch).unsqueeze(1)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    adapted_batch = {
        "motion": databatchTensor.permute(0, 3, 2, 1).flatten(start_dim=2),
        "action": labelbatchTensor,
        "action_text": labeltextbatch,
        "mask": maskbatchTensor,
        "length": lenbatchTensor
    }
    return adapted_batch
