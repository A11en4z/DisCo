import os
import h5py
import json
import random
import pickle
import argparse
from PIL import Image

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class VisualGenomeDataset(Dataset):
    def __init__(self, args, vocab, mode, tokenizer=None, with_clip_embs=True):
        super(VisualGenomeDataset, self).__init__()
        assert mode in ["train", "val", "test"]

        self.mode = mode
        self.image_dir = os.path.join(args.data_dir, 'images')
        self.resolution = args.resolution
        self.h5_path = os.path.join(args.data_dir, 'labels', f'{self.mode}.h5')
        self.h5_file = h5py.File(self.h5_path, 'r')

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.num_objects = len(self.vocab['object_idx_to_name'])
        self.with_clip_embs = with_clip_embs

        self.labels = {}
        self.image_paths = []
        with h5py.File(os.path.join(args.data_dir, 'labels', f'{self.mode}.h5'), 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                elif k == 'object_rotated_boxes':
                    self.labels[k] = torch.FloatTensor(np.asarray(v))
                else:
                    self.labels[k] = torch.IntTensor(np.asarray(v))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.labels['object_names'])

    def __getitem__(self, index):
        image_path = self.image_paths[index].decode("utf-8")
        image_path = os.path.join(self.image_dir, image_path)

        raw_path = self.h5_file['image_paths'][index].decode("utf-8")
        fname = os.path.basename(raw_path)
        image_path = os.path.join(self.image_dir, fname)

        image = Image.open(image_path).convert("RGB")
        WW, HH = image.size
        image = self.image_transforms(image)

        clip_obj_embs, clip_rel_embs = None, None
        if self.with_clip_embs:
            try:
                clip_dir = os.path.join(os.path.dirname(self.image_dir), 'clip')
                clip_path = os.path.join(clip_dir, os.path.splitext(fname)[0] + '.pkl')
                clip_embs = pickle.load(open(clip_path, 'rb'))
                clip_obj_embs = torch.from_numpy(clip_embs['objects'])
                clip_rel_embs = torch.from_numpy(clip_embs['relations'])
            except FileNotFoundError:
                print(f"[Warning] Missing clip_embs: {clip_path}")
                return None

        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.labels['objects_per_image'][index].item()))
        for rel_idx in range(self.labels['relationships_per_image'][index].item()):
            s = self.labels['relationship_subjects'][index, rel_idx].item()
            o = self.labels['relationship_objects'][index, rel_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        obj_idxs += obj_idxs_without_rels

        objs = []
        boxes = []
        words = []
        obj_idx_mapping = {}
        counter = 0
        for i, obj_idx in enumerate(obj_idxs):
            curr_obj = self.labels['object_names'][index, obj_idx].item()
            cx, cy, w, h, angle = self.labels['object_rotated_boxes'][index, obj_idx].tolist()
            # 始终按弧度处理，不做度制转换
            if w < h:
                w, h = h, w
                angle = angle + np.pi / 2.0
            angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
            cx /= WW
            cy /= HH
            w /= WW
            h /= HH
            curr_box = torch.FloatTensor([cx, cy, w, h, angle])
            words.append(self.vocab['object_idx_to_name'][curr_obj])

            objs.append(curr_obj)
            boxes.append(curr_box)
            obj_idx_mapping[obj_idx] = counter
            counter += 1

        objs.append(self.vocab['object_name_to_idx']['__image__'])
        boxes.append(torch.FloatTensor([0.5, 0.5, 1.0, 1.0, 0.0]))

        boxes = torch.stack(boxes)
        objs = torch.LongTensor(objs)
        num_objs = counter + 1

        triples = []
        for rel_idx in range(self.labels['relationships_per_image'][index].item()):
            s = self.labels['relationship_subjects'][index, rel_idx].item()
            p = self.labels['relationship_predicates'][index, rel_idx].item()
            o = self.labels['relationship_objects'][index, rel_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        caption = ''
        for word in words:
            text = word + '; '
            caption += text
        caption = caption[:-2]

        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(len(objs) - 1):
            triples.append([i, in_image, num_objs - 1])
        triples = torch.LongTensor(triples)

        caption = self.tokenizer(
            caption,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return image, objs, clip_obj_embs, boxes, triples, clip_rel_embs, image_path, caption


def collate_fn_graph_batch(batch):
    all_imgs, all_objs, all_clip_obj_embs, all_boxes, all_triples, all_clip_rel_embs, all_img_paths, all_captions = [], [], [], [], [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []

    batch = [b for b in batch if b is not None]

    obj_offset = 0
    for i, (img, objs, clip_obj_emb, boxes, triples, clip_rel_embs, img_path, caption) in enumerate(batch):
        num_objs, num_triples = objs.size(0), triples.size(0)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        obj_offset += num_objs

        all_imgs.append(img[None])
        all_objs.append(objs)
        all_clip_obj_embs.append(clip_obj_emb)
        all_boxes.append(boxes)
        all_triples.append(triples)
        all_clip_rel_embs.append(clip_rel_embs)
        all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
        all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))
        all_img_paths.append(img_path)
        all_captions.append(caption)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_clip_obj_embs = torch.cat(all_clip_obj_embs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_clip_rel_embs = torch.cat(all_clip_rel_embs)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_captions = torch.cat(all_captions)

    return all_imgs, all_objs, all_clip_obj_embs, all_boxes, all_triples, all_clip_rel_embs, all_obj_to_img, all_triple_to_img, all_img_paths, all_captions


def build_train_dataloader(args, tokenizer=None, with_clip_embs=True):
    with open(os.path.join(args.data_dir, 'vocab.json'), 'r') as f:
        vocab = json.load(f)

    train_dataset = VisualGenomeDataset(args, vocab=vocab, mode='train', tokenizer=tokenizer, with_clip_embs=with_clip_embs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=args.dataloader_shuffle,
        collate_fn=collate_fn_graph_batch
    )

    val_dataset = VisualGenomeDataset(args, vocab=vocab, mode='val', tokenizer=tokenizer, with_clip_embs=with_clip_embs)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        collate_fn=collate_fn_graph_batch
    )

    test_dataset = VisualGenomeDataset(args, vocab=vocab, mode='test', tokenizer=tokenizer, with_clip_embs=with_clip_embs)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        collate_fn=collate_fn_graph_batch
    )

    return train_dataloader, val_dataloader, test_dataloader, vocab