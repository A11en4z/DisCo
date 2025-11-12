import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
from data import build_train_dataloader, parse_args

def main(args):
    model = CLIPTextModel.from_pretrained("/data/tmp/clip-vit-base-patch32").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("/data/tmp/clip-vit-base-patch32")

    args.batch_size = 1
    args.data_shuffle = False
    # 修改了build_train_dataloader函数，构造时输出了所有图像的pkl嵌入，现已改回。
    train_dataloader, val_dataloader, test_dataloader, vocab = build_train_dataloader(args, tokenizer, with_clip_embs=False)
    # pbar = tqdm(train_dataloader, file=sys.stdout)
    for split_name, dataloader in zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]):
        print(f"🔄 Processing {split_name} set...")
        pbar = tqdm(dataloader, file=sys.stdout)

        for step, batch in enumerate(pbar):
            try:
                imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption  = batch

                vocab['object_idx_to_name'][0] = 'image'    # __image__ -> image
                vocab['pred_idx_to_name'][0] = 'in'         # __in_image__ -> in

                text_objs = []
                for obj in objs:
                    text_obj = vocab['object_idx_to_name'][obj]
                    text_objs.append(text_obj)
                tokenized_text_objs = tokenizer(text_objs, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                clip_obj_embs = model(tokenized_text_objs.input_ids.to(args.device)).pooler_output
                clip_obj_embs = clip_obj_embs.detach().cpu().numpy()

                text_rels = []
                for triple in triples:
                    s, p, o = triple
                    text_s = vocab['object_idx_to_name'][objs[s]]
                    text_o = vocab['object_idx_to_name'][objs[o]]
                    text_p = vocab['pred_idx_to_name'][p]
                    text_rel = text_s + ' ' + text_p + ' ' + text_o
                    text_rels.append(text_rel)

                tokenized_text_rels = tokenizer(text_rels, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                clip_rel_embs = model(tokenized_text_rels.input_ids.to(args.device)).pooler_output
                clip_rel_embs = clip_rel_embs.detach().cpu().numpy()
                clip_embs = {}
                clip_embs['objects'] = clip_obj_embs
                clip_embs['relations'] = clip_rel_embs

                save_path = img_paths[0].replace('images', 'clip').replace('.png', '.pkl')
                pickle.dump(clip_embs, open(save_path, 'wb'))
            except Exception as e:
                print(f"[Skip] Failed to process {img_paths[0]}: {e}")

if __name__ == '__main__':
    args = parse_args()
    main(args)

# python /root/autodl-tmp/DisCo/prepare_data/construct_textual_graph.py   --data_dir /root/autodl-tmp/vg   --batch_size 1   --dataloader_shuffle False