from tqdm import tqdm
import json
import time
import sys
import argparse
import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from lavis.models import load_model_and_preprocess



def doArgs(argList):
    parser = argparse.ArgumentParser()

    #parser.add_argument('-v', "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument('--job_num',type=int, help="Input file name", required=True)
    # parser.add_argument('--output', action="store", dest="outputFn", type=str, help="Output file name", required=True)

    return parser.parse_args(argList)

def main():


    args = doArgs(sys.argv[1:])
    job_num = args.job_num
    # print "Starting %s" % (progName)
    startTime = float(time.time())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True,
                                                         device=device)
    # raw_image = Image.open(test_folder+img).convert("RGB")
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    def most_frequent(List):
        counter = 0
        item = List[0]
        for i in List:
            curr_frequency = List.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                item = i
        return item

    def extract_tags(tags):
        out = []
        for item in tags:
            out.append(item['name'])
        return out

    # def extract_category(cate):
    #     out = []
    #     for item in cate:
    #         for dic in item:
    #             out.append(dic['name'])
    #     return out

    def remove_useless_tail(texts):
        out = []
        bad_endings = ['in the dark', 'on a black background', 'in the night sky', 'in the sky','in the dark sky', 'with a black background']

        for text in texts:
            # first check ending
            for bad_ending in bad_endings:
                l = len(bad_ending)
                if bad_ending in text and text[-l:] == bad_ending:
                    text = text[:-l]
            # then check 3d
            if '3d model of' or '3D model of' in text:
                continue
            out.append(text)
        return out

    def filter_text(text):

        bad_endings = ['in the dark', 'on a black background', 'in the night sky', 'in the sky','in the dark sky', 'with a black background' ,'3d model','3D model']
        bad_headings = ['a 3d model of', '3d model of' , 'a 3D model of', '3D model of' , 'a 3d model', 'a 3D model', '3d model']
        for bad_ending in bad_endings:
            l = len(bad_ending)
            if bad_ending in text and text[-l:] == bad_ending:
                text = text[:-l]
            # then check 3d
        for bad_heading in bad_headings:
            l = len(bad_heading)
            if bad_heading in text and text[:l] == bad_heading:
                text = text[l:]
                break

        return text

    def find_best_text(cos_scores, cur_texts, thresh=0.9):

        best_text = ''
        l = len(cur_texts)
        good = cos_scores >= thresh
        idx = torch.argmax(good.sum(dim=1))
        count = torch.max(good.sum(dim=1)).item()

        row = good[idx]
        # print("row", row)
        for i in range(len(row)):
            if row[i]  and len(cur_texts[i]) > len(best_text):
                # print( best_text, cur_texts[i])
                best_text = cur_texts[i]
        return best_text, count

    model_clip = SentenceTransformer('clip-ViT-L-14')

    img_folder = "/yuch_ws/views_release"
    # sub_folder_list = os.listdir(img_folder)
    # sub_folder_list.sort()
    valid_file = 'valid_merged_paths_r1_4.json'
    with open(valid_file) as f:
        sub_folder_list = json.load(f)

    sub_folder_list.sort()

    total_n = len(sub_folder_list)
    print("total_n", total_n)  # 772870

    # job_num = 21
    job_length = 10000

    start_n = job_num* job_length
    end_n = (job_num+1) * job_length
    bz = 10

    print("******** cur job_num is ", job_num, "start is", start_n, "end is", end_n)
    print("first few names", sub_folder_list[start_n:start_n + 5])

    batch_s = start_n
    batch_e = batch_s + bz

    bad_folders = []

    data_split_by_count = {}
    for i in range(14):
        data_split_by_count[i] = []


    for folder_idx in tqdm(range(start_n,end_n)):
        cur_folder = sub_folder_list[folder_idx]
        print(folder_idx,cur_folder)
        iter_time_s = time.time()
        data_dict = {}
        data_dict['caption'] = []
        data_dict['texture'] = []
        data_dict['action'] = []
        data_dict['style'] = []
        data_dict['poly'] = []

        if cur_folder[-4:] != "json":
            for idx in range(12):
                im_path = os.path.join(img_folder + "/" + cur_folder, '%03d.png' % idx)
                if not os.path.isfile(im_path):
                    print("bad target", im_path)
                    bad_folders.append(cur_folder)
                    # save the bad items
                    out_text_name = "logs/Bad_folder_names_job_" + str(job_num) + ".txt"
                    with open(out_text_name, 'w') as f:
                        for line in bad_folders:
                            f.write(line + "\n")
                    break
                # case when path exist:
                raw_image = Image.open(im_path).convert("RGB")
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


                cur_prompt = "Question: This is an object centered image, Can you provide a caption for this object. Ignore the balck or white background. Answer:"
                caption = filter_text(model.generate({"image": image, "prompt": cur_prompt})[0])
                data_dict['caption'].append(caption)

                Q = 'Can you tell me what action is it doing? Please ignore the black background.'
                cur_prompt = 'Question: ' + Q + ' Answer:'
                answer = model.generate({"image": image, "prompt": cur_prompt})
                data_dict['action'].append(answer[0])

                Q = 'Can you tell me the style of this image? '
                cur_prompt = 'Question: ' + Q + ' Answer:'
                answer = model.generate({"image": image, "prompt": cur_prompt})
                data_dict['style'].append(answer[0])

                Q = 'This is a rendering image of a 3D asset, Can you tell me whether it is high poly or low poly? '
                cur_prompt = 'Question: ' + Q + ' Answer:'
                answer = model.generate({"image": image, "prompt": cur_prompt})
                data_dict['poly'].append(answer[0])

                Q = 'Can you tell me whether the object has texture or not? '
                cur_prompt = 'Question: ' + Q + ' Answer:'
                answer = model.generate({"image": image, "prompt": cur_prompt})
                data_dict['texture'].append(answer[0])

            # after for loop
            text_emb = model_clip.encode(data_dict['caption'])  # 12 x D_embd
            cos_scores = util.cos_sim(text_emb, text_emb)  # 12 x 12
            best_text, count = find_best_text(cos_scores, data_dict['caption'], thresh=0.85)
            data_dict['best_text'] = best_text
            data_dict['count'] = count

            data_split_by_count[count].append(cur_folder)

            out_text_name = img_folder + "/" + cur_folder + "/BLIP2_best_text.txt"
            out_objarverse_metadata = img_folder + "/" + cur_folder + "/BLIP2_metadata.json"

            with open(out_text_name, 'w') as f:
                f.write(best_text)

            with open(out_objarverse_metadata, 'w') as f:
                json.dump(data_dict, f)

    data_out = "BLIP2_data_split_" + str(job_num) + ".json"
    with open(data_out, 'w') as f:
        json.dump(data_split_by_count, f)
    return

if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    main()




