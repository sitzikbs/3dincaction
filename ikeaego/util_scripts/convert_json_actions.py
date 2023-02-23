import os
import shutil
import json

CONVERSION_DICT = {
       'pick up small coffee table screw': 'pick up small screw',
       'allign small coffee table screw': 'allign small screw',
        'vr interface interaction':'application interface',
        'spin screw':'spin screwdriver',
        'pick up back panel ':'pick up back panel',
        'application interface ':'application interface',
        'pick up drawer side panel':'pick up side panel',
        'pick up screw holder (the strange white thing)':'pick up cam lock',
        'insert screw holder (the strange white thing)':'insert cam lock',
        'lay down screwdriver ':'lay down screwdriver',
        'pick up drawer bottom panel ':'pick up bottom panel',
        'allign drawer bottom panel ':'N/A',
        'allign drawer back panel':'N/A',
        'spin drawer knob screw':'spin drawer knob',
        'pick drawer up side panel':'pick up side panel',
        'pick up stool side ':'pick up stool side',
        'pick up stool beam ':'pick up stool beam',
        'manually insert stool screw ':'manually insert stool screw',
        'pick up bottom stool step ':'pick up bottom stool step',
        'allign top stool step ':'allign top stool step',
        'flip stool ':'flip stool',
        'pick up cam lock screw ':'pick up long drawer screw',
        'pick up cam lock screw':'pick up long drawer screw',
        'allign cam lock screw':'allign long drawer screw',
        'pick up cam lock connecter ':'pick up cam lock',
        'insert cam lock connecter ':'insert cam lock',
        'insert cam lock connecter':'insert cam lock',
        'pick up bottom panel ':'pick up bottom panel',
        'allign cam lock connecter ':'insert cam lock',
        'default':'N/A',
        'turn bottom stool step':'N/A',  # 0 occurances
        'turn side panel':'N/A',  # 0 occurances
        'pick up table top':'N/A',  # 0 occurances
        'pick up coffee tabletop main':'N/A',  # 0 occurances
        'lay down stool side':'N/A',  # 0 occurances
        'lay down leg':'N/A',  # 0 occurances
        'lay down front panel':'N/A',  # 0 occurances
        'flip whole table':'N/A',  # 0 occurances
        'flip coffee tabletop secondary':'N/A',  # 0 occurances
        # 'flip coffee tabletop main':'N/A',  # low occurances
        'allign side panel':'N/A',  # 0 occurances
}


dataset_path = '/home/sitzikbs/Datasets/ikeaego_small_clips/64/'
json_path = os.path.join(dataset_path, 'gt_segments.json')
bkp_path = os.path.join(dataset_path, 'gt_segments_old.json')
shutil.copyfile(json_path, bkp_path) #back up the old annotation file
print('backed up original annotations to' + bkp_path)

with open(json_path) as json_file_obj:
    gt_data = json.load(json_file_obj)

# print('')
for scan in gt_data['database']:
    for i, ann in enumerate(gt_data['database'][scan]['annotation']):
        action = ann['label']
        if action in CONVERSION_DICT.keys():
            gt_data['database'][scan]['annotation'][i]['label'] = CONVERSION_DICT[action]

with open(json_path, 'w') as json_file_obj:
     json.dump(gt_data, json_file_obj)
print('Saved mapped annotations to ' + json_path)