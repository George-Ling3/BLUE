import json
import glob
import argparse
import os

temp = {4:39., #ep1
            10:60,
            22: 48.77,#ep5
            29: 45., #ep9
            66: 70, #ep1
            111: 0., #ep9,5,1
            160:60., #ep1
            174:11.7, #ep1
            183:70., #ep1
    }

def merge_route_json(folder_path):
    txt_file = open(os.path.join(folder_path, 'route_score.txt'), 'w')
    file_paths = glob.glob(f'{folder_path}/*.json')
    file_paths = sorted(file_paths)
    merged_records = []
    driving_score = []
    success_num = 0
    for file_path in file_paths:
        route_id_num = file_path.split('/')[-1][:3]
        if 'merged.json' in file_path: continue
        with open(file_path) as file:
            data = json.load(file)
            records = data['_checkpoint']['records']
            if len(records) == 0:
                print(f'No records in: {file_path}')
                continue
            for rd in records:
                if rd['status'] == 'Failed - Agent crashed':
                    print('Crash!!', rd['route_id'])
                    continue
                rd.pop('index')
                merged_records.append(rd)
                driving_score.append(rd['scores']['score_composed'])
                # txt_file.write(f'{rd["route_id"]}\t{rd["scores"]["score_composed"]}\t{rd["status"]}\n')
                
                if rd['status']=='Completed' or rd['status']=='Perfect':
                    success_flag = True
                    for k,v in rd['infractions'].items():
                        if len(v)>0 and k != 'min_speed_infractions':
                            txt_file.write(f'{route_id_num}\t{rd["route_id"]}\t{rd["scenario_name"]}\t{rd["scores"]["score_composed"]}\t{rd["status"]}\t{k}\t{v}\n')
                            # print(rd['route_id'], k, v)
                            success_flag = False
                            break                            
                    if success_flag:
                        success_num += 1
                        txt_file.write(f'{route_id_num}\t{rd["route_id"]}\t{rd["scenario_name"]}\t{rd["scores"]["score_composed"]}\t{rd["status"]}\n')
                        # print(rd['route_id'])
                else:
                    txt_file.write(f'{route_id_num}\t{rd["route_id"]}\t{rd["scenario_name"]}\t{rd["scores"]["score_composed"]}\t{rd["status"]}\n')
                #     print(rd['status'])
                #     print(rd['route_id'])
    txt_file.close()
    
    if len(merged_records) != 220:
        print(f"-----------------------Warning: there are {len(merged_records)} routes in your json, which does not equal to 220. All metrics (Driving Score, Success Rate, Ability) are inaccurate!!!")
    merged_records = sorted(merged_records, key=lambda d: d['route_id'], reverse=True)
    _checkpoint = {
        "records": merged_records
    }

    merged_data = {
        "_checkpoint": _checkpoint,
        "driving score": sum(driving_score) /  len(driving_score),
        "success rate": success_num /  len(driving_score),
        "eval num": len(driving_score),
    }

    with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
        json.dump(merged_data, file, indent=4)

if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='old foo help', default='eval_results/Bench2Drive/reproduce')
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        merge_route_json(args.folder)
