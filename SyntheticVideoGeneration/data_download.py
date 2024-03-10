import pandas as pd

data = pd.read_csv('results_2M_val.csv')


# make the csv file to 10 rows
data = data[:30]

# save the csv file with name truncated_results_2M_val.csv
data.to_csv('truncated_results_2M_val.csv')


# print all the columns
# print(data.columns)

# get the column name of contentUrl, use wget to save first 10 vids in to a vid folder 
vids  = data['contentUrl']
vid_id = data['videoid']
import os
os.makedirs('vids', exist_ok=True)
# for vid in vids:
def get_vid(vid, vid_id):

    # save the vid with vid_id.mp4
    vid_name = f'vids/{vid_id}.mp4'

    # use wget to download the vid
    os.system(f'wget -O {vid_name} {vid}')

# use multiprocessing to download the vids
import multiprocessing
cpu = multiprocessing.cpu_count()
pool = multiprocessing.Pool(cpu)
pool.starmap(get_vid, zip(vids, vid_id))