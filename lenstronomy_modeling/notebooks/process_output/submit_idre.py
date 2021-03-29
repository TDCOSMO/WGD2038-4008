import sys
import os
import pickle
import time

file_name = str(sys.argv[1])
job_type = str(sys.argv[2])

compute_chunk = int(sys.argv[3])

job_name_list = input

for i in range(int(10000/compute_chunk)):
    start_index = i*compute_chunk
    os.system('./idre_submit_job.sh '+str(file_name)+' '+str(job_type)+' '+str(start_index)+' '+str(compute_chunk))
    print('./idre_submit_job.sh '+str(file_name)+' '+str(job_type)+' '+str(start_index)+' '+str(compute_chunk))
    time.sleep(1)

print(i+1, 'jobs submitted!')
