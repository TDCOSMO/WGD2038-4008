"""
For testing on the local machine, run from command line:

```
python process_output.py <run_id> powerlaw 0 5
```

This will compute velocity dispersion for the first 5 samples from the chain
of <run_id>. The <run_id>_out.txt file needs to be in `base_dir`.
"""

import numpy as np
import sys

from output_class import ModelOutput
from output_class import custom_loglikelihood_addition

is_cluster = False

run_name = str(sys.argv[1])
run_type = str(sys.argv[2])

output_file = run_name #+ '_out.txt'

start_index = int(sys.argv[3])
num_compute = int(sys.argv[4])

print(output_file, run_type, start_index)

if not is_cluster:
    base_dir = '/Users/ajshajib/Research/time_delay_cosmography/J2038/WGD2038'\
               '-4008/lenstronomy_modeling/temp/'
    out_dir = '/Users/ajshajib/Research/time_delay_cosmography/J2038/vel_dis_test/'
else:
    base_dir = '/u/flashscratch/a/ajshajib/2038_shortened_outputs/'
    out_dir = '/u/flashscratch/a/ajshajib/2038_vel_dis/'

output = ModelOutput(output_file, run_type, base_dir, '_out.txt',

                     is_test=False)
print('loaded {}'.format(base_dir+output_file))
print('model type: {}'.format(run_type))

#output.compute_model_time_delays()
#output.save_time_delays()
#print('finished computing time delays', output.model_time_delays.shape)

output.compute_model_velocity_dispersion(start_index=start_index,
                                         num_compute=num_compute,
                                         print_step=5)

np.savetxt(out_dir+'vd_{}_'.format(start_index)+output_file+'_out.txt',
           output.model_velocity_dispersion)
np.savetxt(out_dir+'aani_{}_'.format(start_index)+output_file+'_out.txt',
           output.a_ani)
np.savetxt(out_dir+'reff_{}_'.format(start_index)+output_file+'_out.txt',
           output.r_eff)

print('finished computing velocity dispersions', output.model_velocity_dispersion.shape)