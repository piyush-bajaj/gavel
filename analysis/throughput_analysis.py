import os
import json
import re

def parse_job_type_tuple(job_type):
    match = re.match('\(\'(.*)\', (\d+)\)', job_type)
    if match is None:
        return None
    model = match.group(1).split('(')
    model_name = model[0].rstrip().lstrip()
    batch_size = '0' if len(model) == 1 else model[1].split(' ')[-1].split(')')[0]
    scale_factor = match.group(2)
    return (model_name, scale_factor, batch_size)

throughput_file = "/home/piyush/Desktop/Study/Research Project/ClusterScheduler/gavel/scheduler/simulation_throughputs.json"
if os.path.exists(throughput_file):
    with open(throughput_file, 'r') as input_file:
        data = json.load(input_file)
        
        if data:
            with open("analysis/test.csv", 'w') as output_file:

                for worker_type in data:
                    
                    for model in data[worker_type] :                        

                        for second_model in data[worker_type][model]:

                            output_file.write(worker_type) #worker k80, p1000, v100
                            output_file.write(',')
                            output_file.write(parse_job_type_tuple(model)[0]) #model Resnet-18, Transformer etc
                            output_file.write(',')
                            output_file.write(parse_job_type_tuple(model)[1]) #scaling factor (no. of workers) 1, 2, 4, 8
                            output_file.write(',')
                            output_file.write(parse_job_type_tuple(model)[2])   #batch size of model 16, 32, etc 0 if parameter absent
                            output_file.write(',')

                            if second_model == 'null' : #space sharing is off
                                output_file.write(str(data[worker_type][model][second_model])) #throughput value
                                output_file.write(',')
                                output_file.write('NaN') #model Resnet-18, Transformer etc
                                output_file.write(',')
                                output_file.write('NaN') #scaling factor (no. of workers) 1, 2, 4, 8
                                output_file.write(',')
                                output_file.write('NaN')   #batch size of model 16, 32, etc 0 if parameter absent
                                output_file.write(',')
                                output_file.write('NaN')
                            else : #space sharing is on
                                output_file.write(str(data[worker_type][model][second_model][0])) #throughput value for first model in space sharing
                                output_file.write(',')
                                output_file.write(parse_job_type_tuple(second_model)[0]) #model Resnet-18, Transformer etc
                                output_file.write(',')
                                output_file.write(parse_job_type_tuple(second_model)[1]) #scaling factor (no. of workers) 1, 2, 4, 8
                                output_file.write(',')
                                output_file.write(parse_job_type_tuple(second_model)[2])   #batch size of model 16, 32, etc 0 if parameter absent
                                output_file.write(',')
                                output_file.write(str(data[worker_type][model][second_model][1])) #throughput value for second model in space sharing

                            #new line after each combination
                            output_file.write('\n')


