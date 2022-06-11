import os
import json
import re
import argparse

def parse_job_type_tuple(job_type):
    match = re.match('\(\'(.*)\', (\d+)\)', job_type)
    if match is None:
        return None
    model = match.group(1).split('(')
    model_name = model[0].rstrip().lstrip()
    batch_size = '1' if model_name == 'CycleGAN' else '4' if len(model) == 1 else model[1].split(' ')[-1].split(')')[0]
    scale_factor = match.group(2)
    return (model_name, scale_factor, batch_size)

# throughput_file = "/home/piyush/Desktop/Study/Research Project/ClusterScheduler/gavel/scheduler/simulation_throughputs.json"
def generate_csv(throughput_file, csv):
    if os.path.exists(throughput_file):
        with open(throughput_file, 'r') as input_file:
            data = json.load(input_file)
            
            if data:
                with open(csv, 'w') as output_file:

                    output_file.write('worker_type,model_name_1,scale_factor_1,batch_size_1,throughput_1,model_name_2,scale_factor_2,batch_size_2,throughput_2\n')

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

def main(args):
    if not os.path.exists(args.throughput):
        raise OSError
    else :
        generate_csv(args.throughput, args.csv)
        with open(args.csv, 'r') as f:
            print(f.read())

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Analyze the provided throughputs file')
    parser.add_argument('--throughput', '-t', type=str, help='Path to throughput file', required=True)
    parser.add_argument('--csv', default='analysis/output.csv', type=str, help='Path to the csv file to generate')
    args = parser.parse_args()
    main(args)