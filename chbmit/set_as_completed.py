import os
import json
import shutil


def extract_performance(logfile):
    lastline = logfile.split('\n')[-2]
    last_iter = int(lastline.split("Iter: ")[-1].split(",")[0])
    likelihood = float(lastline.split("likelihood: ")[-1].split(",")[0])
    diff = float(lastline.split("difference: ")[-1])
    return last_iter, likelihood, diff


def mark_complete(path_scheduler, path_logs_folder):

    performance = {}

    with open(path_scheduler) as infile:
        info = json.load(infile)
        print(f'Number of experiments: {len(info)}')

        for k, v in info.items():
            if v['method'] != 'hmm': 
                continue
            trained = False
            p_id = v['dataset']['patient'] # patient id e.g. chb01
            id_exp = v['id']
            max_iter = v['hyperparams_method']['max_iter'] # n max of iterations
            tol = v['hyperparams_method']['tol']

            if not os.path.exists(v['output_path']):
                print(f"Experiment {id_exp} never run")
                continue

            trained = 'model.pkl' in os.listdir(v['output_path'])
            if trained:
                pat_logs = f'chb0{p_id}' if p_id < 10 else  f'chb{p_id}'
                with open(os.path.join(path_logs_folder, pat_logs, f'logexp_{id_exp}.txt'), 'r') as f:
                    logfile = f.read()
                try:
                    last_i, likelih, diff = extract_performance(logfile)
                except:
                    print(" ")

                if last_i < max_iter - 1 and diff <= tol:
                    print(f"Index: {id_exp}, iter: {last_i}, likelihood: {likelih}, difference: {diff}")
                    performance[id_exp] = {"last iteration": last_i, 
                                           "likelihood": likelih,
                                           "difference": diff}
                    v['train_completed'] = True
                    src = os.path.join(v['output_path'], 'model.pkl')
                    dst = os.path.join(v['output_path'], 'model_cp.pkl')
                    shutil.copyfile(src, dst)
                else:
                    v['train_completed'] = False


    with open(path_scheduler, 'w') as outfile:
        json.dump(info, outfile, indent=4)

    with open(os.path.join(os.path.dirname(path_scheduler), 'performance.json'), 'w') as outfile:
        json.dump(performance, outfile, indent=4)
