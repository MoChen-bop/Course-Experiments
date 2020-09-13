import re
import os
from collections import namedtuple

evaluate_result_path = 'G:\\dataset\\corpus\\BNC\\Stem\\evaluateResult'

result_files = {}
for root, dirs, files in os.walk(evaluate_result_path):
    for file in files:
        if file == 'result.txt':
            result_file_path = os.path.join(root, file)
            
            info_list = result_file_path.split('evaluateResult')[1].split('\\')[1:-1]
            domain = info_list[0]
            name = info_list[1]
            for i in info_list[2:]:
                name += '-' + i
            if not domain in result_files.keys():
                result_files[domain] = {}
            result_files[domain][name] = result_file_path

Matrics = namedtuple('matrics', ['OI', 'UI', 'RI', 'AOL', 'AUL', 'P',
                                 'R', 'F1', 'PC', 'RC', 'F1C'])
matrics = {}
for domain, stemmers in result_files.items():
    for stem, file_path in stemmers.items():
#         OI = 0
#         UI = 0
#         RI = 0
#         AOL = 0
#         AUL = 0
#         P = 0
#         R = 0
#         F1 = 0
#         PC = 0
#         RC = 0
#         F1C = 0
        file = open(file_path, 'r', encoding='utf-8')
        for line in file:
            if line[0:2] == 'RI':
                RI = float(line.split()[1])
            elif line[0:2] == 'OI':
                OI = float(line.split()[1])
            elif line[0:2] == 'UI':
                UI = float(line.split()[1])
            elif line[0:3] == 'AOL':
                AOL = float(line.split()[1])
            elif line[0:3] == 'AUL':
                AUL = float(line.split()[1])
            elif line[0:8] == 'Precise:':
                P = float(line.split()[1])
            elif line[0:7] == 'Recall:':
                R = float(line.split()[1])
            elif line[0:3] == 'F1:':
                F1 = float(line.split()[1])
            elif line[0:8] == 'Precise(':
                PC = float(line.split()[1])
            elif line[0:7] == 'Recall(':
                RC = float(line.split()[1])
            elif line[0:3] == 'F1(':
                F1C = float(line.split()[1])
        file.close()
        if domain not in matrics.keys():
            matrics[domain] = {}
        matrics[domain][stem] = Matrics(OI, UI, RI, AOL, AUL, P, R, F1, PC, RC, F1C)

stemmers = ['NOSTEM', 'Porter', 'Lovins', 'HMM-HMM-4-5', 'HMM-HMM-4-6', 'HMM-HMM-5-3',
           'HMM-HMM-5-4', 'HMM-HMM-5-5', 'HMM-HMM-5-6', 'HMM-HMM-6-5', 'SNG-n-3',
           'SNG-n-4', 'SNG-n-5', 'SNG-n-6', 'SNG-n-7', 'YASS-d1-threshold-0.05',
           'YASS-d1-threshold-0.1', 'YASS-d1-threshold-0.15', 'YASS-d1-threshold-0.2',
           'YASS-d1-threshold-0.25', 'YASS-d1-threshold-0.3', 'YASS-d2-threshold-0.1',
           'YASS-d2-threshold-0.2', 'YASS-d2-threshold-0.3', 'YASS-d2-threshold-0.4',
           'YASS-d2-threshold-0.5', 'YASS-d2-threshold-0.6', 'YASS-d3-threshold-0.5',
           'YASS-d3-threshold-1.0', 'YASS-d3-threshold-1.5', 'YASS-d3-threshold-2.0',
           'YASS-d3-threshold-2.5', 'YASS-d3-threshold-3.0', 'YASS-d4-threshold-0.2',
           'YASS-d4-threshold-0.4', 'YASS-d4-threshold-0.6', 'YASS-d4-threshold-0.8',
           'YASS-d4-threshold-1.0', 'YASS-d4-threshold-1.2']

for domain in list(matrics.keys()):
    output_file_path = evaluate_result_path + '\\' + domain + '\\overall_result.txt' 
    output_file = open(output_file_path, 'w', encoding='utf-8')
    output_file.write('\t'.join(['Stemmer_name', 'OI', 'UI', 'RI', 'AOL', 
                      'AUL', 'P', 'R', 'F1', 'PC', 'RC', 'F1C']) + '\n')
    for stemmer in stemmers:
        if stemmer not in matrics[domain].keys():
            continue
        m = matrics[domain][stemmer]
        output_file.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n'
                         % (stemmer, m.OI, m.UI, m.RI, m.AOL, m.AUL, m.P, m.R, m.F1, m.PC, m.RC, m.F1C))
    
    output_file.close()

domains = list(matrics.keys())
for stemmer in stemmers:
    output_file_path = evaluate_result_path + '\\' + stemmer + '.over_result.txt'
    output_file = open(output_file_path, 'w', encoding='utf-8')
    output_file.write('\t'.join(['Domain', 'OI', 'UI', 'RI', 'AOL', 
                      'AUL', 'P', 'R', 'F1', 'PC', 'RC', 'F1C']) + '\n')
    for domain in domains:
        if stemmer not in matrics[domain].keys():
            continue
        m = matrics[domain][stemmer]
        output_file.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n'
                         % (domain, m.OI, m.UI, m.RI, m.AOL, m.AUL, m.P, m.R, m.F1, m.PC, m.RC, m.F1C))
    output_file.close()