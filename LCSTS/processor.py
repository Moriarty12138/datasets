#!/usr/bin/env python
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

INPUT = {
    'valid': './data/PART_III.txt',
    'test': './data/PART_II.txt',
    'train': './data/PART_I.txt'
}

OUTPUT_DIR = 'LCSTS/'

for key in INPUT:
    print('start process: {}\n'.format(key))
    src_file = open(os.path.join(OUTPUT_DIR, key + '.src'), 'a+', encoding='utf-8')
    tgt_file = open(os.path.join(OUTPUT_DIR, key + '.tgt'), 'a+', encoding='utf-8')

    input_file_path = INPUT[key]
    with open(input_file_path, encoding='utf-8') as file:
        contents = file.read()
        soup=BeautifulSoup(contents,'html.parser')
        line_count = 0
        max_lines = 2078888  # 此处设置了一个变量控制最大解析多少条数据
        for doc in tqdm(soup.find_all('doc')):
            if line_count > max_lines:
                break
            short_text = doc.find('short_text').get_text()
            summary = doc.find('summary').get_text()
            src_file.write(short_text.strip() + '\n')
            tgt_file.write(summary.strip() + '\n')
            line_count += 1

    src_file.close()
    tgt_file.close()
