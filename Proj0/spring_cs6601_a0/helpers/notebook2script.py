# Credits: https://github.com/fastai/course-v3/tree/master/nbs/dl2

import os
import sys
import re
import json

def is_export(cell):
    if cell['cell_type'] != 'code': return False
    src = cell['source']
    if len(src) == 0 or len(src[0]) < 7: return False
    return re.match(r'^\s*#\s*export\s*$', src[0], re.IGNORECASE) is not None

def notebook2scriptSingle(fname):
    "Finds cells starting with `#export` and puts them into a new module"
    fname_out = 'submission.py'
    main_dic = json.load(open(fname,'r',encoding="utf-8"))
    code_cells = [c for c in main_dic['cells'] if is_export(c)]
    module = f'''
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: {fname}

'''
    for cell in code_cells: module += ''.join(cell['source'][1:]) + '\n\n'
    # remove trailing spaces
    module = re.sub(r' +$', '', module, flags=re.MULTILINE)
    output_path = fname_out
    open(output_path,'w',encoding="utf-8").write(module[:-2])
    print(f"Converted {fname} to {output_path}")

if __name__ == '__main__':
    notebook2scriptSingle('notebook.ipynb')