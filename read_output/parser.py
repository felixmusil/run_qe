import re,os,sys
import numpy as np



def get_filenames(path,fn_pattern='qe.out',dir_pattern=None):
    '''
    get all the filenames within path having fn_pattern (and dir_pattern)
    :param path: 
    :param fn_pattern: 
    :param dir_pattern: 
    :return: list of filenames with absolute path
    '''
    import fnmatch
    import os
    matches = []
    if dir_pattern is None:
        # Find the paths of all the filename with fn_pattern within path
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, fn_pattern):
                matches.append(os.path.join(root, filename))
        return matches
    else:
        # Find the paths with dir_pattern and all the filename with fn_pattern within dir_pattern (no recursion)
        for root, dirnames, filenames in os.walk(path):
            for dirname in fnmatch.filter(dirnames, dir_pattern):
                filenames = os.listdir(os.path.join(root,dirname))
                for filename in fnmatch.filter(filenames, fn_pattern):
                    matches.append(os.path.abspath(os.path.join(root,dirname, filename)))


def get_patterns(fn, str_patterns):
    '''
    grep str_patterns in file fn
    
    :param fn: 
    :param str_patterns: 
    :return: dic (key,val)->(str_patterns,list of found lines)
    '''
    matches = {key: [] for key in str_patterns}

    for line in open(fn, 'r'):
        for str_pattern in str_patterns:
            if re.search(str_pattern, line):
                matches[str_pattern].append(line.replace('\n', ''))
    return matches

def fn2num(fn,pattern="sg_(.*)-f_"):
    ms = re.search(pattern,fn)
    num = int(ms.group(1))
    return num

def fn2info(fn):
    '''
    Find informations in absolute filename
    :param fn: 
    :return: dic of information, e.g. {'sg':3,'f':0}
    '''
    dirnames = fn.split('/')
    for dirname in dirnames:
        if re.search('sg_',dirname) and re.search('f_',dirname):
            dic = {}
            for info in dirname.split('-'):
                if re.search('_',info):
                    name,num = info.split('_')
                    num = int(num)
                    dic[name] = num
    return dic

def extract_floats(s):
    '''
    get list of floats from a string
    
    :param s: 
    :return: 
    '''
    l = []
    for t in s.split():
        try:
            l.append(float(t))
        except ValueError:
            pass
    return l

def finishedProperly(fn):
    '''
    Check is the qe.out finished properly
    :param fn: 
    :return: 
    '''
    str_patterns = ['JOB DONE.']
    s = get_patterns(fn,str_patterns)['JOB DONE.']
    if s:
        return True
    else:
        return False

def get_energy_per_atom(fn):
    '''
    get the energy per atom in eV from a qe.out
    :param fn: 
    :return: dic with energy 'en [eV/atom]' and 
        info from dirname, e.g. {'en [eV/atom]': -154.6159,'Natom':4.0, 'sg': 35, 'f': 1}
    '''
    Ryd2eV = 13.605698066
    str_patterns = ['total energy','number of atoms/cell']
    info = fn2info(fn)
    if finishedProperly(fn):
        s = get_patterns(fn,str_patterns)
        nat = extract_floats(s['number of atoms/cell'][0])[0]
        en = extract_floats(s['total energy'][0])[0]*Ryd2eV / nat
        info.update({'en [eV/atom]':en,'Natom':nat})
        return info
    else:
        return None