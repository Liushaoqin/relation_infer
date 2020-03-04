import os
import yaml

def reform_text(path):
    pass


if __name__ == '__main__':
    path = 'H:/new_gov_data/滨州市公共数据开放网'
    os.walk(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('csv'):
                try:
                    with open(path + '/' + file, 'r', encoding='gbk') as tmp_file:
                        print(file)
                        line = tmp_file.readline()
                        words = line.split(',')
                        ll = list()
                        tag = False
                        for w in words:
                            if '名称' in w:
                                ll.append(w)
                        if len(ll) > 0:
                            print(ll)
                        print(line)
                except:
                    continue
