import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv

from ir_system import IRSystem

# empty list and variable
WORD_OCCURANCE = 'word_occurance'
FILE_OCCURANCE = 'file_occurance'
PERMITTED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ " 

def OR_process(line):
    return ' OR '.join(line)

def AND_process(line):
    return line.replace(' ', ' AND ')[:-5]
        
stop_word = ['the', 'this', 'that', 'these', 'those', 'there', 'their', 'they', 'them',
            'be', 'is', 'am', 'are', 'we', 'I', 'us', 'he', 'she', 'it', 'a', 'an', 'our', 'been', 'was', 'were',
            'what', 'who', 'where', 'which', 'how', 'why', 'when', 'do', 'does', 'did', 'done',
            'of', 'by','and', 'or', 'for', 'on', 'in', 'at', 'to', 'out', 'from', 'some', 'into', 'off',
            'can', 'must', 'should', 'could', 'may', 'might', 'have', 'had', 'has',
            'thus', 'therefore', 'however', 'but', 'maybe', 'either', 'nor', 'neither',
            '/', '-', '_', ',', '!','?', '.', '+', '*', '#', '@', '%', '(', ')',  "'", '"']

# loop to add text files to a list
def list_of_txt_file():
    txt_files = []
    for file in glob.glob("./Cranfield/*.txt"):
        txt_files.append(file)
    return txt_files

def list_of_query_file():
    file = open("./TEST/query.txt", "r")
    data = [removeSpecialCharacter(line) for line in file.readlines()]
    return data

def removeSpecialCharacter(string):
    return "".join(c for c in string if c in PERMITTED_CHARS)

def count_words(filename):
    txt_files = list_of_txt_file()
    words = {}

    for i, f in enumerate(txt_files):
        file = open(f, "r")
        read_data = file.read()
        per_word = read_data.split()
        appeared_in_file = {}
        for word in per_word:
            safe_word = removeSpecialCharacter(word)
            if len(safe_word) == 0:
                continue
            if word in words:
                if appeared_in_file.get(word, -244) == -244:
                    words[safe_word] = { WORD_OCCURANCE: words[safe_word][WORD_OCCURANCE] + 1, FILE_OCCURANCE: words[safe_word][FILE_OCCURANCE] + 1 }
                else:
                    words[safe_word] = { WORD_OCCURANCE: words[safe_word][WORD_OCCURANCE] + 1, FILE_OCCURANCE: words[safe_word][FILE_OCCURANCE] }
            else:
                words[safe_word] = { WORD_OCCURANCE: 1, FILE_OCCURANCE: 1 }

            appeared_in_file[word] = 1

            
    with open(filename, 'w', newline='') as f:
        fieldnames = ['Word', 'Frequency', 'Contained']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in words.keys():
            writer.writerow({fieldnames[0]: key, fieldnames[1]: words[key][WORD_OCCURANCE], fieldnames[2]: words[key][FILE_OCCURANCE]})

def plotting():
    df = pd.read_csv('my-data.csv')
    df.sort_values(by=['Frequency'], ascending=False, inplace = True)
    print(df.head(10))
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
    ax1.set_title('Frequency')
    ax2.set_title('Contained')
    
    ax1.bar(df['Word'][:100], df['Frequency'][:100])
    ax2.bar(df['Word'][:100], df['Contained'][:100])
    
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    plt.show()

def init(load):
    filelist = list_of_txt_file()
    read_data = []

    for i, f in enumerate(filelist):
        file = open(f, "r")
        read_data.append(file.read())
    
    ir = IRSystem(read_data, stop_word, load)
    return ir

def query_process(ir):
    query = list_of_query_file()
    
    with open("./edited_query.txt", 'w') as f:
        for i in range(len(query)):
            process_query = OR_process(ir.stem_query(query[i]))
            f.write(process_query)
            f.write('\n')
    f.close()
    
def read_query():
    file = open("./edited_query.txt", 'r')
    data = [removeSpecialCharacter(line) for line in file.readlines()]
    return data

ir = init(True)
query_process(ir)
query = read_query()

print(query[2])
print(len(ir.process_query(query[2])))