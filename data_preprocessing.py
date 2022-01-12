import os
import time
import numpy as np

def ts_read_line(line):
    line = line.strip().split(',')
    ts = [float(x) for x in line[2:]]
    return line[1], ts

def text_read_line(line):
    line = line.strip()
    content = line.split(',')
    t = time.strptime(content[0], '%Y/%m/%d %H:%M')
    t = time.strftime('%Y%m%d', t)
    content = ''.join(content[2:]).replace('\t', '').replace(' ', '').replace('\n', '')
    return t, content

def load_data(ts_path, text_path, stock_ids, MAX_TEXT_LEN = 510, WINDOW_SIZE = 5):
    all_data = {}
    ts_files = os.listdir(ts_path)
    for ts_file in ts_files:
        stock_id = ts_file.split('.')[0]
        ts_file = os.path.join(ts_path, ts_file)
        with open(ts_file, 'r') as f:
            ts_data = f.readlines() 
        ts_data = ts_data[1:]

        text_file = os.path.join(text_path, stock_id + '.csv')
        with open(text_file, 'r') as f:
            text_data = f.readlines()
        text_data = text_data[1:]
        
        all_data[stock_id] = {'ts': ts_data, 'text': text_data}

    means = []
    stds = []
    np_ts_data = []
    np_text_data = []

    for stock_id in stock_ids: #all_data:
        ts_data = all_data[stock_id]['ts']
        text_data = all_data[stock_id]['text']
        ts_time_dict = {}
        for ts_line in ts_data:
            t, ts = ts_read_line(ts_line)
            ts_time_dict[t] = ts

        text_time_dict = {}
        for text_line in text_data:
            t, text = text_read_line(text_line)
            if t not in ts_time_dict:
                continue
            if t in text_time_dict:
                last_text = text_time_dict[t][-1]
                del text_time_dict[t][-1]
                last_text = last_text + text
                if len(last_text) > MAX_TEXT_LEN:
                    texts = [last_text[i:i+MAX_TEXT_LEN] for i in range(0,len(last_text),MAX_TEXT_LEN)]
                    text_time_dict[t].extend(texts)
                else:
                    text_time_dict[t].append(last_text)
            else:
                if len(text) > MAX_TEXT_LEN:
                    texts = [text[i:i+MAX_TEXT_LEN] for i in range(0,len(text),MAX_TEXT_LEN)]
                    text_time_dict[t] = texts
                else:
                    text_time_dict[t] = [text]

        sorted_ts_time_dict = sorted(ts_time_dict.items(), key=lambda x:x[0])
        sorted_ts_time_value_list = [x[1] for x in sorted_ts_time_dict]
        sorted_text_time_dict = sorted(text_time_dict.items(), key=lambda x:x[0])
        sorted_text_time_value_list = [x[1] for x in sorted_text_time_dict]

        normalized_ts_value = np.array(sorted_ts_time_value_list)
        mean = np.mean(normalized_ts_value, axis=0)
        std = np.std(normalized_ts_value, axis=0)
        means.append(mean)
        stds.append(std)
        normalized_ts_value = (normalized_ts_value - mean) / std

        for i in range(len(sorted_ts_time_value_list) - WINDOW_SIZE + 1):
            ts_window = normalized_ts_value[i:i+WINDOW_SIZE]
            text_window = sorted_text_time_value_list[i:i+WINDOW_SIZE]
            np_ts_data.append(ts_window)
            np_text_data.append(text_window)

    return np.array(np_ts_data), np_text_data, means, stds

if __name__ == '__main__':
    ts_path = 'data/股价'
    text_path = 'data/文本数据'
    np_ts_data, np_text_data, means, stds = load_data(ts_path, text_path, ['000001','000858'])
    print(np_ts_data[0])


    

        



    


    



