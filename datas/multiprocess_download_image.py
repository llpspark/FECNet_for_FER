import os
from multiprocessing import Process, Queue, Lock
import requests
import time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
proc_num = 20
queue_size = 2048
url_set = set()
urls_queue = Queue(queue_size)
queue_lock = Lock()

save_path = "./fec_data"

def producer(src_file_path, urls_queue):
    with open(src_file_path, "r") as src_fr:
        index = 0
        ori_list = src_fr.readlines()
        for imgs_line in ori_list:
            for img_url in imgs_line.split(",")[0: 11: 5]:
                index = index + 1
                if 0 == index % 2000:
                    print("now process is %d / %d " % (index, len(ori_list)))
                if img_url in url_set:
                    continue
                else:
                    url_set.add(img_url)
                    while urls_queue.full():
                        time.sleep(1)
                    queue_lock.acquire()
                    urls_queue.put(img_url)
                    queue_lock.release()
        for i in range(queue_size):
            urls_queue.put("finish")


def consumer(save_path, urls_queue):
    while True:
        while urls_queue.empty():
            time.sleep(1)
        img_url = urls_queue.get()
        img_url = eval(img_url)
        if img_url == "finish":
            break
        img_name = img_url.split("/")[-1]
        try:
            respone = requests.get(img_url,headers=headers)
            if 200 != respone.status_code:
                print("url is invalid %s " % img_url)
                continue
            img = respone.content
            with open(os.path.join(save_path, img_name), "wb") as img_fw:
                img_fw.write(img)
        except:
            continue

class MultiConsumer(Process):
    def __init__(self, proc_id):
        self.proc_id = proc_id
        super(MultiConsumer, self).__init__()
    def run(self):
        consumer(save_path, urls_queue)


if __name__ == "__main__":
    src_file_path = "/home/gary/grocery/face_studio/fd_studio/dataset/FEC/faceexp-comparison-data-train-public.csv"

    produce_urls = Process(target=producer, args=(src_file_path, urls_queue))
    produce_urls.start()

    process_list = []
    for i in range(proc_num):
        process_list.append(MultiConsumer(i))
    for consumer_proc in process_list:
        consumer_proc.start()
    for consumer_proc in process_list:
        consumer_proc.join()

    produce_urls.join()






