import requests
import multiprocessing  
from multiprocessing import Queue, Lock
import os
#from queue import Queue
import time

start_lines =0 
processer_num = 40
save_data_path='./train_data'
global_queue = Queue(1024)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}

def fill_que_process(filename,q):
  fid = open(filename,'r')
  lines = fid.readlines()
  already_processed = set()
  for i in range(0,start_lines):
    words = lines[i].split(',')
    already_processed.add(words[0])
    already_processed.add(words[5])
    already_processed.add(words[10])
  for i in range(start_lines,len(lines)):
    words = lines[i].split(',')
    imgs = words[0:11:5]
    for img in imgs:
      if img in already_processed:
        continue
      else:
        while q.full():
          time.sleep(1)
        mutex.acquire()
        q.put(img)
        mutex.release()
		already_processed.add(img)
  for i in range(1024):
    q.put('finish')
    
          
          
    
  

class ProcWorker(multiprocessing.Process):
  def __init__(self,pid):
    super().__init__()
    self._pid = pid
  def run(self):
    while True:
      while global_queue.empty():
        time.sleep(1)
      mutex.acquire()
      img = global_queue.get(timeout=1)
      mutex.release()
      img = eval(img)
      try:
        x= requests.get(img,headers=headers)
        if x.status_code != 200 :
          print(img,'failer!!!')
          continue
        fname = img.split('/')[-1]
        fid = open(os.path.join(save_data_path,fname),'wb')
        fid.write(x.content)
        fid.close()
      except:
        continue
      print(img,self._pid)
      if img == 'finish':
        break

mutex = Lock()

f_p = multiprocessing.Process(target = fill_que_process,args =('faceexp-comparison-data-train-public.csv',global_queue))
f_p.start()

proc_stack = []
for i in range(processer_num):
  proc_stack.append(ProcWorker(i))

for p in proc_stack:
  p.start()

for p in proc_stack:
  p.join()

f_p.join()
