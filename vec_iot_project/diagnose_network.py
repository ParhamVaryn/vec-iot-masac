"""Quick sanity check for wireless rates, packet loss and edge load."""
import argparse
import numpy as np
from src.xml_loader import iter_task_timesteps, iter_vehicle_timesteps
from src.masac_env import MASACVECEnv

p=argparse.ArgumentParser()
p.add_argument('--vehicles',default='datasets/vehicles.xml')
p.add_argument('--tasks',default='datasets/tasks.xml')
p.add_argument('--max-tasks',type=int,default=500)
a=p.parse_args()
vehicles={t:v for t,v in iter_vehicle_timesteps(a.vehicles)}
rates=[]; losses=[]; lats=[]; n=0
for t,tasks in iter_task_timesteps(a.tasks):
    if t not in vehicles: continue
    aligned=[x for x in tasks if x.creator in vehicles[t]]
    if not aligned: continue
    env=MASACVECEnv(vehicles[t]); env.set_active_task_creators(x.creator for x in aligned)
    for task in aligned:
        v=vehicles[t][task.creator]
        _,_,mask=env.observation(task,v)
        direct=[]
        for c in env.candidates:
            if c.kind=='DIRECT' and mask[c.index]:
                m=env.evaluate_action(task,v,c.index)
                direct.append(m)
        if direct:
            m=min(direct,key=lambda x:x.latency_s)
            rates.append(m.avg_rate_bps/1e6); losses.append(m.packet_loss); lats.append(m.latency_s)
        n+=1
        if n>=a.max_tasks: break
    if n>=a.max_tasks: break
print('tasks checked:',n)
print('best-direct rate Mbps: mean=',float(np.mean(rates)),'min=',float(np.min(rates)),'max=',float(np.max(rates)))
print('best-direct packet loss: mean=',float(np.mean(losses)))
print('best-direct latency s: mean=',float(np.mean(lats)))
