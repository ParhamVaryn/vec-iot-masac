from src.xml_loader import load_vehicles, load_tasks
from src.simulator import decide

vehicles={v.id:v for v in load_vehicles('datasets/vehicles.xml')}
tasks=load_tasks('datasets/tasks.xml')

print('='*60)
print('VEC-IoT Phase 3.1 Dataset Driven Demo')
print('Calibrated MEC + UAV Relay Simulator')
print('='*60)

results={'LOCAL':0,'EDGE':0,'UAV_RELAY':0}
latencies=[]

for t in tasks:
    if t.creator not in vehicles:
        continue
    v=vehicles[t.creator]
    choice, scores=decide(t,v)
    results[choice]+=1
    latencies.append(scores[choice][0])

    print(f'\nTask {t.id} ({t.creator})')
    for k,(lat,en) in scores.items():
        print(f' {k:10s}: latency={lat:.4f}s energy={en:.6f}J')
    print(' Selected:', choice)

print('\nSUMMARY')
print('Processed tasks:', len(latencies))
print(results)
print('Average selected latency:', sum(latencies)/len(latencies), 's')