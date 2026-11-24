import sys, json, numpy as np, os
D=sys.argv[1]; OUT=sys.argv[2]
def verts(p):
    out=[]
    with open(p) as f:
        for l in f:
            if l.startswith('v '): out.append(l.split()[1:4])
    return np.array(out,np.float32)
neutral=verts(D+'/generic_neutral_mesh.obj')
# full topology of neutral
vt=[];faces=[]
with open(D+'/generic_neutral_mesh.obj') as f:
    for l in f:
        if l.startswith('vt '): vt.append(l.split()[1:3])
        elif l.startswith('f '):
            faces.append([tuple(int(x)-1 for x in c.split('/')[:2]) for c in l.split()[1:]])
vt=np.array(vt,np.float32)
cfg=json.load(open(D+'/vertex_indices.json'))
exprs=cfg['expressions']; ids=cfg.get('identities') or sorted(x[:-4] for x in os.listdir(D) if x.startswith('identity'))
E=np.stack([verts(f'{D}/{n}.obj')-neutral for n in exprs])
I=np.stack([verts(f'{D}/{n}.obj')-neutral for n in ids])
np.savez_compressed(OUT, neutral=neutral, vt=vt, E=E, I=I, exprs=np.array(exprs), ids=np.array(ids))
import pickle; pickle.dump(faces, open(OUT.replace('.npz','_faces.pkl'),'wb'))
print(neutral.shape, vt.shape, E.shape, I.shape, len(faces), {len(f) for f in faces})
print('bbox', neutral.min(0), neutral.max(0))
print([k for k in cfg.keys()])
