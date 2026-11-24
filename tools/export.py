import sys, json, numpy as np, pickle, struct
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as Rot
W, OUT = sys.argv[1], sys.argv[2]
m=np.load(W+'/ict.npz'); f=np.load(W+'/fit.npz'); faces=pickle.load(open(W+'/ict_faces.pkl','rb'))
neutral,E,I=m['neutral'],m['E'],m['I']; exprs=list(m['exprs']); NI=int(f['NI'])
IMG_W, IMG_H = 1478, 1723
ex0=f['ex0'].copy()
def wv(d):
    """Expression = her photo expression (ex0) + deltas, clipped to 0..1.
    Building on ex0 keeps her pressed/rolled lips and eye squint as in the photo,
    so no hidden lip surface is ever exposed and stretched."""
    w=ex0.copy()
    for k,v in d.items():
        for side in (['_L','_R'] if (k+'_L') in exprs else ['']):
            i=exprs.index(k+side); w[i]=np.clip(w[i]+v,0,1)
    return w
# Full smile = what mood 0.47 looked like with the old recipe (beyond that it got weird).
SMILE_MAX=0.455
SMILE=wv({k:v*SMILE_MAX for k,v in {'mouthSmile':0.85,'mouthStretch':0.15,'cheekSquint':0.3,'mouthDimple':0.15,'mouthPress':-0.15}.items()})
FROWN=wv({'mouthFrown':0.9,'mouthShrugLower':0.3,'browInnerUp':0.9,'mouthPress':0.1,'eyeBlink':0.1})
BLINK=ex0.copy(); BLINK[exprs.index('eyeBlink_L')]=1; BLINK[exprs.index('eyeBlink_R')]=1
Vid=neutral+np.tensordot(f['a'],I[:NI],1)
p=f['pose']; R=Rot.from_rotvec(p[:3]); s=p[3]
def xf(V): Y=R.apply(V)*s; return np.stack([Y[:,0]+p[4], Y[:,1]+(IMG_H-p[5]), Y[:,2]],1)
base=xf(Vid+np.tensordot(ex0,E,1))
tS=xf(Vid+np.tensordot(SMILE,E,1)); tF=xf(Vid+np.tensordot(FROWN,E,1)); tB=xf(Vid+np.tensordot(BLINK,E,1))
# vertex subset + material ids
# Mouth stays closed, so no socket/gums/teeth (mats 2-4) — gaps show the photo instead.
ranges=[(0,9408,0),(21451,24590,1)]
mat=np.full(len(neutral),255,np.uint8)
for a,b,mm in ranges: mat[a:b+1]=mm
sel=np.where(mat!=255)[0]; remap=-np.ones(len(neutral),int); remap[sel]=np.arange(len(sel))
tris=[]
for fc in faces:
    vi=[c[0] for c in fc]
    if min(remap[vi])<0: continue
    vi=[remap[v] for v in vi]
    tris.append(vi[:3]);
    if len(vi)==4: tris.append([vi[0],vi[2],vi[3]])
tris=np.array(tris,np.int64)
def normals(P):
    n=np.zeros_like(P); a,b,c=P[tris[:,0]],P[tris[:,1]],P[tris[:,2]]
    fn=np.cross(b-a,c-a)
    for i in range(3): np.add.at(n,tris[:,i],fn)
    return n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
B=base[sel]; S_=tS[sel]-B; F_=tF[sel]-B; K_=tB[sel]-B
nB=normals(B); nS=normals(B+S_); nF=normals(B+F_)
# orientation check: normals should point toward viewer (+z) on the face
print('face normal mean z', nB[mat[sel]==0][:,2].mean())
# fade: rings from outer boundary of the face region
fsel=(mat[sel]==0)
edges=defaultdict(int)
for t in tris:
    if not all(fsel[t]): continue
    for i in range(3):
        e=tuple(sorted((t[i],t[(i+1)%3]))); edges[e]+=1
bedges=[e for e,c in edges.items() if c==1]
adj=defaultdict(list)
for a,b in bedges: adj[a].append(b); adj[b].append(a)
seen=set(); loops=[]
for v in adj:
    if v in seen: continue
    comp=[]; q=[v]; seen.add(v)
    while q:
        x=q.pop(); comp.append(x)
        for y in adj[x]:
            if y not in seen: seen.add(y); q.append(y)
    loops.append(comp)
loops.sort(key=len,reverse=True); print('boundary loops', [len(l) for l in loops[:6]])
outer=loops[0]
nbr=defaultdict(set)
for t in tris:
    for i in range(3): nbr[t[i]].add(t[(i+1)%3]); nbr[t[(i+1)%3]].add(t[i])
dist=np.full(len(sel),999); dq=deque()
for v in outer: dist[v]=0; dq.append(v)
while dq:
    x=dq.popleft()
    for y in nbr[x]:
        if fsel[y] and dist[y]>dist[x]+1: dist[y]=dist[x]+1; dq.append(y)
RINGS=10
fade=np.where(fsel, np.clip(dist/RINGS,0,1), 1.0)
# for interior materials, reuse the fade byte as a depth-based cavity shade
inner=~fsel & (mat[sel]!=1)
zt=B[:,2].max()
fade=np.where(inner, np.clip(1-(zt-B[:,2])/110.0,0.12,1.0)**1.5, fade)
# quantize
lo=B.min(0); hi=B.max(0); posq=np.round((B-lo)/(hi-lo)*65535-32768).astype(np.int16)
def qd(D): sc=np.abs(D).max(); return np.round(D/sc*32767).astype(np.int16), float(sc)
Sq,sS=qd(S_); Fq,sF=qd(F_); Kq,sK=qd(K_)
def qn(N): return np.round(np.concatenate([N,np.zeros((len(N),1))],1)*127).astype(np.int8)
extra=np.stack([mat[sel], np.round(fade*255).astype(np.uint8)],1).astype(np.uint8)
idx=tris.astype(np.uint16).ravel()
assert len(sel)<65536
blobs=[('pos',posq),('dSmile',Sq),('dFrown',Fq),('dBlink',Kq),('n',qn(nB)),('nSmile',qn(nS)),('nFrown',qn(nF)),('extra',extra),('index',idx)]
hdr={'count':int(len(sel)),'tris':int(len(tris)),'lo':lo.tolist(),'hi':hi.tolist(),'scale':{'smile':sS,'frown':sF,'blink':sK},'img':[IMG_W,IMG_H],'layout':[]}
off=0; body=b''
for name,arr in blobs:
    bts=np.ascontiguousarray(arr).tobytes()
    pad=(-len(bts))%4
    hdr['layout'].append({'name':name,'offset':off,'bytes':len(bts),'dtype':str(arr.dtype)})
    body+=bts+b'\0'*pad; off+=len(bts)+pad
h=json.dumps(hdr).encode(); h+=b' '*((-len(h))%4)
open(OUT,'wb').write(struct.pack('<I',len(h))+h+body)
print('verts',len(sel),'tris',len(tris),'bytes',4+len(h)+len(body)); print({k:hdr['scale'][k] for k in hdr['scale']})
