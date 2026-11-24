import sys, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
W=sys.argv[1]
m=np.load(W+'/ict.npz'); L=np.load(W+'/lm.npz')
neutral,E,I=m['neutral'],m['E'],m['I']; exprs=list(m['exprs'])
ICT68=[1225,1888,1052,367,1719,1722,2199,1447,966,3661,4390,3927,3924,2608,3272,4088,3443,268,493,1914,2044,1401,3615,4240,4114,2734,2509,978,4527,4942,4857,1140,2075,1147,4269,3360,1507,1542,1537,1528,1518,1511,3742,3751,3756,3721,3725,3732,5708,5695,2081,0,4275,6200,6213,6346,6461,5518,5957,5841,5702,5711,5533,6216,6207,6470,5517,5966]
MP68=[162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
lm=L['lm']; tgt=lm[MP68,:2]
w=np.ones(68); w[:17]=0.4; w[0]=w[16]=0.15; w[17:27]=0.8; w[60:68]=1.5; w[36:48]=1.5
# initial expression from MediaPipe blendshapes (ARKit names -> ICT names)
arkit={'eyeSquintLeft':('eyeSquint_L',.534),'eyeSquintRight':('eyeSquint_R',.401),'mouthPressLeft':('mouthPress_L',.674),'mouthPressRight':('mouthPress_R',.527),
 'mouthRollLower':('mouthRollLower',.489),'mouthRollUpper':('mouthRollUpper',.261),'mouthShrugLower':('mouthShrugLower',.5),
 'mouthDimpleLeft':('mouthDimple_L',.153),'mouthDimpleRight':('mouthDimple_R',.214),'eyeBlinkLeft':('eyeBlink_L',.077),'eyeBlinkRight':('eyeBlink_R',.068)}
# NB: MediaPipe "Left" = subject's left; ICT _L = subject's left (assumed)
ex0=np.zeros(len(exprs))
for k,(n,v) in arkit.items(): ex0[exprs.index(n)]=v
NI=60
Il=I[:NI][:,ICT68]; El=E[:,ICT68]; nl=neutral[ICT68]
def shape(a,ex): return nl+np.tensordot(a,Il,1)+np.tensordot(ex,El,1)
def proj(p,X):
    rv,s,tx,ty=p[:3],p[3],p[4],p[5]
    Y=Rot.from_rotvec(rv).apply(X)*s
    return np.stack([Y[:,0]+tx, -Y[:,1]+ty],1)
def res(x):
    p=x[:6]; a=x[6:6+NI]
    P=proj(p,shape(a,ex0))
    r=((P-tgt)*w[:,None]).ravel()
    return np.concatenate([r, a*3.0])
# init scale: interocular
s0=np.linalg.norm(tgt[45]-tgt[36])/np.linalg.norm(nl[45,:2]-nl[36,:2])
c=tgt[27:36].mean(0); x0=np.zeros(6+NI); x0[3]=s0
x0[4]=c[0]-s0*nl[27:36,0].mean(); x0[5]=c[1]+s0*nl[27:36,1].mean()
sol=least_squares(res,x0,x_scale='jac')
x=sol.x; P=proj(x[:6],shape(x[6:6+NI],ex0)); err=np.linalg.norm(P-tgt,axis=1)
print('rms px', np.sqrt((err**2).mean()), 'inner rms', np.sqrt((err[17:]**2).mean()), 'max', err.max(), np.argmax(err))
print('pose rotvec', x[:3], 'scale', x[3], 't', x[4:6]); print('id |a|', np.abs(x[6:6+NI]).max())
np.savez(W+'/fit.npz', pose=x[:6], a=x[6:6+NI], ex0=ex0, NI=NI)
