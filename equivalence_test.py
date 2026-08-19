"""
Equivalence test: does depth_forward_subbatch at D=3 reproduce the notebook's
hardcoded 3-block eggroll_forward_subbatch, given identical weights + factors?

If YES to ~1e-4, the depth-generic path is safe: D=3 == your 91% baseline, and
deeper D just adds blocks by the same rule. This is the go/no-go check before
spending pod-hours.
"""
import torch, torch.nn as nn, torch.nn.functional as F, snntorch as snn
torch.manual_seed(0)
DEVICE="cpu"

# ---- config matching N-MNIST 3-block ----
C,CONV1_OUT,CONV2_OUT,CONV3_OUT=2,32,64,128
CONV1_K=CONV2_K=CONV3_K=3; CONV1_PAD=CONV2_PAD=CONV3_PAD=1
POOL_K=2; NUM_STEPS=4; NUM_CLASSES=10; BETA=0.95; H=W=34
H2,W2=H//2,W//2; H3,W3=H2//2,W2//2; H4,W4=H3//2,W3//2
FC_IN=CONV3_OUT*H4*W4
BASE_STD=0.3
INIT_STD_CONV1=BASE_STD/(C*CONV1_K*CONV1_K)**0.5
INIT_STD_CONV2=BASE_STD/(CONV1_OUT*CONV2_K*CONV2_K)**0.5
INIT_STD_CONV3=BASE_STD/(CONV2_OUT*CONV3_K*CONV3_K)**0.5
INIT_STD_FC=BASE_STD/FC_IN**0.5
def clamp_events(x): return x.clamp(0,1)

# ---- helpers (verbatim from notebook) ----
def eggroll_conv2d_cached_input(x_t_shared, patches, base_conv, A, B_lr, rank, sigma, P_sub):
    batch=x_t_shared.size(0); out_ch=base_conv.out_channels; scale=sigma/(rank**0.5)
    base_out=base_conv(x_t_shared); H_out,W_out=base_out.shape[-2:]
    patches_4d=patches.unsqueeze(0).expand(P_sub,-1,-1,-1)
    Bp=torch.einsum("pir,pbil->pbrl",B_lr,patches_4d); ABp=torch.einsum("por,pbrl->pbol",A,Bp)
    corr=(scale*ABp).reshape(P_sub*batch,out_ch,H_out,W_out)
    base_out=base_out.unsqueeze(0).expand(P_sub,-1,-1,-1,-1).reshape(P_sub*batch,out_ch,H_out,W_out)
    return base_out+corr
def eggroll_conv2d(x, base_conv, A, B_lr, rank, sigma, P_sub, batch):
    out_ch=base_conv.out_channels; in_ch=base_conv.in_channels
    k=base_conv.kernel_size[0]; pad=base_conv.padding[0]; scale=sigma/(rank**0.5)
    base_out=base_conv(x); H_out,W_out=base_out.shape[-2:]
    patches=F.unfold(x,kernel_size=k,padding=pad); patches_3d=patches.view(P_sub,batch,in_ch*k*k,patches.shape[-1])
    Bp=torch.einsum("pir,pbil->pbrl",B_lr,patches_3d); ABp=torch.einsum("por,pbrl->pbol",A,Bp)
    return base_out+(ABp*scale).reshape(P_sub*batch,out_ch,H_out,W_out)

# ---- the notebook's hardcoded 3-block forward (verbatim logic) ----
def hardcoded_forward(data, net, factors, P_sub, rank, sigma, patches_per_t):
    (A1,B1,c1),(A2,B2,c2),(A3,B3,c3),(A4,B4,c4)=factors
    batch=data.size(0); db=batch//P_sub
    s1,s2,s3,s4=sigma*INIT_STD_CONV1,sigma*INIT_STD_CONV2,sigma*INIT_STD_CONV3,sigma*INIT_STD_FC
    scale4=s4/(rank**0.5)
    mem1=torch.zeros(batch,CONV1_OUT,H,W); mem2=torch.zeros(batch,CONV2_OUT,H2,W2)
    mem3=torch.zeros(batch,CONV3_OUT,H3,W3); mem_out=torch.zeros(batch,NUM_CLASSES); out_acc=torch.zeros(batch,NUM_CLASSES)
    bc1=(s1/(CONV1_OUT**0.5)*c1).view(P_sub,1,CONV1_OUT,1,1)
    bc2=(s2/(CONV2_OUT**0.5)*c2).view(P_sub,1,CONV2_OUT,1,1)
    bc3=(s3/(CONV3_OUT**0.5)*c3).view(P_sub,1,CONV3_OUT,1,1)
    bc4=(s4/(NUM_CLASSES**0.5)*c4).view(P_sub,1,NUM_CLASSES)
    for t in range(NUM_STEPS):
        x_t=clamp_events(data[:,t]); x_ts=x_t[:db]
        r1=eggroll_conv2d_cached_input(x_ts,patches_per_t[t],net.conv1,A1,B1,rank,s1,P_sub)
        cur1=(r1.view(P_sub,db,CONV1_OUT,H,W)+bc1).reshape(P_sub*db,CONV1_OUT,H,W)
        cur1=net.bn1[t](cur1); spk1,mem1=net.lif1(cur1.float(),mem1); spk1=net.pool1(spk1)
        r2=eggroll_conv2d(spk1,net.conv2,A2,B2,rank,s2,P_sub,db)
        cur2=(r2.view(P_sub,db,CONV2_OUT,H2,W2)+bc2).reshape(P_sub*db,CONV2_OUT,H2,W2)
        cur2=net.bn2[t](cur2); spk2,mem2=net.lif2(cur2.float(),mem2); spk2=net.pool2(spk2)
        r3=eggroll_conv2d(spk2,net.conv3,A3,B3,rank,s3,P_sub,db)
        cur3=(r3.view(P_sub,db,CONV3_OUT,H3,W3)+bc3).reshape(P_sub*db,CONV3_OUT,H3,W3)
        cur3=net.bn3[t](cur3); spk3,mem3=net.lif3(cur3.float(),mem3); spk3=net.pool3(spk3)
        flat=spk3.flatten(1); flat3=flat.view(P_sub,db,FC_IN)
        base4=flat@net.fc_out.weight.T+net.fc_out.bias
        Bp4=torch.einsum("pir,pbi->pbr",B4,flat3); ABp4=torch.einsum("por,pbr->pbo",A4,Bp4)
        cur4=base4+(scale4*ABp4+bc4).reshape(P_sub*db,NUM_CLASSES)
        mem_out=BETA*mem_out+cur4.float(); out_acc+=mem_out
    return out_acc/NUM_STEPS

# ---- a 3-block net that BOTH paths share ----
class Net3(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv1=nn.Conv2d(C,CONV1_OUT,3,padding=1); s.conv2=nn.Conv2d(CONV1_OUT,CONV2_OUT,3,padding=1)
        s.conv3=nn.Conv2d(CONV2_OUT,CONV3_OUT,3,padding=1)
        s.bn1=nn.ModuleList([nn.BatchNorm2d(CONV1_OUT,affine=False) for _ in range(NUM_STEPS)])
        s.bn2=nn.ModuleList([nn.BatchNorm2d(CONV2_OUT,affine=False) for _ in range(NUM_STEPS)])
        s.bn3=nn.ModuleList([nn.BatchNorm2d(CONV3_OUT,affine=False) for _ in range(NUM_STEPS)])
        s.lif1=snn.Leaky(beta=BETA); s.lif2=snn.Leaky(beta=BETA); s.lif3=snn.Leaky(beta=BETA)
        s.pool1=nn.AvgPool2d(2); s.pool2=nn.AvgPool2d(2); s.pool3=nn.AvgPool2d(2)
        s.fc_out=nn.Linear(FC_IN,NUM_CLASSES)
net=Net3().train()

# ---- import the depth-generic path and adapt the shared net into a DepthCSNN-like shim ----
import importlib.util
spec=importlib.util.spec_from_file_location("de","depth_eggroll.py"); de=importlib.util.module_from_spec(spec); spec.loader.exec_module(de)

# Build a DepthCSNN with D=3 and COPY the shared net's weights into it so both see identical params
dnet=de.DepthCSNN(3, in_ch=C, in_hw=H, num_classes=NUM_CLASSES, num_steps=NUM_STEPS, beta=BETA)
with torch.no_grad():
    for i,src in enumerate([net.conv1,net.conv2,net.conv3]):
        dnet.convs[i].weight.copy_(src.weight); dnet.convs[i].bias.copy_(src.bias)
    dnet.fc_out.weight.copy_(net.fc_out.weight); dnet.fc_out.bias.copy_(net.fc_out.bias)
    # sync BN (affine=False has no params, running stats start identical) and LIF
    for i,(a,b) in enumerate(zip([net.bn1,net.bn2,net.bn3],[dnet.bns[0],dnet.bns[1],dnet.bns[2]])):
        pass  # affine=False -> nothing to copy; running stats identical at init
dnet.train()

# identical factors + data
pop,rank,P_sub,db=4,8,4,2
fac_full=de.make_all_factors_depth(123,pop,rank,dnet)
fac_sub=[(A[:P_sub],B[:P_sub],c[:P_sub]) for (A,B,c) in fac_full]
# build matching 3-tuple factors for the hardcoded path from the SAME tensors
hc_factors=tuple((A[:P_sub],B[:P_sub],c[:P_sub]) for (A,B,c) in fac_full[:3]) + ((fac_full[3][0][:P_sub],fac_full[3][1][:P_sub],fac_full[3][2][:P_sub]),)
data=torch.rand(P_sub*db,NUM_STEPS,C,H,W)
patches=[F.unfold(clamp_events(data[:db,t]),kernel_size=3,padding=1) for t in range(NUM_STEPS)]

out_hc=hardcoded_forward(data,net,hc_factors,P_sub,rank,0.05,patches)
out_dg=de.depth_forward_subbatch(data,dnet,fac_sub,P_sub,rank,0.05,NUM_STEPS,BETA)
diff=(out_hc-out_dg).abs().max().item()
print(f"max abs diff between hardcoded-3block and depth-generic-D3: {diff:.2e}")
print("hardcoded[0,:3] =",out_hc[0,:3].tolist())
print("depthgen [0,:3] =",out_dg[0,:3].tolist())
if diff < 1e-4:
    print("EQUIVALENCE PASS -- depth-generic D=3 reproduces the hardcoded 3-block path")
else:
    print("MISMATCH -- do NOT run the grid until this is reconciled")
