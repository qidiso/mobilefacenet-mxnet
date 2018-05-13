
trainning for a week long time .now i get better result.


lr-batch-epoch: 0.001 7999 0
testing verification..
(12000, 128)
infer time 12.323731
[lfw][8000]XNorm: 11.118196
[lfw][8000]Accuracy-Flip: 0.99583+-0.00375
testing verification..
(14000, 128)
infer time 14.580451
[cfp_fp][8000]XNorm: 9.335661
[cfp_fp][8000]Accuracy-Flip: 0.88786+-0.01615
testing verification..
(12000, 128)
infer time 12.362448
[agedb_30][8000]XNorm: 11.044563
[agedb_30][8000]Accuracy-Flip: 0.96083+-0.00827
saving 4
INFO:root:Saved checkpoint to "../models/MobileFaceNet/model-y1-arcface-0004.params"
[8000]Accuracy-Highest: 0.96133

or

infer time 12.754381
[lfw][46000]XNorm: 11.113467
[lfw][46000]Accuracy-Flip: 0.99550+-0.00395
testing verification..
(14000, 128)
infer time 14.228613
[cfp_fp][46000]XNorm: 9.320184
[cfp_fp][46000]Accuracy-Flip: 0.89257+-0.01589
testing verification..
(12000, 128)
infer time 12.060387
[agedb_30][46000]XNorm: 11.055198
[agedb_30][46000]Accuracy-Flip: 0.96117+-0.00746
saving 23

## 前言

本文主要记录下复现mobilefacenet的流程，参考mobilefacenet作者月生给的基本流程，基于insightface的4月27日
```
4bc813215a4603474c840c85fa2113f5354c7180
```
版本代码在P40单显卡训练调试。

## 训练步骤
1.拉取配置[insightface](https://github.com/deepinsight/insightface)工程的基础环境；

2.softmax loss初调：lr0.1，softmax的fc7配置wd_mult=10.0和no_bias=True,训练4万步;


切换到src目录下，修改train_softmax.py：
179-182行：
```
  if args.loss_type==0: #softmax
    _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
```
修改为：

```
  if args.loss_type==0: #softmax
    #_bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    # fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias=True, num_hidden=args.num_classes, name='fc7')
```

363行：

```
 if args.network[0]=='r' or args.network[0]=='y':
```
修改为：

```
 if args.network[0]=='r' :
```
这样保证uniform初始化；


运行：
```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 0 --per-batch-size 512 --emb-size 128 --fc7-wd-mult 10  --data-dir  ../datasets/faces_ms1m_112x112  --prefix ../models/MobileFaceNet/model-y1-softmax
```
 

3.arcface loss调试：s=64, m=0.5, 起始lr=0.1，在[80000, 120000, 140000, 160000]步处降低lr，总共训练16万步。这时，LFW acc能到0.9955左右，agedb-30 acc能到0.959以上。

切换到src目录下：

```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr-steps 80000,120000,140000,160000 --emb-size 128 --per-batch-size 512 --data-dir ../datasets/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-softmax,20 --prefix ../models/MobileFaceNet/model-y1-arcface
```

4.agedb精调：从3步训练好的模型继续用arcface loss训练，s=128, m=0.5，起始lr=0.001，在[20000, 30000, 40000]步降低lr，这时能得到lfw acc 0.9955左右，agedb-30 acc 0.96左右的最终模型。

```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr 0.001 --lr-steps 20000,30000,40000 --emb-size 128 --per-batch-size 512 --margin-s 128 --data-dir ../datasets/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-arcface,80 --prefix ../models/MobileFaceNet/model-y1-arcface
```

