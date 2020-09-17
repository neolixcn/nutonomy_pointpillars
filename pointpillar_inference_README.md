# 本地inference操作步骤
## 1.远程连接服务器
```
ssh -p 8851 songhongli@36.112.101.228
```

## 2. 进入工程路径
```
 cd Projects/pointpillars2/second/pytorch/
```

## 3. 激活虚拟环境
```
source activate pointpillar
```

## 4. 添加PYTHONPATH路径
```
export PYTHONPATH=~/Projects/pointpillars2:$PYTHONPATH
```

## 5. 运行evaluate程序
```
python train.py evaluate --config_path=../configs/pointpillars/xyres_16_4cls.proto --model_dir=/nfs/nas/model/songhongli/ppbaidusecond_pretrained_3cls/
```


## 6. 运行的点云文件修改方法
```
vim /home/songhongli/Projects/pointpillars2/second/data/preprocess.py
```

line 329:

```
points = np.fromfile("/home/songhongli/000000.bin", dtype=np.float32, co    unt=-1).reshape([-1, num_point_features])
```
将这里的fromfile打开的文件路径修改为要运行的一帧点云文件
