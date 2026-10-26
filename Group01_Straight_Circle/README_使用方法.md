# Group 01：N=5000 纯直线粒子 + 圆形边界

## 目的：看现象（先不做复杂理论）

寻找并区分：局部朝向一致、移动团簇、团簇合并/破裂、镜面反射后重新排列、边缘富集、局部涡旋与全局环流、周期性变化、长期动态重排。

**没有任何一种现象被预设为必然出现；所有截图都是确定模拟时间的截面，不能仅凭动画声称达到稳态。**

## 目录

```
Group01_Straight_Circle/
  code/
    model.py          # 粒子运动+圆形边界处理，沿用通过物理校核的底层
    run.py            # 保存/读取轨迹，断点续算，检测数据
    diagnostics.py    # P、Q、E、C等观测量
    group01.py        # 第一组唯一的模拟入口
    plot_group.py     # 统一科研绘图，完全不改物理模型
    visual_style.py   # 统一浅色动画风格，后续各组共用
    test_group01.py   # 圆形镜面、切向滑移、软边界单元测试
  results/
    mirror/            # 镜面反射数据，含四种K的实际样本 t=60
    sliding/           # 切向滑移（待正式模拟）
    soft/              # 软边界（待正式模拟）
  01_查看已跑结果.bat
  02_测试代码.bat
  03_重新运行镜面组.bat
  04_切向滑移组.bat
  05_软边界组.bat
```

## 模型

- 粒子数 N=5000，无固有手性（Ω_i=0）。
- 圆域半径 R=50/√π≈28.2095，面积2500，面密度2。
- 速度 v0=1；邻域距离 r_a=1；无粒子间机械排斥、无附加随机角噪声。
- 角度演化：dθ_i/dt = K Σ_{j: |x_j-x_i|<r_a, j≠i} sin(θ_j-θ_i)/(1+邻居数量)。
- 默认时间步 dt 上限0.02，必要时自动缩小。
- 对比K=0、0.5、1.5、3，使用同一随机种子20261022和相同初始粒子位置/朝向。
- 对于`mirror`，粒子沿射线运动至圆壁，在实际接触点镜面反射，剩余步长继续移动；不是越界后截断。
- 对于`sliding`，碰壁时去除**实际位移速度**的外法向分量、朝向 θ 不强制改成切向，沿壁面受约束运动；粒子可在朝向重新指向内部时离开壁面。**它不是永久吸附，也不是强制旋转的墙力。**
- 对于`soft`，壁面内侧 w 宽的圆环区域施加径向弹簧式斥力：F_w=-k_w max(r-(R-w),0) r_hat；壁面位置 r=R 是可视化参考线。**软边界不是不可穿透的硬墙；粒子可能短暂越过标线**，尤其当参数较弱。后续要检查越界比例并单独论证其物理合理性。

## Windows直接看结果

下载完整压缩包，右键**全部解压**，双击`01_查看已跑结果.bat`。也可以直接打开`results/mirror/index.html`。

## Windows重新模拟

推荐 Python 3.11/3.12：安装时勾选 Add Python to PATH。在压缩包解压后**不要直接在ZIP里双击bat**。

进入 `code` 目录，在命令行依次执行：

```bat
python -m pip install -r requirements.txt
python test_group01.py
python group01.py --boundary mirror --N 5000 --K 0 0.5 1.5 3 --end 60 --root ../my_runs --plot --movies
```

打开`my_runs/mirror/index.html`，里面有全部对照视频、各组曲线、朝向着色、径向密度和少量轨迹。

其他圆壁规则：

```bat
python group01.py --boundary sliding --K 0 0.5 1.5 3 --end 60 --root ../my_runs --plot --movies
python group01.py --boundary soft --K 0 0.5 1.5 3 --end 60 --wall-width 2 --wall-k 4 --root ../my_runs --plot --movies
```

可以先用 `--N 100 --end 5` 测试电脑能否运行。但正式实验保持N=5000，缩小N时的现象不能充当正式结论。

只重新绘图：

```bat
python plot_group.py --root ../results/mirror --movies
```

已经算至 t=60，要继续至 t=300：

```bat
python group01.py --boundary mirror --N 5000 --K 0 1.5 --end 300 --root ../results --resume --plot --movies
```

**提醒：**如果有已存在结果，程序默认拒绝覆盖。`--resume`仅用于将相同模型、参数、初态的结果续算到更大的`--end`；不同K、dt、边界参数要另建文件夹。高强度对齐下计算开销可能增加。

## 结果文件如何看

- `particles.mp4`：纯粒子动画，画面无箭头无尾迹。
- `heading.mp4`：粒子朝向着色，不把方向大小解释为速度大小。
- `snapshots.png`：全过程四个时刻的同尺度对照。
- `selected_trajectories.png`：代表粒子的连续位置，用于寻找碰壁、绕行；不是全部粒子的轨迹。
- `radial_density.png`：以环带面积归一化的相对径向密度，均匀情况下约为1。
- `observables.png`：P全局极化、Q局部对齐、E边界富集、C有符号朝向环流。
- `observables.csv`：每个观测时刻的多项诊断量。
- `history.npz`：真实数值轨迹序列（从t=0开始）、少量额外的高时间分辨率轨迹。
- `checkpoint.npz`：断点续算数据。

圆形镜面反射的单粒子轨迹可以高度规则，**这本身不是涡旋或自发有序的证据**。真正判断集体环流要结合方向场、局部协同和长时间统计。即使环流指标C约为0，也可能存在方向相反、彼此抵消的局部旋转结构。

### 关于首次结果时间尺度

R约28.21，横穿圆直径的速度1飞行时间约56.42，所以t=60仅约一次穿越圆直径的时间；本包结果的状态字段都是`unconfirmed`，不等同于收敛稳态。

### 绘图风格来源

白色背景、深色圆壁、蓝色粒子点、周期朝向色图；参考AeroPlanck项目极简集群可视化的目标，但是**GitHub所指子目录的原始绘图代码没有成功读取，因此这不是作者原画法的精确复制**。
