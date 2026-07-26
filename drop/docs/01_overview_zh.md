# CDWM Drop 世界模型 · 总览（中文）

English counterpart: `00_overview.md`. 本文是给同事的一份**自洽**中文说明（要点 + 示意图 + 数据表），涵盖：相比 HuggingFace v1 数据集的改动、模型设计、主要结果、以及我们排除的方案。

一句话：接触动力学世界模型的 **Drop 任务** —— 物体在桌上方**释放 → 自由下落 → 撞击 → 落定**，模型预测**最终停在哪个稳定姿态 / basin**。是 Grasp 世界模型（WM#1）的姊妹任务，共用同一套物体高斯簇点云。

---

## 1. 相比 HuggingFace v1 数据集，我们做了什么（核心）

### 1.1 数据：80 物体语料 → 我们自建的 88 物体 Gate B 集
- **相似**：同一套 grasp 物体点云作条件；任务形式一致（释放→下落→撞击→落定）。
- **不同**：
  - **物体集**：语料 **80 唯一物体**（108 物体×配置；均匀密度、自然释放）；我们 **90 生成 / 88 使用**（排除 2 个 hammer 变体），从 ~171 grasp 宇宙里挑 **CoM 敏感物体**。**两套只重叠 17 个**（非 hammer 口径 16），73 个是我们独有 → Gate B **不是**语料子集。
  - **质心 CoM**：语料**均匀密度 → 质心固定在几何中心**（无多样性）；我们**每个 episode 采一个不同的隐藏质心**（沿主轴偏移 ±0.35·半轴，质量/惯量不变、只挪质心）→ 有质心多样性可研究。这直接回应"语料假设均匀密度、没有 CoG 多样性"。
  - **释放方式**：语料自然释放（小倾角、形状主导、多数落回）；我们**近边界释放**（从稳定姿态朝"两稳定姿态之间的临界脊"倾 15–55°，此处往哪倒最不确定）。
  - ⚠️ Drop **没有夹爪**，"释放"=松手丢下；"近边界"指释放**朝向**接近翻倒临界点，与抓取无关。

### 1.2 模型：单点端点 → 扩散世界模型"想象"落定分布
- **喂初始条件**（点云 + 释放参数），不喂轨迹步；**预测 K=8 个"接触→落定"姿态锚点**。
- **自由下落被跳过**（零初速释放→朝向不变、无学习信号）；演示里的下落用确定性弹道补上并标注。
- **预测几个未来 = 旋钮 S**：扩散给的是"未来分布"，采样几次就有几个未来；这是它 vs 确定性模型的根本区别。

### 1.3 几何：点云单凸包 → mesh 的 CoACD 端到端
- 世界模型输入一直是**点云**；但**物理引擎只能碰凸几何**，裸点云不能直接仿真。
- 早期 Gate B 用点云**单凸包**（丢凹形，一个 scope 限制）；后来改成 mesh 的 **CoACD 凸分解** + 输入点云从**同一 mesh 表面重采样**（几何一致），证明机制在忠实几何下仍成立。

---

## 2. 设计细节（配图）

### 图 1 · 一条轨迹（模型只预测后半段）
```
 释放          自由下落（跳过·确定性弹道）        接触 k0        翻滚落定（K=8 等间隔锚点）       静止
  |───────────────────────────────────────────|═══════════════════════════════════════|
 t=0          朝向不变·高度=闭式弹道·无信息         ↑冲击       a0 a1 a2 a3 a4 a5 a6 a7
                                                          ↑                            ↑
                                                     接触起始(k0)                   最终静止(=端点)
  模型不预测 ◄─────────────────────────────────►   模型预测：K=8 姿态 ◄──────────────────►
```
- **K=8 不是 8 个物理事件**；是"接触起始 k0 → 静止"之间**按时间等间隔**的 8 个姿态快照（`linspace(k0, n-1, 8)`）。只有 a0（接触起始/冲击）、a7（最终静止）是事件；中间 6 个是翻滚过程的均匀采样。

### 图 2 · 扩散世界模型：从初始条件"想象"落定分布
```
  输入（只有初始条件）             扩散去噪（25 步 DDIM）              输出
  点云 + 释放参数 ──cond──►  纯噪声 ▓▒░→→→→→→→→░▒▓ 干净 ──► 8 姿态轨迹 a0..a7
                            （25 步·噪声表等间隔·整条一次性联合去噪）

  采样 S 次（不同噪声）⇒ S 个想象未来：future1 往左倒 / future2 落回 / … / futureS 往右倒
```
- **两种"步"别混**：轨迹姿态 **8 个**（时间等间隔）vs 扩散去噪 **25 步 DDIM**（噪声表等间隔；训练 1000 步 DDPM）。实际用过 S=4（指标）/ 8（探针）/ 16（演示）。

### 图 3 · 数据：模型输入=点云；物理碰撞几何=mesh 的 CoACD
```
  真实物体 ──观测──► 点云(准确) ──► 世界模型输入
                       │ (裸点云不能直接碰撞)
                       ├─► 单凸包  → 丢凹形（早期 Gate B 限制）
                       └─► CoACD(需mesh) → 一组凸块·忠实凹形（Item 1）
                            └ 只有点云的物体：点云→重建mesh→CoACD（可行但更噪·后续）
```

---

## 3. 主要结果（都显著、通过稳健性检验、Codex 审核）
- **分布式 Gate B**：分布 ≫ 单点（NLL +0.53 显著）；隐藏质心**有因果作用**（三种独立方式）；扩散模型**校准好**（ECE 0.015 vs 单点 0.227）。
- **跨物体迁移**：几何锚定的质心 latent 能迁移到**没见过的物体**（NLL 0.495 vs no_latent 0.679）；抽象编码不迁移。
- **★ Item 1（CoACD 端到端几何）**：忠实几何下机制**依然成立、甚至更强**——分布>单点 **+0.355** 显著(7/7)；grounded>no_latent **+0.245** 显著(7/7，比单凸包 +0.184 更大)；grounded 显著优于 abstract(+0.212) 和错误质心对照 shuffled(+0.236) → 要的是**正确的几何锚定质心**，不是多加通道。
- **★ Item 2（从观测推质心，去掉 oracle）**：用冻结世界模型当似然做贝叶斯推断，观测越多越准——NLL 0.678→0.566；配对增益 k=0→8 **+0.111** 显著(17/19)；后验随观测收敛（熵 3.39→2.82）；边缘化优于单点、更好校准；无泄漏。

## 4. 我们排除的方案（有信息量的负结果）
- **学习式摊销编码器**（一次前向直接从观测预测质心）：能学到有用的几何先验，但**随观测不自适应**（k=0→8 增益 −0.003，不显著）→ 这个逆问题不易摊销，**解析式贝叶斯才是有效方法**（反而强化了 Item 2 的贡献 = "自适应"）。

## 5. 名词 & 里程碑
- **Gate 里程碑**（分阶段代号，非物体子集）：**Gate A**（CoM 多样性值不值得做）→ **A.5**（4 个压力测试）→ **B**（建质心感知的*分布式* drop WM：近边界 + 隐藏质心 + 扩散头预测 basin 分布）。
- **CoACD** = 近似凸分解（把凹 mesh 拆成凸块，让刚体仿真忠实碰撞凹形）。
- **basin** = 一个稳定静止姿态（落定模式）。

---

## 附录 · Gate B 88 物体清单
`episodes`=近边界 episode 数；`basins`=落定过的不同稳定姿态数（多=更多模态）；`在语料?`=是否也在同事 80 物体语料里（共 16）；`有mesh?`=是否可做 CoACD（共 28）。

| # | 物体 object | episodes | basins | 在语料? | 有mesh(CoACD)? |
|---|---|---|---|---|---|
| 1 | 006_mustard_bottle | 651 | 6 |  | 是 |
| 2 | 009_gelatin_box | 1985 | 6 |  |  |
| 3 | 010_potted_meat_can | 1976 | 7 | 是 |  |
| 4 | 011_banana | 1442 | 3 |  |  |
| 5 | 021_bleach_cleanser | 1754 | 7 |  | 是 |
| 6 | 022_windex_bottle | 819 | 7 |  | 是 |
| 7 | 026_sponge | 1964 | 2 |  |  |
| 8 | 028_skillet_lid | 1169 | 5 |  | 是 |
| 9 | 029_plate | 1578 | 2 |  | 是 |
| 10 | 031_spoon | 1030 | 3 |  |  |
| 11 | 033_spatula | 587 | 3 |  |  |
| 12 | 038_padlock | 993 | 7 |  |  |
| 13 | 050_medium_clamp | 1585 | 4 |  |  |
| 14 | 051_large_clamp | 1404 | 2 | 是 | 是 |
| 15 | 072-b_toy_airplane | 962 | 2 |  |  |
| 16 | 3D_Dollhouse_Swing | 1989 | 5 |  |  |
| 17 | Android_Lego | 1857 | 3 |  |  |
| 18 | Beetle_Adventure_Racing_Nintendo_64 | 1818 | 4 |  |  |
| 19 | Blue_Jasmine_Includes_Digital_Copy_UltraViolet_DVD | 1983 | 2 |  |  |
| 20 | Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl | 1536 | 3 |  |  |
| 21 | Brother_LC_1053PKS_Ink_Cartridge_CyanMagentaYellow_1pack | 1252 | 6 |  | 是 |
| 22 | Calphalon_Kitchen_Essentials_12_Cast_Iron_Fry_Pan_Black | 1229 | 6 |  |  |
| 23 | Canon_Ink_Cartridge_Green_6 | 1769 | 5 |  | 是 |
| 24 | Canon_Pixma_Chromalife_100_Magenta_8 | 1925 | 3 |  |  |
| 25 | Canon_Pixma_Ink_Cartridge_8_Green | 2000 | 2 |  |  |
| 26 | Canon_Pixma_Ink_Cartridge_8_Red | 1426 | 3 |  |  |
| 27 | Chef_Style_Round_Cake_Pan_9_inch_pan | 1998 | 2 |  |  |
| 28 | Chefmate_8_Frypan | 814 | 3 |  |  |
| 29 | Cole_Hardware_Deep_Bowl_Good_Earth_1075 | 1148 | 7 |  |  |
| 30 | Cole_Hardware_Plant_Saucer_Brown_125 | 1868 | 2 |  |  |
| 31 | Cole_Hardware_Plant_Saucer_Glazed_9 | 1652 | 2 |  |  |
| 32 | Cole_Hardware_Saucer_Electric | 1891 | 2 |  |  |
| 33 | Cole_Hardware_Saucer_Glazed_6 | 1715 | 2 |  |  |
| 34 | Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue | 1999 | 2 |  |  |
| 35 | Crayola_Model_Magic_Modeling_Material_White_3_oz | 1660 | 5 |  | 是 |
| 36 | Dell_Ink_Cartridge_Yellow_31 | 1864 | 3 |  | 是 |
| 37 | Diamond_Visions_Scissors_Red | 1350 | 3 |  |  |
| 38 | Ecoforms_Plant_Bowl_Atlas_Low | 1410 | 8 |  |  |
| 39 | Ecoforms_Plant_Bowl_Turquoise_7 | 898 | 8 |  |  |
| 40 | Ecoforms_Plant_Container_FB6_Tur | 1371 | 9 |  |  |
| 41 | Ecoforms_Plant_Container_S14Turquoise | 1264 | 2 |  |  |
| 42 | Ecoforms_Plant_Container_S24NATURAL | 1905 | 2 |  |  |
| 43 | Ecoforms_Plant_Container_S24Turquoise | 1893 | 2 |  |  |
| 44 | Ecoforms_Plant_Plate_S11Turquoise | 1812 | 2 |  |  |
| 45 | Ecoforms_Planter_Bowl_Cole_Hardware | 855 | 5 |  |  |
| 46 | Epson_273XL_Ink_Cartridge_Magenta | 1739 | 5 |  |  |
| 47 | Epson_Ink_Cartridge_126_Yellow | 1849 | 5 |  |  |
| 48 | Epson_LabelWorks_LC4WBN9_Tape_reel_labels_047_x_295_Roll_Black_on_White | 1083 | 6 |  |  |
| 49 | Footed_Bowl_Sand | 1365 | 10 |  |  |
| 50 | Fujifilm_instax_SHARE_SP1_10_photos | 1825 | 3 |  |  |
| 51 | Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3 | 1618 | 6 |  | 是 |
| 52 | Granimals_20_Wooden_ABC_Blocks_Wagon | 1963 | 5 | 是 | 是 |
| 53 | Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI | 1800 | 8 | 是 | 是 |
| 54 | Great_Dinos_Triceratops_Toy | 1783 | 10 |  |  |
| 55 | Hasbro_Monopoly_Hotels_Game | 1924 | 6 |  |  |
| 56 | Marc_Anthony_Skip_Professional_Oil_of_Morocco_Conditioner_with_Argan_Oil | 1570 | 3 |  | 是 |
| 57 | Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle | 1999 | 7 |  | 是 |
| 58 | Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion | 950 | 6 |  | 是 |
| 59 | Markings_Letter_Holder | 1960 | 6 |  |  |
| 60 | Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister | 1167 | 7 |  |  |
| 61 | Nintendo_Mario_Action_Figure | 1917 | 7 | 是 | 是 |
| 62 | Nintendo_Yoshi_Action_Figure | 1927 | 5 | 是 | 是 |
| 63 | Now_Designs_Bowl_Akita_Black | 1365 | 6 |  |  |
| 64 | OXO_Cookie_Spatula | 214 | 2 |  |  |
| 65 | OXO_Soft_Works_Can_Opener_SnapLock | 1924 | 8 | 是 | 是 |
| 66 | Razer_Naga_MMO_Gaming_Mouse | 416 | 4 | 是 | 是 |
| 67 | Razer_Taipan_Black_Ambidextrous_Gaming_Mouse | 1183 | 2 |  |  |
| 68 | Remington_1_12_inch_Hair_Straightener | 1041 | 6 | 是 | 是 |
| 69 | Remington_TStudio_Hair_Dryer | 1249 | 2 |  |  |
| 70 | Remington_TStudio_Silk_Ceramic_Hair_Straightener_2_Inch_Floating_Plates | 1072 | 4 |  |  |
| 71 | Room_Essentials_Bowl_Turquiose | 1571 | 5 |  |  |
| 72 | Room_Essentials_Mug_White_Yellow | 1150 | 6 |  |  |
| 73 | Schleich_African_Black_Rhino | 1939 | 4 | 是 | 是 |
| 74 | Schleich_Allosaurus | 1651 | 4 | 是 | 是 |
| 75 | Schleich_Hereford_Bull | 1932 | 3 | 是 | 是 |
| 76 | Schleich_Lion_Action_Figure | 1287 | 5 |  |  |
| 77 | Schleich_S_Bayala_Unicorn_70432 | 1818 | 4 |  |  |
| 78 | Schleich_Spinosaurus_Action_Figure | 2000 | 5 | 是 | 是 |
| 79 | Schleich_Therizinosaurus_ln9cruulPqc | 1720 | 4 | 是 | 是 |
| 80 | SCHOOL_BUS | 1831 | 5 | 是 | 是 |
| 81 | Seagate_1TB_Wireless_Plus_mobile_device_storage | 2000 | 2 |  |  |
| 82 | Smith_Hawken_Woven_BasketTray_Organizer_with_3_Compartments_95_x_9_x_13 | 1982 | 5 |  |  |
| 83 | Super_Mario_3D_World_Deluxe_Set_yThuvW9vZed | 1164 | 2 |  |  |
| 84 | Thomas_Friends_Woodan_Railway_Henry | 1813 | 5 | 是 | 是 |
| 85 | Threshold_Porcelain_Pitcher_White | 365 | 6 |  |  |
| 86 | Threshold_Porcelain_Spoon_Rest_White | 1699 | 7 |  |  |
| 87 | Utana_5_Porcelain_Ramekin_Large | 1992 | 4 |  |  |
| 88 | WHALE_WHISTLE_6PCS_SET | 1709 | 3 |  |  |

_合计 88 物体：16 个也在语料、28 个有 mesh（可 CoACD）；共 134,576 episodes。_
