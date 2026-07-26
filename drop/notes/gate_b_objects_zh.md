# Gate B 数据集说明 + 88 物体清单（中文）

## 一、近边界释放 & 隐藏质心（先澄清两个概念）

**⚠️ Drop 任务里没有夹爪（gripper）。** "释放"就是把物体在桌面上方**松手/丢下**，不是抓取。所以"近边界释放"跟夹爪抓哪一侧无关。

### 什么是"两个稳定姿态之间的边界，倾 15–55°"？
- 一个物体有多个**稳定静止姿态**（比如盒子可以躺 6 个面；瓶子可以立着或躺着）。
- 相邻两个稳定姿态之间，存在一个**不稳定的"边界/临界脊"**（翻倒的临界点，像硬币立在边上）。
- **近边界释放** = 从某个稳定姿态出发，把物体**朝这个临界脊方向旋转 15–55°**再松手（代码：`Rz(随机) · Rrotvec(15~55° 绕水平轴) · R_稳定姿态`）。
- 倾得越多（接近 55°）就越靠近临界点 → 落定往哪边**越不确定** → 这正是**隐藏质心决定往哪倒**的地方（多模态的来源）。语料的自然释放只倾很小，几乎都落回原位。

### "逐 episode 的隐藏质心偏移" = 比旧版更多的 CoM 变化？
**对，正是关键区别：**
- **旧版/语料**：均匀密度 → 每个物体只有**一个** CoM（几何中心）。没有质心多样性。
- **我们 Gate B**：**每个 episode 单独采样一个不同的隐藏质心**（沿某条主轴偏移，delta ∈ ±0.35·半轴长；质量/惯量固定，只变质心位置）。所以同一个物体跨 episode 有**很多种 CoM**，才能研究"质心如何决定落定 basin"。这直接回应同事说的"语料假设均匀密度、没有 CoG 多样性"。

## 二、两个物体宇宙（别混）+ Gate 里程碑

| | 语料 corpus（同事 HF 数据） | Gate B（我们生成） |
|---|---|---|
| 物体数 | **80 唯一物体**（108 个 物体×配置；153,996 settled） | **90 生成 / 88 使用**（排除 2 个 hammer 变体）|
| 质心 CoM | 均匀密度 → 几何中心（无多样性）| 逐 episode 隐藏质心偏移（有多样性）|
| 释放 | 自然释放（稳定姿态 + 小倾角，形状主导）| 近边界释放（倾 15–55° 朝临界脊）|
| 来源 | 6 核心配置 + 75 新纳入 | 从 ~171 grasp/WM#1 宇宙里挑 CoM 敏感物体 |
| 用于 | 端点基线 / traj_k8 / 语料扩散 rollout | Gate B / 跨物体迁移 / CoACD e2e / 从观测推质心 |

- **两套只重叠 17 个物体**（非 hammer 口径 16 个）：**73 个是 Gate-B 独有**，63 个语料独有。Gate B 不是语料的子集。
- Gate B 从 **10 个物体试点**扩到 **88**（"加了 70 来个"就是这次扩容）。
- **Gate 是分阶段里程碑代号**（不是物体子集）：**Gate A**（CoM 多样性值不值得做？）→ **Gate A.5**（4 个压力测试）→ **Gate B**（正式建质心感知的*分布式* drop 世界模型：近边界 + 隐藏质心 + 扩散头预测 basin 分布）。

## 三、88 物体清单
- `episodes` = 该物体的近边界 episode 数；`basins` = 落定过的不同稳定姿态数（多=更多模态）。
- `在语料?` = 是否也在同事的 80 物体语料里（共 16 个）；`有mesh?` = 是否有本地 mesh（可做 CoACD 端到端，共 28 个）。

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

_合计 88 物体：其中 16 个也在语料、28 个有 mesh（可 CoACD）；总 episode 数 134576。_
