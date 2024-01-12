# TIGER-Temporally-Improved-Graph-Entity-Linker

The implementation of our approach is based on the original codebase [BLINK](https://github.com/facebookresearch/BLINK) and [HeCo](https://github.com/liun-online/HeCo).<br>

<br><br>
<div align="center">
<img src="fig.png" width="800" />
</div>
<br><br>

In this work, XXX. Knowledge graphs evolve annually with new entities emerging, existing definitions being revised, and relationships between entities changing. These changes can lead to temporal degradation in models, a phenomenon where model performance diminishes over time, especially when handling downstream tasks like entity linking. Recent studies show that nodes with higher connectivity degrees tend to be more resistant to temporal degradation, so-called structural unfairness. To address this problem, we introduce XXX model. We employ graph contrastive learning to enhance performance for low-degree nodes, thereby alleviating the impact of temporal degradation. The idea is to combine graph contrastive learning with text-based information. Leverage the characteristics of temporal data to construct a cross-year contrastive mechanism, using newly added relationships in each year's data as positive samples and newly removed relationships as negative samples.

## Usage

Please follow the instructions next to reproduce our experiments, and to train a model with your own data.

### 1. Install the requirements

Creating a new environment (e.g. with `conda`) is recommended. Use `requirements.txt` to install the dependencies:

```
conda create -n tiger39 -y python=3.9 && conda activate tiger39
pip install -r requirements.txt
```

### 2. Download the data

| Download link                                                | Size |
| ------------------------------------------------------------ | ----------------- |
| [Our Dataset](https://drive.google.com/drive/folders/1DeHi-cvVOAdYFA4GljaBvpuG0wiYpgch?usp=sharing) | 3.12 GB            |
| [ZESHEL](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel) | 1.55 GB            |
| [WikiLinksNED](https://github.com/yasumasaonoe/ET4EL) | 1.1 GB             |

### 3. Reproduce the experiments

```
train.sh
```

|       Only Forward      |       |        |        |        |        |        |        |        |        |         |         |
|:-----------------------:|:-----:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|
|                         |       |    0   |    1   |    2   |    3   |    4   |    5   |    6   |    7   |    8    |    9    |
|            @1           | BLINK | 0.1333 | 0.1339 | 0.1339 | 0.1356 | 0.1350 | 0.1361 | 0.1349 | 0.1358 |  0.1346 |  0.1430 |
|                         | TIGER | 0.1380 | 0.1397 | 0.1401 | 0.1413 | 0.1417 | 0.1443 | 0.1445 | 0.1477 |  0.1558 |  0.1737 |
|                         | Boost | 3.50\% | 4.30\% | 4.64\% | 4.20\% | 4.99\% | 6.04\% | 7.14\% | 8.79\% | 15.79\% | 21.47\% |
|            @2           | BLINK | 0.1995 | 0.1999 | 0.2004 | 0.2022 | 0.2015 | 0.2028 | 0.2010 | 0.2016 |  0.1999 |  0.2115 |
|                         | TIGER | 0.2055 | 0.2076 | 0.2084 | 0.2093 | 0.2099 | 0.2136 | 0.2138 | 0.2164 |  0.2276 |  0.2530 |
|                         | Boost | 3.03\% | 3.82\% | 4.01\% | 3.48\% | 4.17\% | 5.37\% | 6.36\% | 7.34\% | 13.86\% | 19.62\% |
|            @4           | BLINK | 0.2795 | 0.2789 | 0.2804 | 0.2825 | 0.2810 | 0.2837 | 0.2803 | 0.2813 |  0.2786 |  0.2951 |
|                         | TIGER | 0.2867 | 0.2890 | 0.2901 | 0.2919 | 0.2914 | 0.2949 | 0.2948 | 0.2984 |  0.3106 |  0.3451 |
|                         | Boost | 2.59\% | 3.63\% | 3.43\% | 3.34\% | 3.68\% | 3.94\% | 5.17\% | 6.07\% | 11.49\% | 16.94\% |
|            @8           | BLINK | 0.3721 | 0.3717 | 0.3735 | 0.3753 | 0.3725 | 0.3745 | 0.3714 | 0.3712 |  0.3684 |  0.3881 |
|                         | TIGER | 0.3797 | 0.3822 | 0.3831 | 0.3849 | 0.3836 | 0.3895 | 0.3877 | 0.3900 |  0.4031 |  0.4400 |
|                         | Boost | 2.06\% | 2.82\% | 2.56\% | 2.57\% | 2.98\% | 4.01\% | 4.38\% | 5.06\% |  9.41\% | 13.37\% |
|           @16           | BLINK | 0.4718 | 0.4717 | 0.4725 | 0.4738 | 0.4707 | 0.4729 | 0.4693 | 0.4680 |  0.4663 |  0.4850 |
|                         | TIGER | 0.4796 | 0.4823 | 0.4825 | 0.4833 | 0.4824 | 0.4873 | 0.4860 | 0.4864 |  0.5004 |  0.5377 |
|                         | Boost | 1.66\% | 2.25\% | 2.12\% | 2.02\% | 2.47\% | 3.04\% | 3.56\% | 3.92\% |  7.31\% | 10.87\% |
|           @32           | BLINK | 0.5749 | 0.5746 | 0.5742 | 0.5751 | 0.5716 | 0.5738 | 0.5695 | 0.5679 |  0.5655 |  0.5854 |
|                         | TIGER | 0.5826 | 0.5845 | 0.5845 | 0.5848 | 0.5835 | 0.5875 | 0.5854 | 0.5855 |  0.5976 |  0.6326 |
|                         | Boost | 1.34\% | 1.73\% | 1.79\% | 1.69\% | 2.08\% | 2.40\% | 2.78\% | 3.09\% |  5.68\% |  8.06\% |
|           @64           | BLINK | 0.6742 | 0.6739 | 0.6735 | 0.6743 | 0.6712 | 0.6734 | 0.6698 | 0.6671 |  0.6652 |  0.6833 |
|                         | TIGER | 0.6817 | 0.6828 | 0.6824 | 0.6827 | 0.6820 | 0.6858 | 0.6821 | 0.6802 |  0.6911 |  0.7224 |
|                         | Boost | 1.11\% | 1.32\% | 1.33\% | 1.24\% | 1.60\% | 1.84\% | 1.84\% | 1.97\% |  3.89\% |  5.72\% |
| Forward and Backward    |       |        |        |        |        |        |        |        |        |         |         |
|            @1           | BLINK | 0.1333 | 0.1326 | 0.1331 | 0.1333 | 0.1330 | 0.1326 | 0.1322 | 0.1306 |  0.1306 |  0.1349 |
|                         | TIGER | 0.1380 | 0.1368 | 0.1370 | 0.1375 | 0.1377 | 0.1382 | 0.1384 | 0.1389 |  0.1421 |  0.1489 |
|                         | Boost | 3.50\% | 3.14\% | 2.94\% | 3.16\% | 3.50\% | 4.27\% | 4.72\% | 6.34\% |  8.83\% | 10.42\% |
|            @2           | BLINK | 0.1995 | 0.1985 | 0.1993 | 0.1997 | 0.1991 | 0.1983 | 0.1973 | 0.1958 |  0.1956 |  0.2014 |
|                         | TIGER | 0.2055 | 0.2037 | 0.2042 | 0.2049 | 0.2049 | 0.2058 | 0.2061 | 0.2063 |  0.2105 |  0.2196 |
|                         | Boost | 3.03\% | 2.62\% | 2.50\% | 2.58\% | 2.93\% | 3.80\% | 4.45\% | 5.35\% |  7.63\% |  9.01\% |
|            @4           | BLINK | 0.2795 | 0.2777 | 0.2794 | 0.2797 | 0.2788 | 0.2787 | 0.2771 | 0.2749 |  0.2743 |  0.2829 |
|                         | TIGER | 0.2867 | 0.2845 | 0.2849 | 0.2865 | 0.2860 | 0.2864 | 0.2870 | 0.2866 |  0.2914 |  0.3039 |
|                         | Boost | 2.59\% | 2.46\% | 1.98\% | 2.45\% | 2.59\% | 2.79\% | 3.56\% | 4.25\% |  6.25\% |  7.41\% |
|            @8           | BLINK | 0.3721 | 0.3704 | 0.3721 | 0.3732 | 0.3714 | 0.3708 | 0.3692 | 0.3653 |  0.3651 |  0.3749 |
|                         | TIGER | 0.3797 | 0.3774 | 0.3780 | 0.3796 | 0.3788 | 0.3802 | 0.3802 | 0.3788 |  0.3837 |  0.3958 |
|                         | Boost | 2.06\% | 1.87\% | 1.57\% | 1.72\% | 2.00\% | 2.52\% | 2.97\% | 3.70\% |  5.07\% |  5.59\% |
|           @16           | BLINK | 0.4718 | 0.4703 | 0.4721 | 0.4727 | 0.4710 | 0.4703 | 0.4689 | 0.4643 |  0.4645 |  0.4726 |
|                         | TIGER | 0.4796 | 0.4776 | 0.4782 | 0.4795 | 0.4784 | 0.4797 | 0.4801 | 0.4773 |  0.4823 |  0.4937 |
|                         | Boost | 1.66\% | 1.56\% | 1.29\% | 1.45\% | 1.58\% | 1.99\% | 2.39\% | 2.81\% |  3.83\% |  4.47\% |
|           @32           | BLINK | 0.5749 | 0.5731 | 0.5745 | 0.5748 | 0.5729 | 0.5724 | 0.5711 | 0.5660 |  0.5659 |  0.5745 |
|                         | TIGER | 0.5826 | 0.5803 | 0.5810 | 0.5822 | 0.5809 | 0.5807 | 0.5819 | 0.5788 |  0.5825 |  0.5933 |
|                         | Boost | 1.34\% | 1.25\% | 1.13\% | 1.28\% | 1.40\% | 1.46\% | 1.90\% | 2.26\% |  2.94\% |  3.27\% |
|           @64           | BLINK | 0.6742 | 0.6731 | 0.6741 | 0.6749 | 0.6733 | 0.6730 | 0.6723 | 0.6675 |  0.6665 |  0.6737 |
|                         | TIGER | 0.6817 | 0.6795 | 0.6800 | 0.6813 | 0.6807 | 0.6805 | 0.6803 | 0.6769 |  0.6807 |  0.6906 |
|                         | Boost | 1.11\% | 0.95\% | 0.87\% | 0.95\% | 1.11\% | 1.12\% | 1.19\% | 1.42\% |  2.13\% |  2.52\% |

## Using your own data

If you want to use your own dataset, you only need to use the code in Dataset Construction. Construct your own dataset according to the description of the dataset construction process in the Supplementary Material.
