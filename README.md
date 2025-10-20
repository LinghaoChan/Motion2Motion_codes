# Motion2Motion: Cross-topology Motion Transfer with Sparse Correspondence

[Ling-Hao Chen](https://lhchen.top/)<sup>1,2</sup>, [Yuhong Zhang]()<sup>1</sup>, [Zixin Yin](https://zxyin.github.io/)<sup>3</sup>, [Zhiyang Dou](https://frank-zy-dou.github.io/)<sup>4</sup>, [Xin Chen](https://chenxin.tech/)<sup>5</sup>, [Jingbo Wang](https://wangjingbo1219.github.io/)<sup>6</sup>, [Taku Komura](https://cs.hku.hk/index.php/people/academic-staff/taku)<sup>4</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>2</sup>


<sup>1</sup>Tsinghua University, <sup>2</sup>IDEA Research, <sup>3</sup>HKUST, <sup>4</sup>HKU, <sup>5</sup>ByteDance, <sup>6</sup>Shanghai AI Lab

**Contact**: Ling-Hao CHEN via email (thu [dot] lhchen [at] gmail [dot] com).

<p align="center">
  <strong>✨ACM SIGGRAPH Asia 2025✨</strong>
</p>

<p align="center">
  <a href='https://arxiv.org/abs/2508.13139'>
  <img src='https://img.shields.io/badge/Arxiv-2508.13139-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a> 
  <a href='https://arxiv.org/pdf/2508.13139'>
  <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a> 
  <a href='https://github.com/LinghaoChan/Motion2Motion_codes'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
  <a href='https://youtu.be/EMdKwyHKSTc'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a> 
</p>



<p align="center">
  <video width="800" controls>
    <source src="./assets/cross-topo-retarget.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>


**💡Tips**: As Motion2Motion is quite lightweight, please run it on the CPUs of our personal laptops. Without the necessary, please don't run it on GPUs and don't interrupt others to implement AGI.


## 🚀Running Instructions

### 🔧 Installation

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```


### 💻 Motion2Motion running command

```python
python run_M2M.py \
-e './data/processed/Monkey/__Walk.bvh' \
-d cpu \
--source './data/processed/Flamingo/Flamingo_Walk.bvh' \
--mapping_file configs/mappings_flamingo.json \
--output_dir ./demo_output \
--sparse_retargeting \
--matching_alpha 0.9
```
or running `bash run.sh` in the root directory.
This command will generate new motion sequences by retargeting the input BVH file (`./data/processed/Flamingo/Flamingo_Walk.bvh`) using the source motion from Monkey motions. The generated motions will be saved in the `demo_output/` directory.

Here,
- `-e` or `--example`: Input BVH file (target motion to retarget)
- `-d` or `--device`: Device to run on (cpu/cuda)
- `--source`: Source BVH file (motion to be retargeted from)
- `--mapping_file`: JSON file containing joint mapping between source and target skeletons
- `--output_dir`: Directory to save generated results
- `--sparse_retargeting`: Enable sparse retargeting mode
- `--matching_alpha`: Alpha parameter for motion matching (0.0-1.0)

Generated motions will be saved in the specified output directory (`output_dir`) as BVH files with detailed naming conventions indicating the synthesis parameters used.



### 🔩Configuration Files

Joint mapping files are available in the [`configs/`](configs/) directory to specify the bone binding relationships, aka. the sparse correspondences in the paper. The mapping files are in JSON format, and the structure is as follows:


```bash
{
    "source_name": $SOURCE_CHARACTER_NAME,
    "target_name": $TARGET_CHARACTER_NAME,
    "root_joint": $ROOT_BONE_NAME,
    "mapping": [
        {
            "target": $TARGET_BONE_NAME,
            "source": $SOURCE_BONE_NAME
        },
    ]
}
```

**⚠️ Warnning and Note**: In some cases, two joint names in the same BVH file may overlap, such as "LeftArm" and "LeftArmxxx". This will results some errors in the BVH parsing. To avoid this, please ensure that the joint names in the BVH files are unique. Our solution is to rename the joints in the BVH files before running the code. You can use the `add_random_string_of_joints.py` script to add a random string to the joint names in the BVH files. You can process all [Truebones-Zoo dataset](ttps://truebones.gumroad.com/l/skZMC/) with the script. Usage:

```bash
python utils/add_random_string_of_joints.py -i ./data/processed/Monkey/__Walk.bvh -o ./data/processed/Monkey/__Walk_modified.bvh
```

## 🏹 More Information

- We plan to release more cases of our demo in the paper in Sep. 2025. 

- Considering the copyright of [Truebones-Zoo dataset](ttps://truebones.gumroad.com/l/skZMC/), we can only provide few examples and the file processing tutorials in the doc. Please refer to the "**⚠️ Warnning and Note**" part. 

- We also provide some tools to visualize rest pose ([here](utils/get_rest_pose.py)) and skeleton tree ([here](utils/get_tree.py)). 

## 📖 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{chen2025motion2motion,
  title={Motion2Motion: Cross-topology Motion Transfer with Sparse Correspondence},
  author={Chen, Ling-Hao and Zhang, Yuhong and Yin, Zixin and Dou, Zhiyang and Chen, Xin and Wang, Jingbo and Komura, Taku and Zhang, Lei},
  booktitle={SIGGRAPH Asia 2025},
  year={2025},
  publisher={ACM},
  doi={10.1145/3757377.3763811},
  address={Hong Kong, China},
  isbn={979-8-4007-2137-3/2025/12}
}
```

## ❤️ Acknowledgments

Work done during Ling-Hao Chen's internship at IDEA Research. The author team would like to acknowledge all program committee members for their extensive efforts and constructive suggestions. In addition, Weiyu Li (HKUST), Shunlin Lu (CUHK-SZ), and Bohong Chen (ZJU) had discussed with the author team many times throughout the process. The author team would like to convey sincere appreciation to them as well. Our codes are based on [GenMM](https://github.com/wyysf-98/GenMM).
