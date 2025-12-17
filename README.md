## 🔧 Installation

1.  **Create a conda environment**
    ```bash
    conda create -n disco python=3.9
    conda activate disco
    ```

2.  **Install PyTorch and CUDA dependencies**
    ```bash
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    ```

3.  **Install other dependencies**
    Please create a `requirements.txt` file with the dependencies listed below and run:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Citation
```bibtex
@article{wang2024scene,
    title={Scene graph disentanglement and composition for generalizable complex image generation},
    author={Wang, Yunnan and Li, Ziqiang and Zhang, Wenyao and Zhang, Zequn and Xie, Baao and Liu, Xihui and Zeng, Wenjun and Jin, Xin},
    journal={Advances in Neural Information Processing Systems (NeurlPS)},
    volume={37},
    pages={98478--98504},
    year={2024}
}
```

## License
This repository is released under the MiT license as found in the [LICENSE](LICENSE) file.