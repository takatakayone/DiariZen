# DiariZen
- This repository is a fork of [BUTSpeechFIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen), enhanced with Mamba and Samba architectures integrated into the speaker diarization pipeline.
- DiariZen is a speaker diarization toolkit driven by [AudioZen](https://github.com/haoxiangsnr/spiking-fullsubnet) and [Pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## Implementation Branches
- `feature/add_mamba`: Implementation of Mamba architecture integration
- `feature/add_samba`: Implementation of Samba architecture integration


## Installation
```
# create virtual python environment
conda create --name diarizen python=3.10
conda activate diarizen

# install diarizen 
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt && pip install -e .

# install pyannote-audio
cd pyannote-audio && pip install -e .[dev,testing]

# install dscore
git submodule init
git submodule update
```

## Usage
- For model training and inference, see `recipes/diar_ssl/run_stage.sh`.

## Result

### DER(Diarization Error Rate)

| Model | AMI  | AISHELL-4  | AliMeeting  |
|-------|----------|----------------|-----------------|
| Conformer(Baseline) | 15.61% | 14.71% | 15.41% |
| Mamba | **14.78%** | **8.50%** | **14.33%** |
| Samba | 15.17% | 11.56% | 14.99% |

**Settings:** Computed with a 0.25 s collar

### Training-Time Efficiency

| Model | Wall-Clock h | Avg GPU Memory GB | Peak GPU Memory GB | GPU Util % | Host RAM GB |
|-------|--------------|-------------------|-------------------|------------|-------------|
| Conformer | 24.3 | 7.9 | 21.1 | 83 | **25.6** |
| Mamba | **13.1** | **6.3** | **17.5** | 85 | 68.7 |
| Samba | 19.8 | 7.1 | 19.1 | **86** | 34 |

**Settings:** Single A100-80 GB GPU, batch size 64

### Inference Speed

| Model | Mean time | RTF | Speed (× real time) |
|-------|-----------|-----|---------------------|
| Conformer | 2.37 | 0.0395 | 25.3 × |
| Mamba | **2.31** | **0.0385** | **26.0 ×** |
| Samba | 2.32 | 0.0386 | 25.9 × |

**Settings:** `batch = 1`, `fp32 inference`, `PyTorch 2.1.1`, `cuDNN 8.9.2`

## License
This repository under the [MIT license](https://github.com/BUTSpeechFIT/DiariZen/blob/main/LICENSE).
