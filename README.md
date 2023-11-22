# UFIN

This is the official PyTorch implementation for the paper:

- [UFIN: Universal Feature Interaction Network for Multi-Domain Click-Through Rate Prediction]

## Overview

We propose the Universal Feature Interaction Network (UFIN) approach for CTR prediction.
UFIN exploits textual data to learn universal feature interactions that can be effectively transferred across diverse domains.
For learning universal feature representations, we regard the text and feature as two different modalities and  propose an encoder-decoder network founded on a Large Language Model (LLM)  to enforce the transfer of data from the text modality to the feature modality.
Building upon the above foundation, we further develop a mixture-of-experts (MoE) enhanced adaptive feature interaction model to learn transferable collaborative patterns across multiple domains.
Furthermore,  we propose a multi-domain knowledge distillation framework to enhance  feature interaction learning.
Based on the above methods, UFIN can effectively bridge the semantic gap to learn common knowledge across various domains, surpassing the constraints of ID-based models.
Extensive experiments conducted  on eight datasets show the effectiveness of UFIN, in both multi-domain and cross-platform settings.

![model](./asset/model.jpg)

## Requirements

```
tensorflow==2.4.1
python==3.7.3
cudatoolkit==11.3.1
pytorch==1.11.0
transformers
```

