# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

# Graph Aligner for In-Context Learning2

## Why not TAG?

OFA (ICLR 24), UniGraph (KDD 25), GOFA (ICLR 25), unify graphs from different domains with Text-Attributed Graphs (TAG).

![image.png](image.png)

Why TAG-based GFMs succeed on cross-domain GFMs?

**Citation Network:**

Node *“GCN is a semi-supervised GNN model”*                        Label: Data_Mining (label_id:0)

**Social Network:**

Node *“Trump is a president of the US”*                                        Label: Celebrity         (label_id:0)

Two texts occupying the same space means that if two texts differ, the language model must project them to distinct positions within the semantic space.

**Limitation**: Non-textual modalities (dense floats, categorical codes, molecular sub-graphs, images) must be forced into text.

Typically, the provided graph data has already been vectorized. For example, node features are commonly vectorized using self-supervised methods such as word2vec, and labels are typically represented as integers.

ogb

**Domain 1：**
Text attributes [0.3, 0.5, 0.2] learned from graph itself with word2vec,   label 0

**Domain 2:**
Text attributes [0.3, 0.5, 0.2] learned from graph itself with word2vec,   label 0

However, they are totally different things in two domains, so they are located in the **same positions of different feature and label spaces**.

![image.png](image%201.png)

**Potential solution:** **Combine feature spaces and label spaces** of different-domain graphs.

![image.png](image%202.png)

However, such simple solution can not reflect the relations between domains.

# 1. Domain Alignment

### 1.1 Pre-training Domain Embeddings (Define Domain Space)

We first force **all graphs to share one latent feature semantic space**.  Label-space alignment will then inherit that order. 
Choose any backbone architecture (GAT, GCN, SAGE …) and **one basic** initial parameter vector:

$$
\theta_0 \in \mathbb{R}^{d_\theta},
$$

where $\theta_0 = \mathrm{vec}(\mathbf{W})$ can be initialized by Xavier uniform distribution with gain = 1.0. If we consider 1-layer GNN models, the parameter matrix $\mathbf{W}$ is a single weight matrix with shape $\mathbb{R}^{d_{in} \times d_{out}}$, then we have a simplified GNN: 

$$
\hat{\mathbf{Y}} = \sigma(\mathbf{P}\mathbf{X}\mathbf{W})
$$

where $\theta^\star = \argmin_{\theta} \mathcal{L}(\hat{\mathbf{Y}}, \mathbf{Y};\theta)$ and $\theta = \mathrm{vec}(\mathbf{W})$, and $\mathbf{P} \in \{\mathbf{A}, \hat{\tilde{\mathbf{A}}}, \mathbf{D}^{-1}\mathbf{A},(\mathbf{I}-\mathbf{D}^{-1}\mathbf{A})\}$ . For every pre-training graph $\mathcal{G}=\left\{G_i=\left(V_i, E_i, X_i, Y_i\right) \mid i=1, \ldots, M\right\}$, perform only one, possibly two, optimization gradient steps, we have:

$$
\theta_i=\theta_0-\eta \nabla_\theta \mathcal{L}_i\left(\theta_0\right)
$$

where $\mathcal{L}_i$ is the task loss on $G_i \in \mathcal{G}$, and $\eta$ is a learning rate snapshot step. **This is just one gradient steps, not a full training run**. We call the resulting encoder for $G_i$ as $f_i := f_{\theta_i}$.

Since every $\theta_i$ for $G_i$ is a nearby point on the loss landscape around $\theta_0$, the vector $\Delta \theta_i=\theta_i-\theta_0 \in \mathbb{R}^{d_\theta}$ captures how graph $G_i$ pulls the shared model to fit its own data. Graphs from similar domains produce similar $\Delta \theta$, and dissimilar domains bend the weights in orthogonal directions. **One step gradient is enough to capture the interactions between the graph and labels, which considers three factors of the graph to fully capture the domain characteristics.** 

Since $\theta = \mathrm{vec}(\mathbf{W}) \in \mathbb{R}^{d_\theta}$  where $d_\theta = d_{in} \cdot d_{out}$,projecting the high-dimensional weight-deltas $\{\Delta \theta\}^M_{i = 1}$ to a compact vector with a projection matrix $\mathbf{B}$, we can get a **domain embedding** $e_i$ as: 

$$
e_i=\mathbf{B} \Delta \theta_i \in \mathbb{R}^{d_e}
$$

where $d_e \ll d_\theta$. Then we perform layer normalization on $e_i$ to remove scale variation as $e_i \leftarrow \frac{e_i-\mu(e_i)}{\sigma(e_i)}$, where  $\mu$ and $\sigma$ are mean and standard deviation of $e_i$.

**Metric property of domain embedding** $e_i$. A good $\mathbf{B}\in \mathbb{R}^{d_e \times d_\theta}$ should preserve the geometry (i.e., pair-wise distances) of $\Delta \theta$ set. In other words, a good $\mathbf{B}$ should compress $\Delta \theta$ while preserve the relationships between different $\Delta \theta$:

$$
\forall i,j: \quad \left\|e_i-e_j\right\|_2 \approx\left\|\Delta \theta_i-\Delta \theta_j\right\|_2
$$

$$
\mathbf{B} = \arg\min_{\mathbf{B}} \sum_{i,j} \mathrm{Distance}(\left\|e_i-e_j\right\|_2, \left\|\Delta \theta_i-\Delta \theta_j\right\|_2)  
$$

We can use PCA-based domain embedding to compress weight deltas into compact domain embeddings. Specifically, we first stack all deltas into a data matrix $\Delta \Theta=\left[\begin{array}{c}
\left(\Delta \theta_1\right)^{\top} \\
\vdots \\
\left(\Delta \theta_M\right)^{\top}
\end{array}\right] \in \mathbb{R}^{M \times d_\theta}$,  then compute the mean with $\Delta\bar{\theta}=\frac{1}{M} \sum_{i=1}^M \Delta \theta_i$ and centre every row we have $\Delta\tilde{\Theta}=\Delta \Theta-\mathbf{1}_M \bar{\theta}^{\top}$. Define the Gram matrix as: $\mathbf{G} = \frac{1}{M}\Delta\tilde{\Theta}\Delta\tilde{\Theta}^\top \in \mathbb{R}^{M \times M}$ and compute its eigen-decomposition by **truncated** SVD as:

$$
\mathbf{G} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top, \quad \boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1, \cdots, \lambda_r),
$$

and obtain the right singular vectors $\mathbf{V}=\Delta\tilde{\Theta}^{\top} \mathbf{U} \mathbf{\Lambda}^{-1 / 2} \in \mathbb{R}^{d_\theta \times r}$, which can be used to yield the compact SVD $\Delta\tilde{\Theta} =  \mathbf{U} \boldsymbol{\Lambda}^{\frac{1}{2}}\mathbf{V}^\top$, where $\lambda_1 \geq \cdots \geq \lambda_r>0$ and $r=\operatorname{rank}(\Delta\tilde{\Theta}) \leq \min \left\{M, d_\theta\right\}$. We can choose an intrinsic dimension $d_e \leq r$ (in most cases, $r = M$), and define the projection matrix  $\mathbf{B}$ as:

$$
\mathbf{B} = (\mathbf{V}_{:d_e})^{\top} \in \mathbb{R}^{d_\theta \times d_e}
$$

which is the first $d_e$ eigenvectors transposed (with the largest $d_e$ eigenvalues). Since columns of $\mathbf{V}$ are orthonormal, $\mathbf{B}$ is orthonormal as well and $\mathbf{B}\mathbf{B}^\top = \mathbf{I}_{d_e}$.

**Theorem  1**. (Minimum reconstruction error) For any rank-$d_e$ linear map $Q \in \mathbb{R}^{d_e \times d_\theta}$,  if the following inequation holds:

$$
\sum_{i=1}^M\left\|\Delta \theta_i-Q^{\top} Q \Delta \theta_i\right\|_2^2 \geq \sum_{i=1}^M\left\|\Delta \theta_i-\mathbf{B}^{\top} \mathbf{B} \Delta \theta_i\right\|_2^2
$$

According to Eckart-Young-Mirsky theorem, PCA (i.e., $\mathbf{B}$) is the unique minimizer of the Frobenius-norm reconstruction error.

### **1.2 In-Context Domain Embedding**

Run one gradient step on the same model initialization $\theta_0$ using data from $G_{\mathrm{new}}$ with a few labels as prompt for **in-context learning**:

$$
\Delta\theta^{(1)}_{\mathrm{new}} = \theta_0 - \eta \nabla_\theta \mathcal{L}_{\text {new }}\left(\theta_0\right)
$$

then the domain embedding of new graph is computed as:

$$
e_{\mathrm{new}} = \mathbf{B}(\theta^{(1)}_{\mathrm{new}}-\theta_0)
$$

![image.png](image%203.png)

# 2. Domain-Conditioned Aligner

### 2.1 Feature Alignment

We first define a neural network:

$$
(\gamma^{\text{feat}}_i, \beta^{\text{feat}}_i) = f_{\phi_f} (e_i), \quad \gamma^{\text{feat}}_i, \beta^{\text{feat}}_i \in \mathbb{R}^d 
$$

where $f_{\phi_f}(\cdot) = \mathrm{SoftPlus}(\mathrm{MLP}_{\phi_f}(\cdot))$ is a domain-specific feature adapter. For every node $v \in G_i$, feeding it into a shared GNN encoder whose parameters initialized by $\theta_0$:

$$
h_{iv} = f_{\theta_0}(v, G_i) \in \mathbb{R}^d
$$

through the domain-conditioned map $T_{e_i}(\cdot)$, we have:

$$
z_{iv} = T_{e_i}(h_{iv}) = \gamma^{\text{feat}}_i \odot h_{i v}+\beta^{\text{feat}}_i 
$$

![image.png](image%204.png)

Assuming that $f_\phi(\cdot)$ is Lipschitz with constant $L$, we have:

$$
\left\|\gamma^{\text{feat}}_i-\gamma^{\text{feat}}_j\right\|+\left\|\beta^{\text{feat}}_i-\beta^{\text{feat}}_j\right\| \leq L\left\|e_i-e_j\right\|
$$

Thus, similarity domain embeddings lead to similar domain-conditioned mappings, then node embeddings of neighboring domains land in nearby subspaces in the unified feature spaces spanning by $\{z_{iv}\}$. Besides, $T_{e_i}$ is linear up to a translation, so it cannot destroy existing relational structure inside $h_{iv}$.

**In-Context Feature Alignment.** For a downstream graph $G_{\mathrm{new}}$ from an unknown domain, with its in-context domain embedding $e_{\mathrm{new}}$, we have $(\gamma^{\text{feat}}_{\text{new}}, \beta^{\text{feat}}_{\text{new}}) = f_{\phi_f} (e_{\mathrm{new}})$. The aligned feature of $G_{\mathrm{new}}$ is $z_{\text{new}, v} = \gamma^{\text{feat}}_{\text{new}} \odot f_{\theta_0}(v, G_{\mathrm{new}}) +\beta^{\text{feat}}_{\text{new}}$. With these, the model can do zero-shot inference or be refined with a handful of labelled nodes. These embeddings now live in the **unified feature space**, where graphs with similar domains produce similar feature transforms,  and therefore occupy neighbouring subspaces.

### 2.2 Label Alignment

We first create a shared matrix as base semantic lookup:

$$
\mathbf{E}^{\mathrm{label}} \in \mathbb{R}^{\left|\cup_i^M L_{i}\right|\times d} 
$$

where $L_i = \{1,\cdots,|L_i|\}$ is the label IDs that $G_i$ uses for its nodes, and $\left|\cup_i^M L_{i}\right|$ is the total count of distinct raw label IDs across all pre-training graphs. Each row $\mathbf{E}^{\mathrm{label}}_l \in \mathbb{R}^d$ is the domain-agnostic label embedding of raw label ID $l$ before any domain context is applied. Let $\mathbf{E}^{\mathrm{label}}_l$ be initialized with Gaussian distribution $\mathbf{E}^{\mathrm{label}}_l \sim \mathcal{N}(0, \sigma_0^2\mathbf{I}_d)$ or Uniform distribution $\mathbf{E}^{\mathrm{label}}_l \sim \mathcal{U}[-u, u]$. 

**Domain-aware FiLM Label Embedding.**  Based on the domain embedding $e_i \in \mathbb{R}^{d_e}$ for the graph $G_i$, we apply the same feature alignment principle by defining a domain-specific label adapter  $f_{\phi_l}$ with an identical structure to the feature adapter $f_{\phi_f}$ as: 

$$
(\gamma^{\text{lab}}_i, \beta^{\text{lab}}_i) = f_{\phi_l} (e_i), \quad \gamma^{\text{lab}}_i, \beta^{\text{lab}}_i \in \mathbb{R}^d 
$$

Then the domain-aware label embedding of label $l$ in $G_i$ can be defined as:

$$
u_{il} = \gamma^{\text{lab}}_i \odot \mathbf{E}^{\mathrm{label}}_l +    \beta^{\text{lab}}_i 
$$

where $l = 0,\cdots, |L_i|-1$.

**In-Context Label Alignment.** Let the unseen graph be $G_{\text{new}}$ with its label IDs $\{0,\cdots, L_{\text{new}}-1\}$, we can index the first $L_{\text{new}}$ rows of the label embedding lookup matrix $\mathbf{E}^{\mathrm{label}}_{0:L_{\text{new}}-1}$ as the domain-agnostic label embedding of $G_{\text{new}}$. For every raw label index $l$, its label embedding can be represented as:

$$
u_{\text {new},l}=\gamma_{\text {new }}^{\text {lab }} \odot \mathbf{E}_{l}^{\text {lab }}+\beta_{\text {new }}^{\text {lab }}
$$

# 3. Few-Shot Pre-Training Enabling In-Context Learning

For every pre-training graph $G_i$, we have:

Domain-aligned features for node $v$ as:  

$$
z_{iv} = \gamma^{\text{feat}}_i \odot h_{i v}+\beta^{\text{feat}}_i 
$$

And domain-aligned label embedding for label index $l$ as:

$$
u_{il} = \gamma^{\text{lab}}_i \odot \mathbf{E}^{\mathrm{label}}_l +    \beta^{\text{lab}}_i 
$$

For $G_i$, we organize an episode (training task) as $m$-way $k$-shot prompting in pre-training. Specifically, in each training step, we sample $m$ way label subset $\mathcal{C} = \{l_{i1}, \cdots, l_{im}\} \subseteq\left\{0, \ldots,\left|\mathcal{L}_{G_i}\right|-1\right\}$. For each class $c \in \mathcal{C}$, we sample $k$ support nodes $S_c = \{v_{c,1} , v_{c,2},\cdots, v_{c,k}\}$ with their label $l_{ic}$,  and $T$ query nodes $Q_c = \{q_{c,1}, q_{c,2}, \cdots,q_{c,T}\}$.With these samples, we can build an token sequence as:

$$
\left[ \underbrace{(z_{iv_{1,1}}, u_{il_1}),(z_{iv_{1,2}}, u_{il_1}),\cdots, (z_{iv_{1,k}}, u_{il_1})}_{k\text{-shot prompt for way }l_{i1} }, \cdots,\underbrace{(z_{iv_{m,1}}, u_{il_m}), \cdots,(z_{iv_{m,k}}, u_{il_m})}_{k\text{-shot prompt for way } l_{im}}\right]
$$

where we have $m\times k$ samples in the support set. For a query node $q_{c,t}$, its query token is defined as the domain-aligned node feature $z_{iq_{c,t}}$. The training objective is to make prediction for one query based on one prompt at a time.  Model should build the connection between nodes and label embeddings, i.e., label embedding 和节点特征之间的对应关系(某一个类别的nodes，即node embedding和label embedding之间的自适应匹配). To this end, we propose Dual Prompt-aware Multi-head Attention (DPMA).

### Dual Prompt-Aware Attention (DPAA)

**Feature-side attention.** For the aligned feature tokens $\mathbf{Z} = \left[  z_{iv_{1,1}} z_{iv_{1,2}}, \cdots, z_{iv_{1,k}}, \cdots,z_{iv_{m,1}}, \cdots,z_{iv_{m,k}}\right] \in \mathbb{R}^{(mk) \times d}$ in the pre-training graph $G_i$, we make keys, values and queries parameterized by $\mathbf{W}_K$, $\mathbf{W}_V$ and $\mathbf{W}_Q$ for multi-head attention as:

$$
\begin{aligned}\mathbf{K}^{\text{feat}} &= \mathbf{Z}\mathbf{W}_K \\  \mathbf{V}^{\text{feat}} &= \mathbf{Z}\mathbf{W}_V \\\mathbf{Q}^{\text{feat}} &= z_{iq_{c,t}} \mathbf{W_Q}\end{aligned}
$$

Then the query node $q_{c,t}$’s representation of the prompt-aware attention layer is:

$$
z^{\text{out}}_{iq_{c,t}} = \mathrm{softmax} \left(\frac{\mathbf{Q}^{\text{feat}}\mathbf{K}^{\text{feat}\top}}{\sqrt{d}}\right)\mathbf{V}^{\text{feat}},
$$

where only the query token attends to the prompt, and support tokens cannot interact with each other, enforcing a strict In-Context Learning pattern.

**Label-side attention.** For the aligned label tokens $\mathbf{U} = \left[u_{il_1}, u_{il_2}, \cdots, u_{il_m}\right]$ in an episode, the ground-truth label embedding of the query is $u_{iq_{c,t}}$ we have:

$$
\begin{aligned}\mathbf{K}^{\text{label}} &= \mathbf{U}\mathbf{W}_K \\  \mathbf{V}^{\text{label}} &= \mathbf{U}\mathbf{W}_V \\\mathbf{Q}^{\text{label}} &= u_{iq_{c,t}}\mathbf{W_Q}\end{aligned}
$$

Then the query node $q_{c,t}$’s label embedding of the label-side attention layer is:

$$
u_{iq_{c,t}}^{\text{out}} = \mathrm{softmax} \left(\frac{\mathbf{Q}^{\text{label}}\mathbf{K}^{\text{label}\top}}{\sqrt{d}}\right)\mathbf{V}^{\text{label}},
$$

where  $\mathbf{W}_K$, $\mathbf{W}_V$ and $\mathbf{W}_Q$ are shared in DPAA.

For the embedding $z^{\text{out}}_{iq_{c,t}}$ of the query node $q_{c,t}$, mapping it into label space with a learnable function $\hat{z}^{\text{out}}_{iq_{c,t}} = g(z^{\text{out}}_{iq_{c,t}})$ and we can define a node-label matching loss by compute the logits against $m$ support-class label embeddings:

$$
\boldsymbol{s} = \hat{z}^{\text{out}}_{iq_{c,t}} \mathbf{U}\mathbf{W}_V \in \mathbb{R}^{m \times 1}
$$

$$
\mathcal{L}({q_{c,t}}) = -\log \frac{\exp \boldsymbol{s}[c]}{\sum_{j}^M} \exp \boldsymbol{s}[j]
$$

For each pre-training graph $G_i$, we set $M = |L_{G_i}|$. For all $T$ query nodes in each class of each pre-training graph, we can compute the overall episode loss as:

$$
\mathcal{L} = \frac{1}{|L_{G_i}|T} \sum^{|L_{G_i}|}_{c=1} \sum^T_{t=1}\mathcal{L}({q_{c,t}})   
$$

which forces the network to reading the prompt and matching node to class.

# 4. Applying to an unseen graph $G_{\text{new}}$ with In-Context Learning at Test Time

With the domain-aligned features $z_{\text{new}, v}$ for each node $v$ in $G_{\text{new}}$ and domain-aligned label $u_{\text {new},l}$ for each label index $l$. We first build $|L_{G_{\text{new}}}|$-way $k$-shot prompt by collecting $k$ examples per class, where the support token sequence is built exactly as pre-training stage, but with new graph’s vectors: $\mathbf{Z}^{\text{pmt}}_{\text{new}} = [\cdots,z_{\text{new}, v}, \cdots] \in \mathbb{R}^{(|L_{G_{\text{new}}}|k) \times d}$. The prompt label embeddings $\mathbf{U}^{\text{pmt}}_{\text{new}} = [\cdots,u_{\text {new},l},\cdots] \in \mathbb{R}^{|L_{G_{\text{new}}}| \times d}$. For the feature-side attention of **DPAA**, by applying the pre-trained  $\mathbf{W}_K$, $\mathbf{W}_V$ on both $\mathbf{Z}^{\text{pmt}}_{\text{new}}$ and $\mathbf{U}^{\text{pmt}}_{\text{new}}$, we have $\left(\mathbf{K}^{\text{feat}}_{\text{new}},\mathbf{V}^{\text{feat}}_{\text{new}}\right)$ and $\left(\mathbf{K}^{\text{label}}_{\text{new}},\mathbf{V}^{\text{label}}_{\text{new}}\right)$. For a test node $q$ with its domain-aligned node feature $z_{\text{new}, q}$ to be predicted, we first compute the interaction between  $z_{\text{new}, q}$ and the prompt sequence (used to prompt the prediction) to get a prompt-aware embedding of $q$ as:  

$$
\begin{equation}\begin{aligned}\mathbf{Q}^{\text{feat}}_{\text{new}} &= z_{\text{new}, q} \mathbf{W}_Q\\z^{\text{out}}_{\text{new},q} &= \mathrm{softmax} \left(\frac{\mathbf{Q}^{\text{feat}}_{\text{new}}\mathbf{K}^{\text{feat}\top}_{\text{new}}}{\sqrt{d}}\right)\mathbf{V}^{\text{feat}}_{\text{new}},\end{aligned}\end{equation}
$$

Then mapping the query feature embedding to the label space with the pre-trained $\hat{z}^{\text{out}}_{\text{new},q} = g(z^{\text{out}}_{\text{new},q})$ and compute the logits against all $|L_{G_{\text{new}}}|$ support-class aligned label embeddings as:

$$
\boldsymbol{s}_q = \hat{z}^{\text{out}}_{\text{new},q}\mathbf{U}^{\text{pmt}}_{\text{new}}\mathbf{W}_V \in \mathbb{R}^{|L_{G_{\text{new}}}|\times1} 
$$

The prediction of the query node $q$ is given by:

$$
\hat{y}_q = \arg\max \boldsymbol{s}_q
$$


## Architecture

### Core Components

- **`model/`**: Neural network architectures
  - `base.py`: Base GNN implementations and custom SAGE convolution
  - `FiLM.py`: Domain adaptation FiLM layers
  - `fingerprint.py`: Domain fingerprint generation
  - `pt_model.py`: Main pretraining model
  - `space_comb.py`: Feature space combination utilities

- **`data_process/`**: Data loading and preprocessing
  - `data.py`: Dataset classes for single graphs and combined datasets
  - `datahelper.py`: Data preprocessing utilities and text encoders
  - `task_constructor.py`: Task construction for different learning scenarios

- **`configs/`**: Hydra configuration files
  - `main.yaml`: Main experiment configuration with paths and pretraining settings
  - `model.yaml`: Model architecture hyperparameters
  - `data.yaml`: Dataset configurations and task definitions

- **`utils/`**: Utility functions
  - `config.py`: Configuration utilities and OmegaConf resolvers
  - `exp.py`: Experiment initialization
  - `logging.py`: Logging utilities
  - `utils.py`: General utilities including seed setting

### Key Design Patterns

- Uses Hydra for configuration management with hierarchical configs
- Implements custom PyTorch Geometric MessagePassing layer (MySAGEConv)
- Supports both node features and edge features in graph convolutions
- Uses PCA for feature dimensionality reduction and alignment
- Implements domain-specific fingerprinting for cross-dataset alignment

## Development Commands

### Running Pretraining
```bash
python pretrain.py
```

### Configuration Override Examples
```bash
# Override specific config values
python pretrain.py model.name=CustomModel pretrain.pretrain_epochs=100

# Use different dataset configuration
python pretrain.py dataset=Xcora pretrain.gpu=0
```

## Important Implementation Details

### Custom Graph Convolution
The project uses a custom SAGE convolution (`MySAGEConv`) that incorporates edge features:
- Located in `model/base.py:23-114`
- Combines node features with edge features during message passing
- Uses `(x_j + xe).relu()` for message computation

### Configuration System
- Uses Hydra with nested configurations
- Main config merges data.yaml and model.yaml automatically
- Support for environment-specific user configs
- Custom OmegaConf resolvers for eval and alias renaming

### Dataset Handling
- Supports both PyTorch Geometric datasets (pyg) and OFA text datasets (ofa)
- Implements custom dataset classes for single graphs and multi-dataset combinations
- Uses metadata lookup system in `data.yaml` for dataset routing

### Pretraining Architecture
- Combines multiple datasets into single training batches
- Uses domain embeddings for cross-dataset alignment
- Supports various task types: node classification, link prediction, knowledge graph completion

## File Organization

- Configuration files use YAML with Hydra @package declarations
- Python modules follow standard structure with clear separation of concerns
- Datasets stored in `datasets/` with source-specific subdirectories
- Output and temporary files organized under project root with unique identifiers

## Dependencies

Key dependencies inferred from imports:
- PyTorch and PyTorch Geometric
- Hydra for configuration
- OmegaConf for config manipulation
- scikit-learn for PCA
- Weights & Biases for experiment tracking
- rootutils for project root management