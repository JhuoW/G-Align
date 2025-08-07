# G-Align

```
datasets/
├── ofa
│   ├── SingleTextGraph.arxiv
│   │   ├── arxiv_CS_categories.txt
│   │   ├── labelidx2arxivcategeory.csv.gz
│   │   ├── nodeidx2paperid.csv.gz
│   │   ├── ogbn_arxiv
│   │   │   ├── mapping
│   │   │   │   ├── labelidx2arxivcategeory.csv.gz
│   │   │   │   ├── nodeidx2paperid.csv.gz
│   │   │   │   └── README.md
│   │   │   ├── processed
│   │   │   │   ├── geometric_data_processed.pt
│   │   │   │   ├── pre_filter.pt
│   │   │   │   └── pre_transform.pt
│   │   │   ├── raw
│   │   │   │   ├── edge.csv.gz
│   │   │   │   ├── node-feat.csv.gz
│   │   │   │   ├── node-label.csv.gz
│   │   │   │   ├── node_year.csv.gz
│   │   │   │   ├── num-edge-list.csv.gz
│   │   │   │   └── num-node-list.csv.gz
│   │   │   ├── RELEASE_v1.txt
│   │   │   └── split
│   │   │       └── time
│   │   │           ├── test.csv.gz
│   │   │           ├── train.csv.gz
│   │   │           └── valid.csv.gz
│   │   ├── processed
│   │   │   ├── data.pt
│   │   │   ├── geometric_data_processed.pt
│   │   │   ├── pca_64.pt
│   │   │   ├── pre_filter.pt
│   │   │   ├── pre_transform.pt
│   │   │   └── texts.pkl
│   │   └── raw
│   └── SingleTextGraph.WikiCS
│       ├── metadata.json
│       ├── processed
│       │   ├── data.pt
│       │   ├── data_undirected.pt
│       │   ├── geometric_data_processed.pt
│       │   ├── pca_64.pt
│       │   ├── pre_filter.pt
│       │   ├── pre_transform.pt
│       │   └── texts.pkl
│       └── raw
│           └── data.json
└── pyg
    └── Planetoid.PubMed
        ├── processed
        │   ├── data.pt
        │   ├── pca_64.pt
        │   ├── pre_filter.pt
        │   └── pre_transform.pt
        ├── PubMed
        │   ├── processed
        │   │   ├── data.pt
        │   │   ├── pre_filter.pt
        │   │   └── pre_transform.pt
        │   └── raw
        │       ├── ind.pubmed.allx
        │       ├── ind.pubmed.ally
        │       ├── ind.pubmed.graph
        │       ├── ind.pubmed.test.index
        │       ├── ind.pubmed.tx
        │       ├── ind.pubmed.ty
        │       ├── ind.pubmed.x
        │       └── ind.pubmed.y
        └── raw

```
