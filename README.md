# ClaimDetective-Training

This repo contains the code used to train [ClaimDetective](https://github.com/lawrence-chillrud/ClaimDetective), a check-worthiness / claim detection classification model made by fine-tuning RoBERTa under-the-hood. Much of this code was made to reproduce results from [Williams et. al.](https://arxiv.org/abs/2009.02431)

## Code Overview

1. [roberta.py](roberta.py) contains the code to train and test the claim detection model. 

2. [source](source) is a directory containing other source code used to help `roberta.py` run. 

    * [ernie.py](source/ernie.py) contains basic helper functions for argument parsing and output formatting etc.
    * [models.py](source/models.py) contains the model architecture. 

## Data Directories

When fine-tuning RoBERTa for claim-detection, 3 different datasets were used in varying combinations...

1. [ClaimBuster\_Datasets](ClaimBuster_Datasets) contains the relevant files from the [ClaimBuster dataset](https://zenodo.org/record/3609356#.X8q9RxNKhnE) described in the following paper: [Arslan et. al.](https://arxiv.org/abs/2004.14425) Briefly, the ClaimBuster dataset consists of *23,533 statements extracted from all U.S. general election presidential debates (1960-2016) which were then annotated by human coders.*

* [clef2019-factchecking-task1](clef2019-factchecking-task1) contains the relevant files from the [CLEF-2019 CheckThat! dataset](https://github.com/apepa/clef2019-factchecking-task1#scorers) (CT19-T1 corpus) described in the following paper: [Atanasova et. al.](https://groups.csail.mit.edu/sls/publications/2019/Mohtarami-CLEF2019.pdf) Briefly, the CT19-T1 corpus contains *23,500 human-annotated sentences from political speeches and debates during the 2016 U.S. presidential election.*

* [clef2020-factchecking-task1](clef2020-factchecking-task1) contains the relevant files from the [CLEF-2020 CheckThat! dataset](https://github.com/sshaar/clef2020-factchecking-task1#clef2020-checkthat-task-1) (CT20-T1(en) corpus) described in the following paper: [Barron-Cedeno et. al.](https://arxiv.org/abs/2007.07997) Briefly, the CT20-T1(en) corpus contains *962 human-annotated tweets about the novel coronavirus caused by SARS-CoV-2.*

## Workflow

Description of workflow and available arguments passed to `roberta.py` coming soon. For now, run the following for information:
```
python roberta.py --help
```

## To Do

In no particular order:

1. Implement loss functions in varying combinations to see if they help model performance.
    * Contrastive Sampling Ranking loss from: [Hansen et. al.](http://ceur-ws.org/Vol-2380/paper_56.pdf)
    * Adversarial loss from: [Meng et. al.](https://arxiv.org/abs/2002.07725)
    * Consistency loss from: [Xie et. al.](https://arxiv.org/abs/1904.12848)

2. Use the `.json` files from the ClaimBuster dataset to train the model instead of the `.csv` files... 

3. Send misclassified examples to colleagues... 

