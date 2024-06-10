# ABSA-QUAD HF

This repo attempts to reproduce the results from [Aspect Sentiment Quad Prediction as Paraphrase Generation](https://arxiv.org/abs/2110.00796) (Zhang et al., 2021) using libraries and T5 model from Hugging Face.

## Setup

```
cmd.exe
python -m venv .venv
.venv\Scripts\activate.bat

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install datasets
pip install accelerate
pip install evaluate
pip install rouge-score
```

## Example usage

Fine tune ``t5-small`` model on the training split of the ``rest15`` dataset--a dataset for the Aspect Sentiment Quad Prediction task.

```
python absa-quad-hf.py --dataset rest15 --base-model t5-small --train
```

Run inference on a random sample. Given an input sentence, predict the resulting aspect sentiment quad in paraphrased form.

```
python absa-quad-hf.py --dataset rest15 --base-model t5-small --inf

input: I picked the Grilled Black Cod as my entree , which I absolutely devoured while someone commented that the Grilled Salmon dish was better .
output:
food quality is great because Grilled Black Cod is devoured [SSEP] food
```

Evaluate the fine-tuned model. Run inference on the *input* portion of the eval split of the ``rest15`` dataset and compare that to the *labeled* portion of the split.

```
python absa-quad-hf.py --dataset rest15 --base-model t5-small --eval

Rogue1: 80.318753%
rouge2: 72.354588%
rougeL: 79.076134%
rougeLsum: 79.112983%
number of gold spans: 795, predicted spans: 825, hit: 325
F1:        40.123457%
Precision: 39.393939%
Recall:    40.880503%
```

## Results

| Model    | Dataset | Pre   | Rec   | F1    | Eval (hh:mm:ss) | Train (hh:mm:ss) |
| -------- | ------- | ----- | ----- | ----- | --------------- | ---------------- |
| t5-small | rest15  | 39.39 | 40.88 | 40.12 | 00:01:37        | 00:18:49         |
| t5-base  | rest15  | 44.56 | 46.91 | 45.71 | 00:03:14        | 00:54:45         |
| t5-large | rest15  | 48.99 | 49.18 | 49.08 | 00:08:16        | 10:14:59         |

| Model    | Dataset | Pre   | Rec   | F1    | Eval (hh:mm:ss) | Train (hh:mm:ss) |
| -------- | ------- | ----- | ----- | ----- | --------------- | ---------------- |
| t5-small | rest16  | 53.21 | 54.94 | 54.06 | 00:01:37        | 00:32:52         |
| t5-base  | rest16  | 56.16 | 58.69 | 57.40 | 00:03:16        | 01:34:09         |
| t5-large | rest16  | 58.51 | 60.20 | 59.34 | 00:04:44        | 20:51:35         |

## GPU

GTX 1070 was used for results above.

```
            Arch     Cores                        Clock      Memory                        PCIe      TDP    Cost   TFLOPS
                     shader:tex:render:tensor:rt  (base)                                                           SP/DP/HP/Tensor
            ----     ----------                   ---------  ------                        ----      ---

Desktop:

GTX 1070    Pascal    1920:120: 64                1506 MHz    8GB GDDR5  256-bit  256GB/s  Gen3 x16  150W    $379   5.7/  .18/   .1/  -

RTX 2060    Turing    1920:120: 48:240: 30        1365        6GB GDDR6  192-bit  336GB/s  Gen3 x16  160W    $299   5.2/  .16/ 10.4/ 41
RTX 2070 S            2560:160: 64:320: 40        1605        8GB        256      448                215W    $499   8.2/  .25/ 16.4/ 65
TITAN RTX

RTX 3060    Ampere    3584:112: 48:112: 28        1320 MHz   12GB GDDR6  192-bit  360GB/s  Gen4 x16  170W    $329   9.4/  .14/  9.4/ 75
RTX 3070              5888:184: 96:184: 46        1500        8GB GDDR6  256      448                200W    $499  17.6/  .27/ 17.6/141
RTX 3080              8960:280: 96:280: 70        1260       12GB GDDR6X 384      912                350W    $799  22.6/  .35/ 22.6/180

RTX 4080    Ada       9728:304:112:304: 76        2210 MHz   16GB GDDR6X 256-bit  717GB/s  Gen4 x16  320W   $1199  43.0/  .67/ 43.0/172
RTX 4090             16384:512:176:512:128        2230       24GB        384-bit 1008GB/s            450W   $1599  73.1/ 1.10/ 73.1/292

Data Center:

A100        Ampere    6912                         765 MHz   40GB HBM2   5120b   1555GB/s            250W  <$15k?  19.5/ 9.70/312.0/  -

H100        Hopper   14592                        1065 MHz   80GB HBM2E  5120b   2039GB/s            350W  <$40k?  51.2/25.60/756.4/  -

H200    ?
```
