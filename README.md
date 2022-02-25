# Generative Principal Component Analysis

This repository contains the codes for the paper: 

Zhaoqiang Liu, Jiulong Liu, Subhroshekhar Ghosh, Jun Han, and Jonathan Scarlett, "Generative Principal Component Analysis", accepted to International Conference on Learning Representations (ICLR), 2022.

-------------------------------------------------------------------------------------

## Dependencies

* Python 3.6

* Tensorflow 1.5.0

* Scipy 1.1.0

*  PyPNG

## Running the code

-------------------------------------------------------------------------------------

We provide the guideline to run our Algorithm 1 (PPower) on the MNIST and Fashion-MNIST datasets. 


Run PPower (and TPower, Power) on the MNIST dataset for the spiked covariance model,

python mnist_main_spikedcov.py --num-outer-measurement-ls 100 200 300 400 500 --beta-ls 1 --num-experiments 10 --method-ls Power TPower PPower

python mnist_main_spikedcov.py --num-outer-measurement-ls 300 --beta-ls 0.6 0.7 0.8 0.9 1 2 3 4 --num-experiments 10 --method-ls Power TPower PPower


Run PPower (and TPower, Power) on the MNIST dataset for phase retrieval,

python mnist_main_phaseretrieval.py --num-outer-measurement-ls 50 100 200 400 800 1600   --num-experiments 10 --method-ls Power TPower PPower




Run PPower (and TPower, Power) on the Fashion-MNIST dataset for the spiked covariance model,

python fashion_main_spikedcov.py --num-outer-measurement-ls 50 100 200 300 400 500 --beta-ls 1 --num-experiments 10 --method-ls Power TPower PPower

python fashion_main_spikedcov.py --num-outer-measurement-ls 300 --beta-ls 0.6 0.7 0.8 0.9 1 2 3 4 --num-experiments 10 --method-ls Power TPower PPower


Run PPower (and TPower, Power) on the Fashion-MNIST dataset for phase retrieval,

python fashion_main_phaseretrieval.py --num-outer-measurement-ls 50 100 200 400 800 1600   --num-experiments 10 --method-ls Power TPower PPower




Run PPower (and TPower, TPowerW, Power) on the CelebA dataset for the spiked covariance model,

python celebA_main_spikedcov.py --num-input-images 8 --num-outer-measurement-ls 3000 --beta-ls 0.6   0.8  1 2 3 4  --num-experiments 10 --method-ls Power TPower TPowerW PPower
python celebA_main_spikedcov.py --num-input-images 8 --num-outer-measurement-ls 2000 3000 6000 10000 --beta-ls 1  --num-experiments 10 --method-ls Power TPower TPowerW PPower


Run PPower (and TPower, TPowerW, Power) on the CelebA dataset for phase retrieval,


python celebA_main_phaseretrieval.py --num-input-images 8 --num-outer-measurement-ls 2000 3000 6000 10000 15000  24000 36000 --num-experiments 10 --method-ls Power TPower TPowerW PPower



## References

Large parts of the code are derived from [Bora et al.](https://github.com/AshishBora/csgm), [ Shah et al.](https://github.com/shahviraj/pgdgan), and [Liu et al.] (https://github.com/selwyn96/Quant_CS)



