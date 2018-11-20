# pix2pix-tensorflow
This repository is a Tensorflow implementation of the Isola's [Image-to-Image Tranaslation with Conditional Adversarial Networks, CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf). 

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619365-d285e190-85f2-11e8-8e52-9d53ddfc5653.png">
</p>

## Requirements
- tensorflow 1.8.0
- python 3.5.3  
- numpy 1.14.2  
- matplotlib 2.0.2  
- scipy 0.19.0

## Generated Results
- **facades dataset**  
**A to B**: from RGB image to generate label image  
**B to A**: from label image to generate RGB image
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42618142-0e1e6690-85ef-11e8-9aa0-e4bf88c3172d.png" width=700>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619115-165cb430-85f2-11e8-8b0c-bfd3470751b3.png" width=700>
</p>

- **maps dataset**
**A to B**: from statellite image to generate map image
**B to A**: from map image to generate statellite image
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619232-697bce44-85f2-11e8-95a8-8294be1489cd.png" width=700>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619276-8a6ac128-85f2-11e8-99c2-eb6743aa4458.png" width=700>
</p>

## Generator & Discriminator Structure
- **Generator structure**
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619487-2533caa6-85f3-11e8-9449-ada599622256.png" width=700>
</p>

- **Discriminator structure**
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619942-699a0e0c-85f4-11e8-97e0-b7403cd9abc7.png" width=400>
</p>

## Documentation
### Download Dataset
Download datasets (script borrowed from [torch code](https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh))
```
bash ./src/download_dataset.sh [dataset_name]
```
 - `dataset_name` supports `cityscapes`, `edges2handbags`, `edges2shoes`, `facades`, and `maps`.  
**Note**: our implementation has tested on `facades` and `maps` dataset only. But you can easily revise the code to run on other datasets.

### Directory Hierarchy
``` 
├── pix2pix
│   ├── src
│   │   ├── dataset.py
│   │   ├── download_dataset.sh
│   │   ├── main.py
│   │   ├── pix2pix.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
├── Data
│   ├── facades
│   └── maps
```  
**Note**: please put datasets on the correct position based on the Directory Hierarchy.

### Training pix2pix Model
Use `main.py` to train a pix2pix model. Example usage:

```
python main.py --dataset=facades --which_direction=0 --is_train=true
```
 - `gpu_index`: gpu index, default: `0`
 - `dataset`: dataset name for choice [`facades`|`maps`], default: `facades`
 - `which_direction`: AtoB (`0`) or BtoA (`1`), default: AtoB `0`
 - `batch_size`: batch size for one feed forward, default: `1`
 - `is_train`: 'training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.0002`
 - `beta1`: momentum term of Adam, default: `0.5`
 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `20000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_batch`: sample size for check generated image quality, default: `4`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`
 
### Evaluating pix2pix Model
Use `main.py` to evaluate a pix2pix model. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.  

### Citation
```
  @misc{chengbinjin2018pix2pix,
    author = {Cheng-Bin Jin},
    title = {pix2pix tensorflow},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/pix2pix-tensorflow}},
    note = {commit xxxxxxx}
  }
```
### Attributions/Thanks
- This project borrowed some code from [yenchenlin](https://github.com/yenchenlin/pix2pix-tensorflow) and [pix2pix official websit](https://phillipi.github.io/pix2pix/)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.

## Related Projects
- [DiscoGAN](https://github.com/ChengBinJin/DiscoGAN-TensorFlow)
