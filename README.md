# Deep learning on multimodal sensor data at the wireless edge for vehicular network

## Citing This Paper
Please cite the following paper if you intend to use this code for your research.

B. Salehi, G. Reus-Muns, D. Roy, Z. Wang, T. Jian, J. Dy, S. Ioannidis, and K. Chowdhury, “Deep Learning on Multimodal Sensor Data at the Wireless Edge for Vehicular Network,” IEEE Transactions on Vehicular Technology, vol. 71, no. 7, pp. 7639-7655, July 2022.

### Bibtex:

 `@article{salehi2022deep,
  title={Deep learning on multimodal sensor data at the wireless edge for vehicular network},
  author={Salehi, Batool and Reus-Muns, Guillem and Roy, Debashri and Wang, Zifeng and Jian, Tong and Dy, Jennifer and Ioannidis, Stratis and Chowdhury, Kaushik},
  journal={IEEE Transactions on Vehicular Technology},
  volume={71},
  number={7},
  pages={7639--7655},
  year={2022},
  publisher={IEEE}
}`
## Dataset:
We validate our approach on synthetic Raymobtime ([https://pages.github.com/](https://www.lasse.ufpa.br/raymobtime/)) and Real-world NEU dataset (https://genesys-lab.org/multimodal-fusion-nextg-v2x-communications). This repository is based on Raymobtime dataset. However, it can be applied to any other dataset (including NEU dataset), by changing the models to account for new input shapes.

We designed custom image feature extractors for Raymobtime dataset. To download the features please visit: (https://drive.google.com/drive/folders/1kRU8nmnvRj8DNU-VZYNqKnlZZNr84vu-?usp=sharing). The results in the paper are generated using the data mentioned in this link.

If you wish to generate the image features from scratch, please use the code available in create_image_feature directory. 

## Dataset and Pre-trained Models:
The pre-trained model are available here (https://drive.google.com/drive/folders/1kRU8nmnvRj8DNU-VZYNqKnlZZNr84vu-?usp=sharing).


## Train and Test:
Please use the command below to run the framework. For example for testing, set the argument train_or_test as "test" and use the pre-trained models.
python main.py  --id_gpu "gpu id to use" --data_folder "local path to dataset" --input coord img lidar --test_data_folder "local path to dataset" --model_folder ""local path to model folders"" --image_feature_to_use custom --train_or_test test



### Instruction for train and testing our pipleline

Case1: If you want to predict the output with our model:
- Go to test_front_end folder
- The models are saved in model_folder directory. You need to pass "test_model.json" and 'best_weights.coord_img_lidar_custom.h5'
- pass this command
`python test.py --test_data_folder itu_s009/baseline_data/ --input coord img lidar`
Note: We used three modalites with our customized image extracted features.



Case2: If you want to regenerate the weights.
1. We first train on single modalites. Run these commands.
For coordinates:

`python main.py --data_folder /home/batool/beam_selection_NU/baseline_data/ --test_data_folder /home/batool/beam_selection_NU/baseline_data/ --input coord --epochs 80`

For Images:
`python main.py --data_folder /home/batool/beam_selection_NU/baseline_data/ --test_data_folder /home/batool/beam_selection_NU/baseline_data/ --input img  --epochs 20 --image_feature_to_use custom`

For lidar:
`python main.py --data_folder /home/batool/beam_selection_NU/baseline_data/ --test_data_folder /home/batool/beam_selection_NU/baseline_data/ --input lidar  --epochs 50`


Using these command, you should have the json and weights of three modalites, OR, you can use our trained models. They are avaialbe in 'model_folder'

2. Then, we attach the fusion netwrok. We then reload the weights we learned from single modalites and train the model:
`python main.py --data_folder /home/batool/beam_selection_NU/baseline_data/ --test_data_folder /home/batool/beam_selection_NU/baseline_data/ --input coord img lidar  --epochs 60 --image_feature_to_use custom --restore_models True`

### Extra notes
In our code, we assumed that our extracted features are in a folder named 'input_img_custom' in baseline data directory.
