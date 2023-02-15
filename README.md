# Deep learning on multimodal sensor data at the wireless edge for vehicular network

### Citing This Paper
Please cite the following paper if you intend to use this code for your research.

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
