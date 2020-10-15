######################Readme file for customized image front end######################
Note: Please keep the directory structure as indicated.
Note: since feature generation takes time we recommend using bash mode for running.

----------------------------------------------
CASE1: If you want to USE feature extractor.
run this command:

python main.py --base_path base_path_to_code_folder --train False --test False  --restore_models True --model_path where_models_are_saved_lodaed --model_json path_to_json --model_weight path_to_hdf_file --path_of_entire_image path_to_the_source_images

---------------------------------------------------
CASE2: If you want to CREATE the feature extractor from scratch

Step1: You need to create a dataset of 40*40 samples of car, truck, bus,background, in order to do that.

a) some samples from car, truck,bus, background are avaiable in "samples" folder.
b) You need to crop these samples with window size of 40*40 and stride size of 5 (We drived these numbers empricaly.). For that you can use crop.py

Stage 2) balance the dataset.
The generated crops are not balanced. So we try to balance it by using data augmentation and adding light effects. you can use uniform.py script for that.

Stage 3) split dataset:
We split the dataset to train/validation/test using split.py.
python main.py --base_path base_path_to_code_folder --train True --test True  --restore_models False --model_path where_models_are_saved_lodaed  --path_of_entire_image path_to_the_source_images


---------------------------------------------------
---------------------------------------------------
After generating features:
We extract the type of receiver from CoordVehiclesRxPerScene_s009.csv and discard non-relevant images. For that you can use pipeline.py.

