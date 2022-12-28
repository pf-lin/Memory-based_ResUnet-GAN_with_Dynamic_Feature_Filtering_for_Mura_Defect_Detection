# Memory-based ResUnet GAN with Dynamic Feature Filtering for Mura Defect Detection

### Server Connect
```py
ssh -L 9999:127.0.0.1:9999 sallylab@140.115.59.158
```
### Docker
```py
sudo docker exec -it levi_container bash
```
### Project Execution
```py
python mura_detector_memory.py
```

## Basic Setting
#### Memory module setting
```py
 mem_dim = 2000
 feature_num = 2048
 percentage = 10
```
<!-- ![image](https://user-images.githubusercontent.com/81354674/209820069-f93d6d55-86df-43bc-8499-387f9ce86e05.png)-->

#### Image setting
```py
 ORI_SIZE = (512, 512)
 IMG_H = 64
 IMG_W = 64
 IMG_C = 3  ## Change this to 1 for grayscale.
```
<!-- ![image](https://user-images.githubusercontent.com/81354674/209825098-c0c61c5a-14dd-40c2-8c42-68783b530b9c.png)-->

## Run Training
#### Edit code
```py
 """ run trainning process """
 train_images = glob(train_images_path)
 train_images_dataset = load_image_train(train_images, batch_size)
 train_images_dataset = train_images_dataset.cache().prefetch(buffer_size=AUTOTUNE)

 #*****training on, testing off*****
 run_trainning(resunetgan, train_images_dataset, num_epochs, path_gmodal, path_dmodal, logs_path, logs_file, name_model, steps, resume=resume_trainning)
```
<!-- ![image](https://user-images.githubusercontent.com/81354674/209825663-75249693-b96e-4dbe-876f-e954e1b5b336.png)-->  
\*`Testing` **部分註解**


## Run Testing

#### Model path setting
```py
 # test use
 path_gmodal = f"mura_data/{colour}/saved_model/8k/64_RGB_model_name_400_5000_g_model_best_xxx_0.xxx.h5"
 path_dmodal = f"mura_data/{colour}/saved_model/8k/64_RGB_model_name_crop_400_5000_d_model_best_xxx_0.xxx.h5"
```
<!--![image](https://user-images.githubusercontent.com/81354674/209825914-f85d78f4-11db-4ddf-b616-1273aafeeae5.png)-->

#### Edit code
```py
 """ run testing """
 class_names = ["normal_8k", "smura_8k"] # normal_8k = 0, smura_8k = 1
 test_dateset = load_image_test(test_data_path, class_names)
 resunetgan.testing(test_dateset, path_gmodal, path_dmodal, name_model)
```
<!--![image](https://user-images.githubusercontent.com/81354674/209828320-5eb99061-fa10-4292-9a46-d97186f5f1b3.png)-->  
\*`Training` **部分註解**
