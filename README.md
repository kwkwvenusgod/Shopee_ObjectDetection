# Object Detection for shopee Dataset

## Train process
This project does not contain any training dataset. The model trained model is based on 
[DeepFashion](https://drive.google.com/drive/folders/0B7EVK8r0v71pRXllRUdQcC1zTHc). If you want to fine-tune
the model, please configure `train_dataset_config.json`  with the proper dataset path in value field.

Then execute
```bash
 python train_for_shopee.py 
```
## Test process
The file `test_for_shopee.py` is written for shopee data. Please excute
```bash
    python test_for_shopee.py -p <path-to-your-test-image-folder>
```
### model
The final model is over 100mb, so github does not allowed to push, so the model is put in [my google drive](1P2WJhWiBI7zw4qq8E43cCVT1QPlkr-qW)

1. You can download it in the [url](https://drive.google.com/open?id=1o9d3eGp0z_brNnHO9_XNyRNsJMShHRTa) and put it in `model_output` folder then execute the above python script
2. You can either directly run the script and then `chrome` will pop out which will need your authentication for further downloading.


<embed src="/report/Object-Detection-for-Shopee-Data.pdf" width="750px" height="800px"/>

# Detailed Explanation
Slides are prepared for the explanation of my solution. It is int the `report` folder. For more details, you may refer the slides instead.
