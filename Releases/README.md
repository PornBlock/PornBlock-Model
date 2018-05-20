# Model 1.0.0
- This model contains most basic implementation (Required just to set up an end to end working pipeline)

# Dataset:
- Train: 74(-ve) + 106(+ve)
- Test: 28 (-ve) + 28 (+ve)
- Simply scraped few files from google
- For negative examples, sampled few files from IMAGENET data

# Model:
- Simple convolutional neural network. Used the script from Keras tutorial for cat and dog classifier

# Results
- loss: 0.2795 
- acc: 0.8951 
- val_loss: 1.2582 
- val_acc: 0.7250