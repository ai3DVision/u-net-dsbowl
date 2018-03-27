### U-net

This is a rough solution for [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018), using U-net to predict the mask of nuclei. The preprocessing procedure for train just merges all of the masks of nuclei into one mask image, and data augmentation and any tricks are not applied in this project. The post-processing procedure picks up every connected region in the output mask as an instance. So this is a very simple implementation, and the best experiment result gets score [LB 0.321].  

##### Usage

```
#merge masks
python3 merge_masks.py
#train
python3 u_net.train.py
#test (the output is images)
python3 u_net.test.py
#test (the output is a CSV file)
python3 u_net.kaggle_csv.py
```

##### Results

![train](/home/papa/git/u_net/ukaggle/data/train.png)

![sample1](/home/papa/git/u_net/ukaggle/data/sample1.png)

![sample2](/home/papa/git/u_net/ukaggle/data/sample2.png)

![sample3](/home/papa/git/u_net/ukaggle/data/sample3.png)

![sample4](/home/papa/git/u_net/ukaggle/data/sample4.png)

