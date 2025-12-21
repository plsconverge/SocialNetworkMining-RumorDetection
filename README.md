# SocialNetworkMining-RumorDetection

Project for lecture Social Network Mining.

The project should be structured as 
```
root/
├── src/        # code folder
│   ├── data    # code for data loading and processing
│   ├── models    # models for machine learning or deep learning
│   ├── trainers    # trainers for models ** also provide results
│   └── helpers     # helper functions
│
├── data/       # data folder
│   └── CED_Dataset         # should be downloaded from github
│
├── visualization      # folder for visualization scripts
│       ├── visual_daily_distribution.py        # visualize distribution of reposts according to time periods in a day
│       ├── visual_interval.py          # visualize distribution of repost time intervals
│       ├── visual_propagation.py       # visualize the propagation diagram for a single graph
│       └── visual_timeline.py      # visualize the propagation timeline of a single graph
│
└── README.md
```

The CED dataset is contributed by thunlp group. 
It can be downloaded from https://github.com/thunlp/Chinese_Rumor_Dataset
