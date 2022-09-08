### Question 1.2
Every run uses an eval batch size of 100000 (averaging over at least 100 rollouts).

|Common parameters|Value|
|---|---|
|n_layers|2|
|size|64|
|train_batch_size|100|

|Environment|Expert|Average return|Std return|Learning rate|Loss|Train steps|
|---|---|---|---|---|---|---|
|Ant|4713.653|4689.482|421.529|5e-3|MSELoss|1000
|HalfCheetah|4205.778|4003.580|106.259|5e-3|MSELoss|1000|
|Hopper|3772.670|1627.715|324.464|1e-3|L1Loss|10000|
|Walker2d|5566.846|4045.609|1616.452|3e-3|L1Loss|10000|
