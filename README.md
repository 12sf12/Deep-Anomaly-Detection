# Deep-Anomaly-Detection

To run the first stage (Self-Supervision), you need to execute the following command:

```bash
exec python main.py --batch_size_train 512 --depth 40 --widen_factor 4 --number_of_workers 20
```

To run the second and third stages (Training and Evaluation), you need to execute the following command:

```bash
exec python main.py --batch_size_train 512 --depth 40 --widen_factor 4 --number_of_workers 20 --ssl_address 'path to ssl trained model' --class_id 1
```
