## Create Own Dataset
1. Create own `bddl` configure file for your new environment, like `create_new_tasks.ipynb`;

2. Put bddl file in path `libero/libero/init_files`, and run command to collect origin actions;
    ```bash
    python scripts/collect_demonstration_same.py --bddl-file libero/libero/bddl_files/file.bddl --robots Panda
    ```
    It will collect a raw hdf5 file in `demonstration_data/`.

3. Run the following command to collect whole data, image size should be $(256, 256)$;
    ```bash
    python scripts/create_dataset.py --demo-file demonstration_data/path/demo.hdf5 --use-camera-obs
    ```

4. Run to preprocess the data, delete noob action;
    ```bash
    python convert_raw_hdf5/regenerate_libero_dataset.py \
    --libero_task_suite libero_object \
    --libero_raw_data_dir libero/datasets/libero_object \
    --libero_target_dir libero/datasets/libero_object_no_noops
    ```